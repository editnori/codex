use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use codex_common::CliConfigOverrides;
use codex_common::SandboxModeCliArg;
use codex_exec::Cli as ExecCli;
use serde::Deserialize;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Default base lexicon path (relative to the working directory) if --dict-script is not provided.
const DEFAULT_DICT_SCRIPT: &str = "smartsig_sig_lexicon_ext.script";

/// Default overlay lexicon path (relative to the working directory) if --overlay-script is not provided.
const DEFAULT_OVERLAY_SCRIPT: &str = "smartsig_sig_lexicon_overlay.script";

/// JSON Schema for the model's output (lexicon mention adjudication).
///
/// The model outputs ONLY mention_id -> concept_id selections (plus confidence).
/// Offsets/spans are handled deterministically by Rust.
const DECISION_SCHEMA_JSON: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SigSpanDecisions",
  "type": "object",
  "additionalProperties": false,
  "required": ["doc_id", "decisions", "errors"],
  "properties": {
    "doc_id": { "type": ["string", "null"] },
    "decisions": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["mention_id", "concept_id", "confidence"],
        "properties": {
          "mention_id": { "type": "string", "minLength": 1 },
          "concept_id": { "type": "string", "minLength": 1 },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["code", "message"],
        "properties": {
          "code": { "type": "string", "minLength": 1 },
          "message": { "type": "string", "minLength": 1 }
        }
      }
    }
  }
}"#;

/// JSON Schema for canonical SIG rendering (LLM).
///
/// The model outputs ONLY the canonical string (no spans/offsets).
const CANONICAL_SCHEMA_JSON: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SigCanonical",
  "type": "object",
  "additionalProperties": false,
  "required": ["doc_id", "canonical", "errors"],
  "properties": {
    "doc_id": { "type": ["string", "null"] },
    "canonical": {
      "type": "object",
      "additionalProperties": false,
      "required": ["text", "confidence", "extra_notes"],
      "properties": {
        "text": { "type": "string", "minLength": 1 },
        "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
        "extra_notes": {
          "type": "array",
          "items": { "type": "string", "minLength": 1 }
        }
      }
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["code", "message"],
        "properties": {
          "code": { "type": "string", "minLength": 1 },
          "message": { "type": "string", "minLength": 1 }
        }
      }
    }
  }
}"#;


/// JSON Schema for the model's output (learning unknown residuals -> lexicon patches).
///
/// The model outputs ONLY patch suggestions. Offsets/spans are handled by Rust.
/// Patches are validated in Rust before being written to the overlay.
const LEARN_SCHEMA_JSON: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "SigLexiconPatches",
  "type": "object",
  "additionalProperties": false,
  "required": ["patches", "errors"],
  "properties": {
    "patches": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["span_id", "term", "category", "concept_id", "confidence"],
        "properties": {
          "span_id": { "type": "string", "minLength": 1 },
          "term": { "type": "string", "minLength": 1 },
          "category": { "type": "string", "minLength": 1 },
          "concept_id": { "type": "string", "minLength": 1 },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    },
    "errors": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["code", "message"],
        "properties": {
          "code": { "type": "string", "minLength": 1 },
          "message": { "type": "string", "minLength": 1 }
        }
      }
    }
  }
}"#;

#[derive(Debug, Parser)]
pub struct SigConceptExtractCommand {
    /// SIG text as a literal string. Conflicts with --input.
    #[arg(long = "sig", value_name = "SIG", conflicts_with = "input")]
    pub sig: Option<String>,

    /// Path to input text file. Use '-' to read from stdin.
    #[arg(long = "input", short = 'i', value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub input: Option<PathBuf>,

    /// Path to an HSQLDB .script base lexicon file (SmartSig-derived).
    #[arg(long = "dict-script", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub dict_script: Option<PathBuf>,

    /// Optional overlay lexicon (.script). Loaded in addition to the base lexicon.
    ///
    /// If omitted, the extractor will load DEFAULT_OVERLAY_SCRIPT if it exists.
    #[arg(long = "overlay-script", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub overlay_script: Option<PathBuf>,

    /// Max candidates per mention span (per category).
    #[arg(long = "max-candidates", default_value_t = 8)]
    pub max_candidates: usize,

    /// Max mention spans to include in the prompt.
    #[arg(long = "max-mentions", default_value_t = 250)]
    pub max_mentions: usize,

    /// Minimum lexicon term length to match.
    ///
    /// NOTE: SIGs contain many short-but-meaningful terms (po, prn, qd, q, mg, ml, etc).
    /// This defaults to 1 to allow high-yield 1-char markers like "q".
    #[arg(long = "min-term-length", default_value_t = 1)]
    pub min_term_length: usize,

    /// Include candidates on each span in the final JSON output (debug/training).
    #[arg(long = "include-candidates", default_value_t = false)]
    pub include_candidates: bool,

    /// Emit rejected lexicon matches as spans with normalized.kind=REJECTED.
    /// If false, rejected spans are omitted.
    #[arg(long = "include-rejected", default_value_t = true)]
    pub include_rejected: bool,

    /// Run deterministic auto-span detection (numeric/date/icd).
    #[arg(long = "auto-spans", default_value_t = true)]
    pub auto_spans: bool,

    /// Attempt to learn unknown residual spans by mapping them to existing canonical concepts,
    /// then append alias rows to the overlay script.
    ///
    /// This is how you get "real-time" correction without requiring the model to output offsets.
    #[arg(long = "learn-unknowns", default_value_t = true, action = clap::ArgAction::Set)]
    pub learn_unknowns: bool,

    /// Comma-delimited list of categories to allow learning into (e.g., dx,route,freq).
    ///
    /// Default is dx only (safe + high yield).
    #[arg(long = "learn-categories", value_delimiter = ',', default_value = "dx,route,device,time_of_day,timing_event,day_of_week,site,aux_warning,aux_insurance,aux_drug,aux_auxiliary,aux_provider,aux_misc")]
    pub learn_categories: Vec<String>,

    /// Minimum confidence threshold to apply a learned alias patch.
    #[arg(long = "learn-min-confidence", default_value_t = 0.95)]
    pub learn_min_confidence: f64,

    /// Minimum confidence threshold to apply a learned TAG (category-only patch when concept_id="__NO_MATCH__").
    ///
    /// This controls how aggressively the model can classify residual spans into non-core buckets
    /// like aux_warning / aux_auxiliary / aux_provider / aux_misc, or into dx when no code match exists.
    #[arg(long = "learn-tag-min-confidence", default_value_t = 0.90)]
    pub learn_tag_min_confidence: f64,

    /// If set, the extractor will NOT write any patches to disk; it will only report them in output.
    #[arg(long = "learn-dry-run", default_value_t = false)]
    pub learn_dry_run: bool,

    /// Output a canonical SIG string.
    ///
    /// Default is true. If false, `canonical.text` falls back to the original SIG.
    ///
    /// Canonical rendering uses the LLM (strict JSON schema) and is constrained to the
    /// already-extracted normalized spans (no hallucinated clinical content).
    #[arg(long = "canonical", default_value_t = true, action = clap::ArgAction::Set)]
    pub canonical: bool,

    /// Number of Unicode-scalar characters of context to include on each side of a mention.
    ///
    /// This is inserted into CANDIDATES_JSON as a `window` string like:
    ///   "...left...[[MENTION]]...right..."
    ///
    /// It makes disambiguation robust when a mention text appears multiple times in a SIG.
    #[arg(long = "context-window", default_value_t = 48)]
    pub context_window: usize,

    /// Optional output file for the final JSON result.
    #[arg(long = "output", short = 'o', value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub output: Option<PathBuf>,

    /// Output all lexicon candidates without LLM adjudication.
    #[arg(long = "all-candidates", alias = "all", default_value_t = false)]
    pub all_candidates: bool,

    /// Optional output schema override for decision stage. If omitted, uses the built-in schema.
    #[arg(long = "schema", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    pub schema: Option<PathBuf>,

    /// Model the agent should use.
    #[arg(long, short = 'm')]
    pub model: Option<String>,

    /// Use open-source provider.
    #[arg(long = "oss", default_value_t = false)]
    pub oss: bool,

    /// Specify which local provider to use (lmstudio or ollama).
    #[arg(long = "local-provider")]
    pub oss_provider: Option<String>,

    /// Select the sandbox policy to use when executing model-generated shell commands.
    #[arg(long = "sandbox", short = 's', value_enum)]
    pub sandbox_mode: Option<SandboxModeCliArg>,

    /// Configuration profile from config.toml to specify default options.
    #[arg(long = "profile", short = 'p')]
    pub config_profile: Option<String>,

    /// Convenience alias for low-friction sandboxed automatic execution.
    #[arg(long = "full-auto", default_value_t = false)]
    pub full_auto: bool,

    /// Skip all confirmation prompts and execute commands without sandboxing.
    #[arg(
        long = "dangerously-bypass-approvals-and-sandbox",
        alias = "yolo",
        default_value_t = false,
        conflicts_with = "full_auto"
    )]
    pub dangerously_bypass_approvals_and_sandbox: bool,

    /// Tell the agent to use the specified directory as its working root.
    #[arg(long = "cd", short = 'C', value_name = "DIR", value_hint = clap::ValueHint::DirPath)]
    pub cwd: Option<PathBuf>,

    /// Allow running Codex outside a Git repository.
    #[arg(long = "skip-git-repo-check", default_value_t = false)]
    pub skip_git_repo_check: bool,

    /// Additional directories that should be writable alongside the primary workspace.
    #[arg(long = "add-dir", value_name = "DIR", value_hint = clap::ValueHint::DirPath)]
    pub add_dir: Vec<PathBuf>,

    /// Print events to stdout as JSONL.
    #[arg(long = "json", alias = "experimental-json", default_value_t = false)]
    pub json: bool,

    #[clap(skip)]
    pub config_overrides: CliConfigOverrides,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MentionKey {
    start: usize, // byte offsets
    end: usize,   // byte offsets
    category: String,
}

#[derive(Debug)]
struct MentionTemp {
    start: usize,
    end: usize,
    category: String,
    text: String,
    candidates: HashMap<String, CandidateTemp>,
}

#[derive(Debug)]
struct CandidateTemp {
    concept_id: String,
    matched_term: String,
    score: usize,
}

#[derive(Debug, Default)]
struct LexiconMetadata {
    prefterm: HashMap<String, String>,
    tui: HashMap<String, Vec<String>>,
}

#[derive(Debug)]
struct Mention {
    mention_id: String,
    start_byte: usize,
    end_byte: usize,
    start_char: usize,
    end_char: usize,
    category: String,
    text: String,
    candidates: Vec<Candidate>,
}

#[derive(Serialize, Clone, Debug)]
struct Candidate {
    concept_id: String,
    preferred: String,
    matched_term: String,
    tui: Vec<String>,
    score: usize,
}

#[derive(Serialize)]
struct CandidatePayload {
    sig: String,
    mentions: Vec<MentionCandidates>,
}

#[derive(Serialize)]
struct MentionCandidates {
    mention_id: String,
    category: String,
    start_char: usize,
    end_char: usize,
    text: String,
    window: String,
    candidates: Vec<CandidateForModel>,
}


#[derive(Serialize)]
struct CandidateForModel {
    concept_id: String,
    preferred: String,
}

/// Decision model output
#[derive(Debug, Deserialize)]
struct DecisionOutput {
    doc_id: Option<String>,
    decisions: Vec<Decision>,
    errors: Vec<DecisionError>,
}

#[derive(Debug, Deserialize)]
struct Decision {
    mention_id: String,
    concept_id: String,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct DecisionError {
    code: String,
    message: String,
}

/// Learn model output
#[derive(Debug, Deserialize)]
struct LearnOutput {
    patches: Vec<LearnPatch>,
    errors: Vec<DecisionError>,
}

#[derive(Debug, Deserialize, Clone)]
struct LearnPatch {
    span_id: String,
    term: String,
    category: String,
    concept_id: String,
    confidence: f64,
}

/// Canonical concept index entry (for learn stage)
#[derive(Debug, Clone)]
struct CanonicalConcept {
    concept_id: String,
    preferred: String,
}

/// Final output
#[derive(Serialize)]
struct SigExtractionOutput {
    doc_id: Option<String>,
    sig: String,
    spans: Vec<SigSpanOutput>,
    errors: Vec<OutputError>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    patches: Vec<AppliedPatch>,
    #[serde(skip_serializing_if = "Option::is_none")]
    canonical: Option<CanonicalSig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    structure: Option<SigStructure>,
}

#[derive(Serialize)]
struct AppliedPatch {
    span_id: String,
    term: String,
    category: String,
    concept_id: String,
    preferred: String,
    confidence: f64,
    applied: bool,
    reason: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct OutputError {
    code: String,
    message: String,
}

#[derive(Serialize, Clone)]
struct SigSpanOutput {
    span_id: String,
    category: String,
    role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,
    span: SpanOut,
    normalized: NormalizedOut,
    matched_term: String,
    confidence: f64,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidates: Option<Vec<CandidateDebug>>,
}

#[derive(Serialize, Clone)]
struct CandidateDebug {
    concept_id: String,
    preferred: String,
    score: usize,
    matched_term: String,
}

#[derive(Debug, Serialize, Clone)]
struct SpanOut {
    start: usize,
    end: usize,
    text: String,
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
enum NormalizedOut {
    Coded { concept_id: String, preferred: String },
    Rejected { reason: String },
    Numeric { number_kind: String, normalized: String },
    Date { iso: String },
    Icd { code: String },
    Text { label: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanRole {
    Core,
    NonCore,
    Residual,
}

#[derive(Serialize, Deserialize, Clone)]
struct CanonicalSig {
    text: String,
    confidence: f64,
    source: String,
    extra_notes: Vec<String>,
    spans: Vec<CanonicalSpan>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CanonicalSpan {
    category: String,
    start: usize,
    end: usize,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    concept_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preferred: Option<String>,
}


#[derive(Debug, Clone, Serialize)]
struct SigStructure {
    /// "SIMPLE" or "COMPLEX"
    kind: String,
    /// Heuristic flags describing complexity / data-quality risks
    flags: Vec<String>,
    /// Step/Clause segmentation of the original SIG
    steps: Vec<SigStep>,
}

#[derive(Debug, Clone, Serialize)]
struct SigStep {
    step_id: String,
    /// "INSTRUCTION" | "REPEAT" | "CONSTRAINT" | "NOTE"
    kind: String,
    span: SpanOut,
    /// All span_ids that fall within this step range (including residual/non-core)
    span_ids: Vec<String>,
    /// Convenience lists so downstream consumers can quickly focus on core vs non-core
    core_span_ids: Vec<String>,
    noncore_span_ids: Vec<String>,
    residual_span_ids: Vec<String>,
}


pub async fn run(args: SigConceptExtractCommand, codex_linux_sandbox_exe: Option<PathBuf>) -> Result<()> {
    let started = Instant::now();
    let sig_text = read_input_text(args.sig.as_ref(), args.input.as_ref())?;
    let dict_scripts = resolve_dict_scripts(args.dict_script.as_ref(), args.overlay_script.as_ref())?;

    eprintln!("SIG length: {} chars", sig_text.chars().count());
    eprintln!("Lexicons:");
    for p in &dict_scripts {
        eprintln!("  - {}", p.display());
    }

    let byte_to_char = build_byte_to_char_map(&sig_text);

    // 1) Lexicon scan -> mention candidates
    let (mentions_map, concept_ids) = scan_lexicon_terms(&dict_scripts, &sig_text, args.min_term_length, args.max_mentions)?;
    eprintln!(
        "Lexicon scan complete in {} ms ({} mention spans, {} concept_ids).",
        started.elapsed().as_millis(),
        mentions_map.len(),
        concept_ids.len()
    );

    let meta = load_lexicon_metadata(&dict_scripts, &concept_ids)?;
    let mut mentions = build_mentions(mentions_map, &meta, &byte_to_char, args.max_candidates, args.max_mentions);

    // Prune contained overlaps within the same category (e.g., "as needed" within "as needed for").
    mentions = prune_contained_mentions(mentions);

    if mentions.is_empty() {
        let out = SigExtractionOutput {
            doc_id: None,
            sig: sig_text.clone(),
            spans: vec![],
            errors: vec![OutputError {
                code: "no_lexicon_matches".to_string(),
                message: "No lexicon matches found in the SIG.".to_string(),
            }],
            patches: vec![],
            canonical: Some(CanonicalSig {
                text: sig_text.clone(),
                confidence: 0.0,
                source: "fallback".to_string(),
                extra_notes: vec![],
                spans: vec![],
            }),
            structure: Some(compute_sig_structure(&sig_text, &[])),
        };
        write_output_json(&out, args.output.as_ref())?;
        return Ok(());
    }

    // Add a per-mention explicit reject option, so the model can always choose deterministically.
    for mention in &mut mentions {
        mention.candidates.push(Candidate {
            concept_id: "__REJECT__".to_string(),
            preferred: "Reject (not a real match)".to_string(),
            matched_term: mention.text.clone(),
            tui: vec![],
            score: 0,
        });
    }

    // 2) Deterministic auto spans (numeric/date/icd + residual later)
    let (auto_spans, auto_cov) = if args.auto_spans {
        detect_auto_spans(&sig_text, &byte_to_char)
    } else {
        (vec![], vec![])
    };

    // If requested, emit candidates only and skip LLM.
    if args.all_candidates {
        let mut spans = Vec::new();
        for mention in &mentions {
            spans.push(span_from_mention_candidates(mention, args.include_candidates));
        }

        spans.extend(compute_residual_spans(&sig_text, &byte_to_char, &mentions, &auto_cov));
        spans.extend(auto_spans);

        let mut out = SigExtractionOutput {
            doc_id: None,
            sig: sig_text,
            spans,
            errors: vec![],
            patches: vec![],
            canonical: None,
            structure: None,
        };

        // Canonical is always present.
        // In --all-candidates mode we have not adjudicated to a single ontology concept per span,
        // so we cannot safely canonicalize. We fall back to the raw SIG.
        out.canonical = Some(CanonicalSig {
            text: out.sig.clone(),
            confidence: 0.0,
            source: "fallback".to_string(),
            extra_notes: vec![],
            spans: vec![],
        });

        out.structure = Some(compute_sig_structure(&out.sig, &out.spans));

        write_output_json(&out, args.output.as_ref())?;
        return Ok(());
    }

    // 3) LLM adjudication: choose concept_id per mention_id (or __REJECT__)
    let payload = build_candidate_payload(&sig_text, &mentions, args.context_window);
    let payload_json = serde_json::to_string(&payload)?;
    let prompt = build_decision_prompt(&sig_text, &payload_json);

    let decision_schema = match args.schema.as_ref() {
        Some(path) => path.clone(),
        None => write_schema_tempfile("codex-sig-decision-schema", DECISION_SCHEMA_JSON)?,
    };

    let mut llm_output_path = std::env::temp_dir();
    llm_output_path.push(format!("codex-sig-decisions-{}.json", std::process::id()));

    let mut exec_cli = ExecCli::try_parse_from(["codex", "exec"])?;
    exec_cli.prompt = Some(prompt);
    exec_cli.output_schema = Some(decision_schema);
    exec_cli.last_message_file = Some(llm_output_path.clone());
    exec_cli.model = args.model.clone();
    exec_cli.oss = args.oss;
    exec_cli.oss_provider = args.oss_provider.clone();
    exec_cli.sandbox_mode = args.sandbox_mode;
    exec_cli.config_profile = args.config_profile.clone();
    exec_cli.full_auto = args.full_auto;
    exec_cli.dangerously_bypass_approvals_and_sandbox = args.dangerously_bypass_approvals_and_sandbox;
    exec_cli.cwd = args.cwd.clone();
    exec_cli.skip_git_repo_check = args.skip_git_repo_check;
    exec_cli.add_dir = args.add_dir.clone();
    exec_cli.json = false;
    exec_cli.config_overrides = args.config_overrides.clone();

    run_exec_silently(exec_cli, codex_linux_sandbox_exe.clone()).await?;

    let decisions_raw = fs::read_to_string(&llm_output_path)
        .with_context(|| format!("Failed to read decisions from {}", llm_output_path.display()))?;
    if decisions_raw.trim().is_empty() {
        anyhow::bail!("Decision output was empty; no spans were produced.");
    }

    let decision_output: DecisionOutput =
        serde_json::from_str(&decisions_raw).context("Failed to parse model decisions JSON")?;

    // 4) Build spans from decisions
    let mut errors: Vec<OutputError> = decision_output
        .errors
        .into_iter()
        .map(|e| OutputError { code: e.code, message: e.message })
        .collect();

    let decisions_map = build_decisions_map(&decision_output.decisions, &mentions, &mut errors);

    let mut spans: Vec<SigSpanOutput> = Vec::new();
    for mention in &mentions {
        if let Some(span) = build_span_from_decision(
            mention,
            decisions_map.get(mention.mention_id.as_str()).copied(),
            args.include_candidates,
            args.include_rejected,
            &mut errors,
        ) {
            spans.push(span);
        }
    }

    spans.extend(compute_residual_spans(&sig_text, &byte_to_char, &mentions, &auto_cov));
    spans.extend(auto_spans);

    // 5) Learn unknown residuals (optional)
    let mut applied_patches: Vec<AppliedPatch> = Vec::new();
    if args.learn_unknowns {
        let overlay_path = resolve_overlay_script_path(args.overlay_script.as_ref())?;
        let learn_categories: HashSet<String> =
            args.learn_categories.iter().map(|s| s.trim().to_string()).collect();

        let unknowns = collect_unknown_residuals(&spans, &learn_categories);
        if !unknowns.is_empty() {
            let canonical_index = load_canonical_index(&dict_scripts, &learn_categories)?;
            let learn_prompt = build_learn_prompt(&sig_text, &unknowns, &canonical_index, &learn_categories);

            let learn_schema = write_schema_tempfile("codex-sig-learn-schema", LEARN_SCHEMA_JSON)?;
            let mut learn_out_path = std::env::temp_dir();
            learn_out_path.push(format!("codex-sig-lexicon-patches-{}.json", std::process::id()));

            let mut learn_cli = ExecCli::try_parse_from(["codex", "exec"])?;
            learn_cli.prompt = Some(learn_prompt);
            learn_cli.output_schema = Some(learn_schema);
            learn_cli.last_message_file = Some(learn_out_path.clone());
            learn_cli.model = args.model.clone();
            learn_cli.oss = args.oss;
            learn_cli.oss_provider = args.oss_provider.clone();
            learn_cli.sandbox_mode = args.sandbox_mode;
            learn_cli.config_profile = args.config_profile.clone();
            learn_cli.full_auto = args.full_auto;
            learn_cli.dangerously_bypass_approvals_and_sandbox = args.dangerously_bypass_approvals_and_sandbox;
            learn_cli.cwd = args.cwd.clone();
            learn_cli.skip_git_repo_check = args.skip_git_repo_check;
            learn_cli.add_dir = args.add_dir.clone();
            learn_cli.json = false;
            learn_cli.config_overrides = args.config_overrides.clone();

            run_exec_silently(learn_cli, codex_linux_sandbox_exe.clone()).await?;

            let learn_raw = fs::read_to_string(&learn_out_path)
                .with_context(|| format!("Failed to read learn output from {}", learn_out_path.display()))?;
            if !learn_raw.trim().is_empty() {
                let learn_out: LearnOutput =
                    serde_json::from_str(&learn_raw).context("Failed to parse learn output JSON")?;

                // Carry learn errors into top-level errors
                for e in learn_out.errors {
                    errors.push(OutputError { code: e.code, message: e.message });
                }

                let (new_spans, patches) = apply_learn_patches(
                    &sig_text,
                    &byte_to_char,
                    &spans,
                    &learn_out.patches,
                    &canonical_index,
                    &learn_categories,
                    args.learn_min_confidence,
                    args.learn_tag_min_confidence,
                    &overlay_path,
                    args.learn_dry_run,
                )?;

                spans = new_spans;
                applied_patches = patches;
            }
        }
    }

    // Stable sort output by start position then by category then by length (descending).
    spans.sort_by(|a, b| {
        a.span
            .start
            .cmp(&b.span.start)
            .then_with(|| a.category.cmp(&b.category))
            .then_with(|| (b.span.end - b.span.start).cmp(&(a.span.end - a.span.start)))
    });

    apply_semantic_tags(&mut spans);

    let structure = compute_sig_structure(&sig_text, &spans);

    let mut out = SigExtractionOutput {
        doc_id: decision_output.doc_id,
        sig: sig_text.clone(),
        spans,
        errors,
        patches: applied_patches,
        canonical: None,
        structure: Some(structure),
    };

    // 6) Canonical (always present)
    //
    // We default to LLM canonicalization because:
    // - It can reorder clauses grammatically (dose/route/frequency/prn/limits)
    // - It can drop obvious boilerplate/residuals safely
    // - It preserves meaning using the already-normalized spans (no new clinical facts)
    //
    // If canonicalization fails for any reason, we fall back to the conservative v1 deterministic renderer,
    // and if that fails, we fall back to the raw SIG.
    let canonical_sig = if args.canonical {
        match llm_render_canonical(&out.sig, &out.spans, &args, codex_linux_sandbox_exe.clone()).await {
            Ok(canon) => canon,
            Err(e) => {
                out.errors.push(OutputError {
                    code: "canonical_llm_failed".to_string(),
                    message: format!("Canonical LLM rendering failed; falling back: {e}"),
                });
                render_canonical_simple(&out.spans).unwrap_or_else(|| CanonicalSig {
                    text: out.sig.clone(),
                    confidence: 0.0,
                    source: "fallback".to_string(),
                    extra_notes: vec![],
                    spans: vec![],
                })
            }
        }
    } else {
        CanonicalSig {
            text: out.sig.clone(),
            confidence: 0.0,
            source: "fallback".to_string(),
            extra_notes: vec![],
            spans: vec![],
        }
    };

    out.canonical = Some(canonical_sig);

    write_output_json(&out, args.output.as_ref())?;
    Ok(())
}

fn build_candidate_payload(sig_text: &str, mentions: &[Mention], window_chars: usize) -> CandidatePayload {
    let char_to_byte = build_char_to_byte_map(sig_text);
    let char_len = char_to_byte.len().saturating_sub(1);

    let mentions = mentions
        .iter()
        .map(|m| {
            let left_start = m.start_char.saturating_sub(window_chars);
            let right_end = (m.end_char + window_chars).min(char_len);

            let left = slice_chars(sig_text, &char_to_byte, left_start, m.start_char);
            let right = slice_chars(sig_text, &char_to_byte, m.end_char, right_end);

            let window = format!("{left}[[{text}]]{right}", left = left, text = m.text, right = right);

            MentionCandidates {
                mention_id: m.mention_id.clone(),
                category: m.category.clone(),
                start_char: m.start_char,
                end_char: m.end_char,
                text: m.text.clone(),
                window,
                candidates: m
                    .candidates
                    .iter()
                    .map(|c| CandidateForModel {
                        concept_id: c.concept_id.clone(),
                        preferred: c.preferred.clone(),
                    })
                    .collect(),
            }
        })
        .collect();

    CandidatePayload {
        sig: sig_text.to_string(),
        mentions,
    }
}


fn build_decision_prompt(sig_text: &str, candidates_json: &str) -> String {
    format!(
        "You are a strict SIG span normalizer.\n\
Your job is ONLY to choose the best canonical concept for each candidate span.\n\
\n\
Rules:\n\
1) Output ONLY JSON that matches the provided JSON Schema.\n\
2) For EVERY mention_id in CANDIDATES_JSON, output EXACTLY ONE decision.\n\
3) Your chosen concept_id MUST be one of that mention's candidates.\n\
4) If the span is not a valid match, choose concept_id=\"__REJECT__\".\n\
5) Do NOT output any spans, offsets, token indices, or free text.\n\
6) You MUST use the full SIG context to disambiguate.\n\
   - Each mention has start_char/end_char and a `window` field with the mention bracketed as [[...]].\n\
   - Prefer decisions that make the overall SIG semantically consistent (dose/route/frequency/prn/limits).\n\
7) Be conservative. If unsure, reject.\n\
\n\
SIG (full context):\n\
{sig}\n\
\n\
CANDIDATES_JSON:\n\
{candidates_json}\n",
        sig = sig_text,
        candidates_json = candidates_json
    )
}


fn build_learn_prompt(
    sig_text: &str,
    unknowns: &[SigSpanOutput],
    canonical_index: &HashMap<String, Vec<CanonicalConcept>>,
    allowed_categories: &HashSet<String>,
) -> String {
    // Provide a compact canonical list per category. This is intentionally explicit so the model
    // outputs only known concept_ids (or "__NO_MATCH__").
    let mut canon_map: HashMap<String, Vec<HashMap<&'static str, String>>> = HashMap::new();
    for (cat, concepts) in canonical_index {
        let mut vec = Vec::new();
        for c in concepts {
            let mut o = HashMap::new();
            o.insert("concept_id", c.concept_id.clone());
            o.insert("preferred", c.preferred.clone());
            vec.push(o);
        }
        canon_map.insert(cat.clone(), vec);
    }

    let canon_json = serde_json::to_string(&canon_map).unwrap_or_else(|_| "{}".to_string());
    let unknown_json = serde_json::to_string(
        &unknowns
            .iter()
            .map(|s| {
                serde_json::json!({
                    "span_id": s.span_id,
                    "text": s.span.text,
                    "start": s.span.start,
                    "end": s.span.end
                })
            })
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".to_string());

    // Categories allowed for classification (including tag-only categories like aux_warning).
    let mut allowed: Vec<String> = allowed_categories.iter().cloned().collect();
    allowed.sort();
    let allowed_json = serde_json::to_string(&allowed).unwrap_or_else(|_| "[]".to_string());

    // Categories that have a canonical list (i.e., we can map to a concept_id).
    let mut canonical_cats: Vec<String> = canonical_index.keys().cloned().collect();
    canonical_cats.sort();
    let canonical_cats_json = serde_json::to_string(&canonical_cats).unwrap_or_else(|_| "[]".to_string());

    // Tag-only categories are those allowed but not present in CANONICALS_JSON.
    let mut tag_only: Vec<String> = allowed
        .iter()
        .filter(|c| !canonical_index.contains_key(*c))
        .cloned()
        .collect();
    tag_only.sort();
    let tag_only_json = serde_json::to_string(&tag_only).unwrap_or_else(|_| "[]".to_string());

    format!(
        "You are helping expand a SIG extractor by labeling UNKNOWN residual spans.
\

\
You have two valid actions per UNKNOWN span:
\
A) Alias learning (preferred): map the span to an EXISTING canonical concept_id (from CANONICALS_JSON), OR
\
B) Tag-only classification: assign the span to a SAFE category bucket when no concept_id applies.
\

\
Rules:
\
1) Output ONLY JSON that matches the provided JSON Schema.
\
2) Each patch.category MUST be one of the strings in ALLOWED_CATEGORIES_JSON.
\
3) If patch.category is in TAG_ONLY_CATEGORIES_JSON, you MUST set patch.concept_id = \"__NO_MATCH__\".
\
4) If patch.category is in CANONICAL_CATEGORIES_JSON, you may either:
\
   a) choose patch.concept_id from CANONICALS_JSON[patch.category], OR
\
   b) if no concept_id is appropriate but the CATEGORY is still clear (e.g., an unmapped dx abbreviation),
\
      set patch.concept_id = \"__NO_MATCH__\" and keep confidence moderate-to-high.
\
5) Do NOT invent new concept_ids.
\
6) patch.term MUST be the exact surface text from UNKNOWN_SPANS_JSON (do not paraphrase).
\
7) Prefer aux_* buckets for non-core content (warnings, provider notes, misc free text).
\

\
SIG:
\
{sig}
\

\
UNKNOWN_SPANS_JSON:
\
{unknowns}
\

\
ALLOWED_CATEGORIES_JSON:
\
{allowed}
\

\
TAG_ONLY_CATEGORIES_JSON:
\
{tag_only}
\

\
CANONICAL_CATEGORIES_JSON:
\
{canon_cats}
\

\
CANONICALS_JSON:
\
{canon}
",
        sig = sig_text,
        unknowns = unknown_json,
        allowed = allowed_json,
        tag_only = tag_only_json,
        canon_cats = canonical_cats_json,
        canon = canon_json
    )
}


fn collect_unknown_residuals(spans: &[SigSpanOutput], _learn_categories: &HashSet<String>) -> Vec<SigSpanOutput> {
    spans
        .iter()
        .filter(|s| s.category == "residual_text")
        .filter(|s| matches!(s.normalized, NormalizedOut::Text { .. }))
        .filter(|s| match &s.normalized {
            NormalizedOut::Text { label } => label != "STOPWORD",
            _ => true,
        })
        .filter(|s| {
            // Heuristic: only learn meaningful alphanumeric-ish tokens.
            //
            // We explicitly exclude common grammar/connector tokens (and/or/by/for/etc) and
            // numeric words (one/two/half/etc) to prevent garbage patches.
            let t = s.span.text.trim();
            if t.is_empty() {
                return false;
            }
            let lower = t.to_ascii_lowercase();
            if lower.len() < 2 {
                return false;
            }
            if matches!(
                lower.as_str(),
                "and"
                    | "or"
                    | "to"
                    | "for"
                    | "by"
                    | "a"
                    | "an"
                    | "the"
                    | "do"
                    | "not"
                    | "of"
                    | "in"
                    | "on"
                    | "at"
                    | "per"
                    | "as"
                    | "if"
                    | "then"
                    | "with"
                    | "without"
            ) {
                return false;
            }
            if number_word_to_numeric(&lower).is_some() {
                return false;
            }
            if lower.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            lower.chars().any(|c| c.is_ascii_alphanumeric())
        })
        .cloned()
        .collect()
}


fn apply_learn_patches(
    sig_text: &str,
    _byte_to_char: &[usize],
    spans: &[SigSpanOutput],
    patches: &[LearnPatch],
    canonical_index: &HashMap<String, Vec<CanonicalConcept>>,
    allowed_categories: &HashSet<String>,
    min_confidence: f64,
    tag_min_confidence: f64,
    overlay_path: &PathBuf,
    dry_run: bool,
) -> Result<(Vec<SigSpanOutput>, Vec<AppliedPatch>)> {
    let mut out_spans: Vec<SigSpanOutput> = spans.to_vec();
    let mut applied: Vec<AppliedPatch> = Vec::new();

    // Build lookup: concept_id -> preferred (across canonical categories)
    let mut concept_lookup: HashMap<String, String> = HashMap::new();
    for (_cat, concepts) in canonical_index {
        for c in concepts {
            concept_lookup.insert(c.concept_id.clone(), c.preferred.clone());
        }
    }

    // Index spans by span_id
    let mut span_by_id: HashMap<String, usize> = HashMap::new();
    for (i, s) in out_spans.iter().enumerate() {
        span_by_id.insert(s.span_id.clone(), i);
    }

    // Helper: is this a safe "tag-only" category we can apply when concept_id="__NO_MATCH__"?
    fn is_tag_category(cat: &str) -> bool {
        cat == "dx" || cat.starts_with("aux_")
    }

    fn tag_label(cat: &str) -> String {
        match cat {
            "dx" => "DX_TEXT".to_string(),
            "aux_warning" => "WARNING".to_string(),
            "aux_auxiliary" => "AUX_NOTE".to_string(),
            "aux_provider" => "PROVIDER_NOTE".to_string(),
            "aux_misc" => "MISC_NOTE".to_string(),
            other => other.to_ascii_uppercase(),
        }
    }

    for p in patches {
        let cat = p.category.trim().to_string();
        let term = p.term.trim().to_string();

        if !allowed_categories.contains(&cat) {
            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: p.category.clone(),
                concept_id: p.concept_id.clone(),
                preferred: "".to_string(),
                confidence: p.confidence,
                applied: false,
                reason: "category_not_allowed".to_string(),
            });
            continue;
        }

        let reject = p.concept_id == "__REJECT__";
        if reject {
            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: cat.clone(),
                concept_id: p.concept_id.clone(),
                preferred: "".to_string(),
                confidence: p.confidence,
                applied: false,
                reason: "reject".to_string(),
            });
            continue;
        }

        let no_match = p.concept_id == "__NO_MATCH__";
        if no_match {
            // TAG-ONLY classification (no concept_id). This does NOT write to the overlay.
            if !is_tag_category(&cat) {
                applied.push(AppliedPatch {
                    span_id: p.span_id.clone(),
                    term: p.term.clone(),
                    category: cat.clone(),
                    concept_id: p.concept_id.clone(),
                    preferred: "".to_string(),
                    confidence: p.confidence,
                    applied: false,
                    reason: "no_match".to_string(),
                });
                continue;
            }

            if p.confidence < tag_min_confidence {
                applied.push(AppliedPatch {
                    span_id: p.span_id.clone(),
                    term: p.term.clone(),
                    category: cat.clone(),
                    concept_id: p.concept_id.clone(),
                    preferred: "".to_string(),
                    confidence: p.confidence,
                    applied: false,
                    reason: format!("below_tag_threshold_{tag_min_confidence}"),
                });
                continue;
            }

            let idx = match span_by_id.get(&p.span_id) {
                Some(i) => *i,
                None => {
                    applied.push(AppliedPatch {
                        span_id: p.span_id.clone(),
                        term: p.term.clone(),
                        category: cat.clone(),
                        concept_id: p.concept_id.clone(),
                        preferred: "".to_string(),
                        confidence: p.confidence,
                        applied: false,
                        reason: "span_id_not_found".to_string(),
                    });
                    continue;
                }
            };

            let span = out_spans[idx].clone();
            let start_char = span.span.start;
            let end_char = span.span.end;

            out_spans[idx] = SigSpanOutput {
                span_id: span.span_id.clone(),
                category: cat.clone(),
                role: role_to_string(default_role_for_category(&cat)),
                tags: Vec::new(),
                span: SpanOut {
                    start: start_char,
                    end: end_char,
                    text: span.span.text.clone(),
                },
                normalized: NormalizedOut::Text { label: tag_label(&cat) },
                matched_term: if term.is_empty() { span.span.text.clone() } else { term.clone() },
                confidence: p.confidence,
                source: "learn_tag".to_string(),
                candidates: None,
            };

            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: cat.clone(),
                concept_id: p.concept_id.clone(),
                preferred: "".to_string(),
                confidence: p.confidence,
                applied: true,
                reason: "tagged".to_string(),
            });
            continue;
        }

        // Alias learning (requires canonical concept_id)
        if p.confidence < min_confidence {
            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: cat.clone(),
                concept_id: p.concept_id.clone(),
                preferred: "".to_string(),
                confidence: p.confidence,
                applied: false,
                reason: format!("below_threshold_{min_confidence}"),
            });
            continue;
        }

        // Category must exist in canonical index to allow mapping
        if !canonical_index.contains_key(&cat) {
            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: cat.clone(),
                concept_id: p.concept_id.clone(),
                preferred: "".to_string(),
                confidence: p.confidence,
                applied: false,
                reason: "category_not_in_canonicals".to_string(),
            });
            continue;
        }

        let preferred = match concept_lookup.get(&p.concept_id) {
            Some(x) => x.clone(),
            None => {
                applied.push(AppliedPatch {
                    span_id: p.span_id.clone(),
                    term: p.term.clone(),
                    category: cat.clone(),
                    concept_id: p.concept_id.clone(),
                    preferred: "".to_string(),
                    confidence: p.confidence,
                    applied: false,
                    reason: "unknown_concept_id".to_string(),
                });
                continue;
            }
        };

        // Basic consistency check: concept_id must start with "<category>:"
        let expected_prefix = format!("{}:", cat);
        if !p.concept_id.starts_with(&expected_prefix) {
            applied.push(AppliedPatch {
                span_id: p.span_id.clone(),
                term: p.term.clone(),
                category: cat.clone(),
                concept_id: p.concept_id.clone(),
                preferred: preferred.clone(),
                confidence: p.confidence,
                applied: false,
                reason: "category_concept_mismatch".to_string(),
            });
            continue;
        }

        let idx = match span_by_id.get(&p.span_id) {
            Some(i) => *i,
            None => {
                applied.push(AppliedPatch {
                    span_id: p.span_id.clone(),
                    term: p.term.clone(),
                    category: cat.clone(),
                    concept_id: p.concept_id.clone(),
                    preferred: preferred.clone(),
                    confidence: p.confidence,
                    applied: false,
                    reason: "span_id_not_found".to_string(),
                });
                continue;
            }
        };

        // Write overlay patch line (unless dry-run)
        let rword = compute_rword(&term);
        if !dry_run {
            ensure_overlay_exists(overlay_path)?;
            append_overlay_alias(overlay_path, &p.concept_id, &cat, &term, &rword)?;
        }

        // Replace residual span with coded span (real-time correction)
        let span = out_spans[idx].clone();
        let start_char = span.span.start;
        let end_char = span.span.end;

        let normalized = NormalizedOut::Coded {
            concept_id: p.concept_id.clone(),
            preferred: preferred.clone(),
        };

        out_spans[idx] = SigSpanOutput {
            span_id: span.span_id.clone(),
            category: cat.clone(),
            role: role_to_string(default_role_for_category(&cat)),
            tags: Vec::new(),
            span: SpanOut {
                start: start_char,
                end: end_char,
                text: span.span.text.clone(),
            },
            normalized,
            matched_term: term.clone(),
            confidence: p.confidence,
            source: if dry_run { "learn_suggest".to_string() } else { "learn_apply".to_string() },
            candidates: None,
        };

        applied.push(AppliedPatch {
            span_id: p.span_id.clone(),
            term: term.clone(),
            category: cat.clone(),
            concept_id: p.concept_id.clone(),
            preferred,
            confidence: p.confidence,
            applied: !dry_run,
            reason: if dry_run { "dry_run".to_string() } else { "applied".to_string() },
        });
    }

    // Also: prune punctuation-only residuals aggressively (common in pain/sob patterns)
    out_spans.retain(|s| {
        if s.category != "residual_text" {
            return true;
        }
        let t = s.span.text.trim();
        if t.is_empty() {
            return false;
        }
        // Keep only if there's at least one alnum; drop pure punctuation.
        t.chars().any(|c| c.is_ascii_alphanumeric())
    });

    // Recompute canonical string is done later.
    let _ = sig_text; // keep signature stable; used for future improvements
    let _ = _byte_to_char;
    Ok((out_spans, applied))
}


fn ensure_overlay_exists(path: &PathBuf) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let mut f = File::create(path).with_context(|| format!("Failed to create overlay {}", path.display()))?;
    let ts = chrono_like_utc_string();
    writeln!(
        f,
        "-- SmartSig SIG Lexicon OVERLAY (.script)\n-- Purpose: incremental, append-only alias additions learned from residual spans\n-- Generated: {ts}\n"
    )
    .context("Failed to write overlay header")?;
    Ok(())
}

fn overlay_has_alias(path: &PathBuf, concept_id: &str, category: &str, term: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let term_esc = escape_hsqldb_string(term);
    let needle = format!(
        "INSERT INTO CUI_TERMS VALUES('{}','{}',0,'{}',",
        concept_id, category, term_esc
    );

    let f = File::open(path).with_context(|| format!("Failed to open overlay {}", path.display()))?;
    let reader = BufReader::new(f);
    for line in reader.lines() {
        let line = line?;
        if line.contains(&needle) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn append_overlay_alias(path: &PathBuf, concept_id: &str, category: &str, term: &str, rword: &str) -> Result<()> {
    if overlay_has_alias(path, concept_id, category, term)? {
        return Ok(());
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open overlay {}", path.display()))?;

    let term_esc = escape_hsqldb_string(term);
    let rword_esc = escape_hsqldb_string(rword);

    writeln!(
        f,
        "INSERT INTO CUI_TERMS VALUES('{}','{}',0,'{}','{}');",
        concept_id, category, term_esc, rword_esc
    )
    .context("Failed to append overlay row")?;

    Ok(())
}

fn escape_hsqldb_string(s: &str) -> String {
    s.replace('\'', "''")
}

fn compute_rword(term: &str) -> String {
    let lower = term.to_ascii_lowercase();
    let tokens = tokenize_words_for_rword(&lower);
    if tokens.is_empty() {
        return lower;
    }
    let stop: HashSet<&str> = ["of", "and", "or", "the", "a", "an", "to", "for"].into_iter().collect();

    let mut best = String::new();
    for t in tokens {
        if stop.contains(t.as_str()) {
            continue;
        }
        if t.len() > best.len() {
            best = t;
        }
    }
    if best.is_empty() {
        lower
            .split_whitespace()
            .max_by_key(|s| s.len())
            .unwrap_or(&lower)
            .to_string()
    } else {
        best
    }
}

fn chrono_like_utc_string() -> String {
    // Avoid pulling chrono as a dependency; we only need a readable UTC timestamp.
    // Format: YYYY-MM-DDTHH:MM:SSZ
    let now = std::time::SystemTime::now();
    let dt: chrono_stub::DateTimeUtc = chrono_stub::DateTimeUtc::from_system_time(now);
    dt.to_rfc3339()
}

/// Minimal internal UTC formatter without external deps.
mod chrono_stub {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub struct DateTimeUtc {
        secs: i64,
    }

    impl DateTimeUtc {
        pub fn from_system_time(t: SystemTime) -> Self {
            let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            Self { secs }
        }

        pub fn to_rfc3339(&self) -> String {
            // Very small RFC3339-ish formatter; correctness at second granularity is sufficient for logs.
            // Convert unix seconds to date via a simple civil-from-days algorithm.
            let (year, month, day, hour, min, sec) = unix_to_ymdhms(self.secs);
            format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
        }
    }

    // Based on Howard Hinnant's civil-from-days (public domain).
    fn unix_to_ymdhms(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
        let mut s = secs;
        if s < 0 {
            s = 0;
        }
        let sec = (s % 60) as u32;
        let mut m = s / 60;
        let min = (m % 60) as u32;
        m /= 60;
        let hour = (m % 24) as u32;
        let days = (m / 24) as i64;
        let (y, mo, d) = civil_from_days(days);
        (y, mo, d, hour, min, sec)
    }

    fn civil_from_days(z: i64) -> (i32, u32, u32) {
        let z = z + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097;                          // [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
        let y = yoe as i32 + (era * 400) as i32;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);   // [0, 365]
        let mp = (5 * doy + 2) / 153;                        // [0, 11]
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;       // [1, 31]
        let m = (mp + if mp < 10 { 3 } else { -9 }) as i32;  // [1, 12]
        let year = y + if m <= 2 { 1 } else { 0 };
        (year, m as u32, d)
    }
}

fn build_decisions_map<'a>(
    decisions: &'a [Decision],
    mentions: &[Mention],
    errors: &mut Vec<OutputError>,
) -> HashMap<&'a str, &'a Decision> {
    let mut map: HashMap<&str, &Decision> = HashMap::new();
    let mention_ids: HashSet<&str> = mentions.iter().map(|m| m.mention_id.as_str()).collect();

    for decision in decisions {
        if !mention_ids.contains(decision.mention_id.as_str()) {
            errors.push(OutputError {
                code: "unknown_mention_id".to_string(),
                message: format!("Decision mention_id {} not found in candidates.", decision.mention_id),
            });
            continue;
        }
        map.insert(decision.mention_id.as_str(), decision);
    }
    map
}

fn build_span_from_decision(
    mention: &Mention,
    decision: Option<&Decision>,
    include_candidates: bool,
    include_rejected: bool,
    errors: &mut Vec<OutputError>,
) -> Option<SigSpanOutput> {
    let role = default_role_for_category(&mention.category);
    let role_str = role_to_string(role);

    let candidates_debug = if include_candidates {
        Some(
            mention
                .candidates
                .iter()
                .map(|c| CandidateDebug {
                    concept_id: c.concept_id.clone(),
                    preferred: c.preferred.clone(),
                    score: c.score,
                    matched_term: c.matched_term.clone(),
                })
                .collect(),
        )
    } else {
        None
    };

    let span = SpanOut {
        start: mention.start_char,
        end: mention.end_char,
        text: mention.text.clone(),
    };

    let decision = match decision {
        Some(d) => d,
        None => {
            errors.push(OutputError {
                code: "missing_decision".to_string(),
                message: format!(
                    "No decision returned for mention_id {} (category {}).",
                    mention.mention_id, mention.category
                ),
            });
            let normalized = NormalizedOut::Rejected { reason: "missing_decision".to_string() };
            return Some(SigSpanOutput {
                span_id: mention.mention_id.clone(),
                category: mention.category.clone(),
                role: role_to_string(SpanRole::Residual),
                tags: Vec::new(),
                span,
                normalized,
                matched_term: mention.text.clone(),
                confidence: 0.0,
                source: "lexicon".to_string(),
                candidates: candidates_debug,
            });
        }
    };

    // Validate concept_id is in candidate list
    let candidate = mention.candidates.iter().find(|c| c.concept_id == decision.concept_id);

    let candidate = match candidate {
        Some(c) => c,
        None => {
            errors.push(OutputError {
                code: "invalid_concept_id".to_string(),
                message: format!(
                    "Concept id {} not in candidates for mention {}.",
                    decision.concept_id, mention.mention_id
                ),
            });
            let normalized = NormalizedOut::Rejected { reason: "invalid_concept_id".to_string() };
            return Some(SigSpanOutput {
                span_id: mention.mention_id.clone(),
                category: mention.category.clone(),
                role: role_to_string(SpanRole::Residual),
                tags: Vec::new(),
                span,
                normalized,
                matched_term: mention.text.clone(),
                confidence: 0.0,
                source: "lexicon".to_string(),
                candidates: candidates_debug,
            });
        }
    };

    if candidate.concept_id == "__REJECT__" {
        if !include_rejected {
            return None;
        }
        let normalized = NormalizedOut::Rejected { reason: "model_reject".to_string() };
        return Some(SigSpanOutput {
            span_id: mention.mention_id.clone(),
            category: mention.category.clone(),
            role: role_to_string(SpanRole::Residual),
            tags: Vec::new(),
            span,
            normalized,
            matched_term: mention.text.clone(),
            confidence: decision.confidence,
            source: "lexicon".to_string(),
            candidates: candidates_debug,
        });
    }

    let normalized = NormalizedOut::Coded {
        concept_id: candidate.concept_id.clone(),
        preferred: candidate.preferred.clone(),
    };

    Some(SigSpanOutput {
        span_id: mention.mention_id.clone(),
        category: mention.category.clone(),
        role: role_str,
        tags: Vec::new(),
        span,
        normalized,
        matched_term: candidate.matched_term.clone(),
        confidence: decision.confidence,
        source: "lexicon".to_string(),
        candidates: candidates_debug,
    })
}

/// For --all-candidates output: represent each mention as candidates-only (no adjudication).
fn span_from_mention_candidates(mention: &Mention, include_candidates: bool) -> SigSpanOutput {
    let role = default_role_for_category(&mention.category);
    let role_str = role_to_string(role);

    let candidates_debug = if include_candidates {
        Some(
            mention
                .candidates
                .iter()
                .map(|c| CandidateDebug {
                    concept_id: c.concept_id.clone(),
                    preferred: c.preferred.clone(),
                    score: c.score,
                    matched_term: c.matched_term.clone(),
                })
                .collect(),
        )
    } else {
        None
    };

    SigSpanOutput {
        span_id: mention.mention_id.clone(),
        category: mention.category.clone(),
        role: role_str,
        tags: Vec::new(),
        span: SpanOut {
            start: mention.start_char,
            end: mention.end_char,
            text: mention.text.clone(),
        },
        normalized: NormalizedOut::Text { label: "CANDIDATES_ONLY".to_string() },
        matched_term: mention.text.clone(),
        confidence: 0.0,
        source: "lexicon".to_string(),
        candidates: candidates_debug,
    }
}

fn default_role_for_category(category: &str) -> SpanRole {
    match category {
        // Core dosing semantics:
        "action" | "dose_unit" | "route" | "freq" | "time_unit" | "time_of_day" | "timing_event"
        | "day_of_week"
        | "food" | "flags" | "site" | "device" | "dx" | "numeric" => SpanRole::Core,

        // Auxiliary / non-core:
        "constraint" | "temporal" | "sequence" | "meta" | "sliding" | "aux_date" | "aux_icd" => SpanRole::NonCore,

        // Fallback:
        _ => SpanRole::NonCore,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DoseUnitKind {
    Form,
    Measure,
    Modifier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeasureKind {
    Mass,
    Volume,
    Other,
}

fn dose_unit_kind(concept_id: &str, preferred: &str) -> (DoseUnitKind, Option<MeasureKind>) {
    let pref = preferred.to_ascii_lowercase();

    // Delivery modifiers / adverbs (these appear in the dose_unit mapping).
    if pref == "liberally" || pref == "sparingly" {
        return (DoseUnitKind::Modifier, None);
    }

    match concept_id {
        // Measure units (subset of dose_unit)
        "dose_unit:C25613" => (DoseUnitKind::Measure, Some(MeasureKind::Other)), // Percentage
        "dose_unit:C28253" => (DoseUnitKind::Measure, Some(MeasureKind::Mass)),  // Milligram
        "dose_unit:C48152" => (DoseUnitKind::Measure, Some(MeasureKind::Mass)),  // Microgram
        "dose_unit:C48155" => (DoseUnitKind::Measure, Some(MeasureKind::Mass)),  // Gram
        "dose_unit:C48519" => (DoseUnitKind::Measure, Some(MeasureKind::Mass)),  // Ounce (mass)
        "dose_unit:C28254" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Milliliter
        "dose_unit:C48505" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Liter
        "dose_unit:C48494" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Fluid Ounce
        "dose_unit:C48529" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Pint
        "dose_unit:C48580" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Gallon
        "dose_unit:C48491" => (DoseUnitKind::Measure, Some(MeasureKind::Volume)), // Metric Drop
        "dose_unit:C44278" => (DoseUnitKind::Measure, Some(MeasureKind::Other)), // Unit
        "dose_unit:C48512" => (DoseUnitKind::Measure, Some(MeasureKind::Other)), // Milliequivalent
        "dose_unit:C48513" => (DoseUnitKind::Measure, Some(MeasureKind::Other)), // Millimole
        "dose_unit:C67315" => (DoseUnitKind::Measure, Some(MeasureKind::Other)), // MilliUnit

        // "Each" is frequently a grammar token ("each day", "each nostril") rather than a dose unit.
        "dose_unit:C64933" => (DoseUnitKind::Modifier, None), // Each
        _ => {
            // Heuristic fallback: common measure strings.
            if pref.ends_with("gram") || pref.contains("equivalent") || pref.contains("mole") {
                (DoseUnitKind::Measure, Some(MeasureKind::Mass))
            } else if pref.contains("liter") || pref.contains("fluid ounce") || pref.contains("pint") || pref.contains("gallon") || pref.contains("drop") {
                (DoseUnitKind::Measure, Some(MeasureKind::Volume))
            } else if pref == "percentage" || pref == "unit" || pref.ends_with("unit") {
                (DoseUnitKind::Measure, Some(MeasureKind::Other))
            } else {
                (DoseUnitKind::Form, None)
            }
        }
    }
}

fn apply_semantic_tags(spans: &mut [SigSpanOutput]) {
    // Clear existing tags so repeated runs are deterministic
    for s in spans.iter_mut() {
        s.tags.clear();
    }

    // 1) Dose-unit kind + mode tagging (strength-based vs unit-based)
    for s in spans.iter_mut() {
        if s.category != "dose_unit" {
            continue;
        }
        let (concept_id, preferred) = match &s.normalized {
            NormalizedOut::Coded { concept_id, preferred } => (concept_id.as_str(), preferred.as_str()),
            _ => continue,
        };

        let (kind, mk) = dose_unit_kind(concept_id, preferred);
        match kind {
            DoseUnitKind::Form => {
                s.tags.push("UNIT_KIND:FORM".to_string());
                s.tags.push("DOSE_UNIT_MODE:UNIT_BASED".to_string());
            }
            DoseUnitKind::Modifier => {
                s.tags.push("UNIT_KIND:MODIFIER".to_string());
            }
            DoseUnitKind::Measure => {
                s.tags.push("UNIT_KIND:MEASURE".to_string());
                match mk.unwrap_or(MeasureKind::Other) {
                    MeasureKind::Mass => {
                        s.tags.push("MEASURE_KIND:MASS".to_string());
                        s.tags.push("DOSE_UNIT_MODE:STRENGTH_BASED".to_string());
                        s.tags.push("ROLE:STRENGTH_UNIT".to_string());
                    }
                    MeasureKind::Volume => {
                        s.tags.push("MEASURE_KIND:VOLUME".to_string());
                        // Volume dosing behaves like "unit-based" dosing in SIGs (e.g., 5 mL).
                        s.tags.push("DOSE_UNIT_MODE:UNIT_BASED".to_string());
                        s.tags.push("ROLE:VOLUME_UNIT".to_string());
                    }
                    MeasureKind::Other => {
                        s.tags.push("MEASURE_KIND:OTHER".to_string());
                        s.tags.push("DOSE_UNIT_MODE:STRENGTH_BASED".to_string());
                        s.tags.push("ROLE:MEASURE_UNIT".to_string());
                    }
                }
            }
        }
    }

    // Helper: find the next "significant" span after i (skip punct + residual),
    // and limit binding distance to avoid spurious role assignment.
    fn next_significant_index(spans: &[SigSpanOutput], i: usize) -> Option<usize> {
        let start = spans.get(i)?.span.end;
        for j in (i + 1)..spans.len() {
            let s = &spans[j];
            if s.span.start < start {
                continue;
            }
            if s.category == "punct" || s.category == "residual_text" {
                continue;
            }
            if s.span.start.saturating_sub(start) > 6 {
                return None;
            }
            return Some(j);
        }
        None
    }

    // 2) Numeric role tagging based on the next span (dose quantity vs strength vs volume vs time)
    for i in 0..spans.len() {
        if spans[i].category != "numeric" {
            continue;
        }
        let j = match next_significant_index(spans, i) {
            Some(x) => x,
            None => continue,
        };

        let (is_measure, is_form, is_volume) = {
            let next = &spans[j];
            let is_measure = next.tags.iter().any(|t| t == "UNIT_KIND:MEASURE");
            let is_form = next.tags.iter().any(|t| t == "UNIT_KIND:FORM");
            let is_volume = next.tags.iter().any(|t| t == "MEASURE_KIND:VOLUME");
            (is_measure, is_form, is_volume)
        };

        if spans[j].category == "dose_unit" {
            if is_form {
                spans[i].tags.push("ROLE:DOSE_QUANTITY".to_string());
            } else if is_measure && is_volume {
                spans[i].tags.push("ROLE:DOSE_VOLUME".to_string());
            } else if is_measure {
                spans[i].tags.push("ROLE:STRENGTH_VALUE".to_string());
            }
        } else if spans[j].category == "time_unit" {
            spans[i].tags.push("ROLE:TIME_VALUE".to_string());
        }
    }

    // 3) Containment: numeric/time_unit fully inside an accepted freq span => mark redundant
    for k in 0..spans.len() {
        if spans[k].category != "freq" {
            continue;
        }
        let a0 = spans[k].span.start;
        let a1 = spans[k].span.end;
        for i in 0..spans.len() {
            if i == k {
                continue;
            }
            let b0 = spans[i].span.start;
            let b1 = spans[i].span.end;
            if b0 >= a0 && b1 <= a1 {
                if spans[i].category == "numeric" || spans[i].category == "time_unit" {
                    spans[i].tags.push("CONTEXT:WITHIN_FREQ".to_string());
                    if spans[i].role == "CORE" {
                        spans[i].role = "NON_CORE".to_string();
                    }
                }
            }
        }
    }

    // Normalize + dedupe tags
    for s in spans.iter_mut() {
        s.tags.sort();
        s.tags.dedup();
    }
}

fn role_to_string(role: SpanRole) -> String {
    match role {
        SpanRole::Core => "CORE".to_string(),
        SpanRole::NonCore => "NON_CORE".to_string(),
        SpanRole::Residual => "RESIDUAL".to_string(),
    }
}

fn compute_sig_structure(sig: &str, spans: &[SigSpanOutput]) -> SigStructure {
    // -------------------------
    // 1) Quality/complexity flags
    // -------------------------
    let mut flags: Vec<String> = Vec::new();

    if looks_truncated(sig) {
        flags.push("TRUNCATED_OR_INCOMPLETE".to_string());
    }

    // Presence-based complexity signals
    let mut has_then = false;
    let mut has_repeat = false;
    let mut has_max = false;
    let mut has_taper = false;
    let mut has_day_of_week = false;

    let mut freq_ids: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut dose_amount_values: Vec<String> = Vec::new();

    for s in spans.iter() {
        // Detect certain patterns via coded concept_ids
        if s.category == "sequence" {
            if let NormalizedOut::Coded { concept_id, .. } = &s.normalized {
                let cid = concept_id.as_str();
                if cid == "sequence:THEN" || cid == "sequence:AND_THEN" || cid == "sequence:FOLLOWED_BY" {
                    has_then = true;
                }
                if cid == "sequence:REPEAT" {
                    has_repeat = true;
                }
                if cid == "sequence:TAPER" || cid == "sequence:TITRATE" || cid == "sequence:INCREASE" || cid == "sequence:DECREASE" {
                    has_taper = true;
                }
            }
        }

        if s.category == "constraint" {
            if let NormalizedOut::Coded { concept_id, .. } = &s.normalized {
                if concept_id.as_str() == "constraint:MAX" {
                    has_max = true;
                }
            }
        }

        if s.category == "day_of_week" {
            has_day_of_week = true;
        }

        if s.category == "freq" {
            if let NormalizedOut::Coded { concept_id, .. } = &s.normalized {
                freq_ids.insert(concept_id.clone());
            }
        }

        // Collect dose-amount numerics for simple conflict heuristics
        if s.category == "numeric" && s.tags.iter().any(|t| t == "ROLE:DOSE_AMOUNT") {
            if let NormalizedOut::Numeric { normalized, .. } = &s.normalized {
                dose_amount_values.push(normalized.clone());
            }
        }
    }

    if has_then {
        flags.push("HAS_SEQUENCE_THEN".to_string());
    }
    if has_repeat {
        flags.push("HAS_REPEAT_CLAUSE".to_string());
    }
    if has_max {
        flags.push("HAS_MAX_CONSTRAINT".to_string());
    }
    if has_taper {
        flags.push("HAS_TAPER_OR_TITRATION".to_string());
    }
    if has_day_of_week {
        flags.push("HAS_DAY_OF_WEEK_SCHEDULE".to_string());
    }

    // Potential contradiction heuristic: multiple distinct frequency concepts and no obvious sequencing
    if freq_ids.len() >= 2 && !has_then {
        flags.push("POSSIBLE_FREQUENCY_CONFLICT".to_string());
    }

    // Potential dose conflict heuristic (very conservative)
    if dose_amount_values.len() >= 2 && !has_then && !has_taper {
        flags.push("POSSIBLE_MULTIPLE_DOSE_AMOUNTS".to_string());
    }

    // -------------------------
    // 2) Deterministic step/Clause segmentation
    // -------------------------
    let mut boundaries: Vec<usize> = Vec::new();
    boundaries.push(0);
    boundaries.push(sig.len());

    // Sentence-like separators
    for (i, b) in sig.as_bytes().iter().enumerate() {
        match *b {
            b'.' | b';' | b'\n' => boundaries.push(i + 1),
            _ => {}
        }
    }

    // Sequence markers ("then", "repeat", etc.) as clause boundaries (start a new clause)
    for s in spans.iter() {
        if s.category != "sequence" {
            continue;
        }
        if let NormalizedOut::Coded { concept_id, .. } = &s.normalized {
            let cid = concept_id.as_str();
            // Only treat these as clause boundaries; "and" is intentionally excluded
            if cid == "sequence:THEN"
                || cid == "sequence:AND_THEN"
                || cid == "sequence:FOLLOWED_BY"
                || cid == "sequence:REPEAT"
                || cid == "sequence:TAPER"
                || cid == "sequence:TITRATE"
                || cid == "sequence:INCREASE"
                || cid == "sequence:DECREASE"
            {
                boundaries.push(s.span.start);
            }
        }
    }

    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries.retain(|&b| b <= sig.len());

    let mut steps: Vec<SigStep> = Vec::new();

    for w in boundaries.windows(2) {
        let mut start = w[0];
        let mut end = w[1];
        if start >= end {
            continue;
        }

        // Trim whitespace at boundaries
        while start < end && sig.as_bytes()[start].is_ascii_whitespace() {
            start += 1;
        }
        while end > start && sig.as_bytes()[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        if start >= end {
            continue;
        }

        let seg_text = sig[start..end].trim().to_string();
        if seg_text.is_empty() {
            continue;
        }

        let mut span_ids: Vec<String> = Vec::new();
        let mut core_ids: Vec<String> = Vec::new();
        let mut noncore_ids: Vec<String> = Vec::new();
        let mut residual_ids: Vec<String> = Vec::new();

        // Collect spans that fall within this segment
        for sp in spans.iter() {
            if sp.span.start >= start && sp.span.end <= end {
                span_ids.push(sp.span_id.clone());
                match sp.role.as_str() {
                    "CORE" => core_ids.push(sp.span_id.clone()),
                    "NON_CORE" => noncore_ids.push(sp.span_id.clone()),
                    "RESIDUAL" => residual_ids.push(sp.span_id.clone()),
                    _ => {}
                }
            }
        }

        // Determine a coarse step kind
        let mut kind = "NOTE".to_string();

        // If there's any action/dose_unit/route/freq within segment, treat as instruction
        let has_core_instruction = spans.iter().any(|sp| {
            sp.span.start >= start
                && sp.span.end <= end
                && sp.role == "CORE"
                && (sp.category == "action"
                    || sp.category == "dose_unit"
                    || sp.category == "route"
                    || sp.category == "freq")
        });

        if has_core_instruction {
            kind = "INSTRUCTION".to_string();
        } else {
            // If segment has MAX constraint or REPEAT marker, specialize
            let has_max_here = spans.iter().any(|sp| {
                sp.span.start >= start
                    && sp.span.end <= end
                    && sp.category == "constraint"
                    && matches!(&sp.normalized, NormalizedOut::Coded { concept_id, .. } if concept_id == "constraint:MAX")
            });

            if has_max_here {
                kind = "CONSTRAINT".to_string();
            }

            let has_repeat_here = spans.iter().any(|sp| {
                sp.span.start >= start
                    && sp.span.end <= end
                    && sp.category == "sequence"
                    && matches!(&sp.normalized, NormalizedOut::Coded { concept_id, .. } if concept_id == "sequence:REPEAT")
            });

            if has_repeat_here {
                kind = "REPEAT".to_string();
            }
        }

        steps.push(SigStep {
            step_id: format!("step:{}", steps.len() + 1),
            kind,
            span: SpanOut {
                start,
                end,
                text: seg_text,
            },
            span_ids,
            core_span_ids: core_ids,
            noncore_span_ids: noncore_ids,
            residual_span_ids: residual_ids,
        });
    }

    // Structure kind
    let kind = if steps.len() <= 1 && flags.is_empty() {
        "SIMPLE".to_string()
    } else {
        "COMPLEX".to_string()
    };

    SigStructure { kind, flags, steps }
}

fn looks_truncated(sig: &str) -> bool {
    let s = sig.trim();

    if s.is_empty() {
        return false;
    }

    // Common truncation markers
    if s.contains("...") || s.contains('…') {
        return true;
    }

    // Unbalanced delimiters
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;
    for ch in s.chars() {
        match ch {
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
    }
    if paren != 0 || bracket != 0 || brace != 0 {
        return true;
    }

    // Ends with a dangling word fragment marker
    if s.ends_with('-') || s.ends_with(':') {
        return true;
    }

    // Very common EMR truncation fragments
    let lower = s.to_ascii_lowercase();
    for frag in ["(not cov", "not cov", "not cover", "not co", "trunc", "[trunc"].iter() {
        if lower.ends_with(frag) {
            return true;
        }
    }

    false
}


fn read_input_text(sig: Option<&String>, input: Option<&PathBuf>) -> Result<String> {
    if let Some(literal) = sig {
        return Ok(literal.to_string());
    }

    match input {
        Some(path) if path.as_path() == Path::new("-") => read_stdin(),
        Some(path) => fs::read_to_string(path).with_context(|| format!("Failed to read input file {}", path.display())),
        None => {
            if std::io::stdin().is_terminal() {
                anyhow::bail!("No input provided. Use --sig, --input, or pipe via stdin.");
            }
            read_stdin()
        }
    }
}

fn read_stdin() -> Result<String> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("Failed to read stdin")?;
    if buffer.trim().is_empty() {
        anyhow::bail!("No input provided via stdin.");
    }
    Ok(buffer)
}

fn resolve_dict_scripts(dict_script: Option<&PathBuf>, overlay_script: Option<&PathBuf>) -> Result<Vec<PathBuf>> {
    let base = resolve_dict_script(dict_script)?;
    let mut scripts = vec![base];

    // Overlay resolution:
    // - If explicit --overlay-script is provided, load it (if exists).
    // - Else, load DEFAULT_OVERLAY_SCRIPT if it exists in cwd.
    let overlay = match overlay_script {
        Some(p) => Some(p.clone()),
        None => {
            let cwd = std::env::current_dir().context("Failed to determine current directory")?;
            let candidate = cwd.join(DEFAULT_OVERLAY_SCRIPT);
            if candidate.exists() {
                Some(candidate)
            } else {
                None
            }
        }
    };

    if let Some(p) = overlay {
        if p.exists() {
            scripts.push(p);
        }
    }

    Ok(scripts)
}

fn resolve_dict_script(dict_script: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = dict_script {
        return Ok(path.clone());
    }

    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    let candidate = cwd.join(DEFAULT_DICT_SCRIPT);
    if candidate.exists() {
        Ok(candidate)
    } else {
        anyhow::bail!(
            "No base lexicon provided. Use --dict-script or place {} relative to the working directory.",
            DEFAULT_DICT_SCRIPT
        );
    }
}

fn resolve_overlay_script_path(overlay_script: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(path) = overlay_script {
        return Ok(path.clone());
    }
    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    Ok(cwd.join(DEFAULT_OVERLAY_SCRIPT))
}


// ---------------------------------------------------------------------------
// Internal helper: run `codex exec` without polluting this command's stdout.
// ---------------------------------------------------------------------------
//
// The `codex exec` runner prints its final message to stdout by design.
// Because `sig-concept-extract` also prints its own JSON output, calling
// `codex_exec::run_main` directly would interleave multiple JSON objects on
// stdout and corrupt JSONL outputs.
//
// We keep the existing in-process execution (so we can pass large prompts
// without hitting OS argv length limits) and temporarily redirect stdout to
// /dev/null while the internal `exec` runs.
//
// NOTE: This is only enabled on unix platforms. On non-unix, stdout will not
// be redirected (and you may see mixed outputs); consider running via WSL.
#[cfg(unix)]
unsafe extern "C" {
    fn dup(fd: i32) -> i32;
    fn dup2(oldfd: i32, newfd: i32) -> i32;
    fn close(fd: i32) -> i32;
}

#[cfg(unix)]
struct StdoutRedirectGuard {
    saved_fd: i32,
}

#[cfg(unix)]
impl StdoutRedirectGuard {
    fn redirect_to_devnull() -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        let saved = unsafe { dup(1) };
        if saved < 0 {
            return Err(anyhow::anyhow!(std::io::Error::last_os_error()));
        }

        let devnull = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/null")
            .context("Failed to open /dev/null for stdout redirection")?;
        let devnull_fd = devnull.as_raw_fd();
        let rc = unsafe { dup2(devnull_fd, 1) };
        if rc < 0 {
            unsafe {
                close(saved);
            }
            return Err(anyhow::anyhow!(std::io::Error::last_os_error()));
        }

        Ok(Self { saved_fd: saved })
    }
}

#[cfg(unix)]
impl Drop for StdoutRedirectGuard {
    fn drop(&mut self) {
        unsafe {
            // Restore stdout and close the saved fd.
            let _ = dup2(self.saved_fd, 1);
            let _ = close(self.saved_fd);
        }
    }
}

async fn run_exec_silently(exec_cli: ExecCli, codex_linux_sandbox_exe: Option<PathBuf>) -> Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _guard = StdoutRedirectGuard::redirect_to_devnull()?;
        let res = codex_exec::run_main(exec_cli, codex_linux_sandbox_exe).await;
        // Guard drop restores stdout.
        let _ = std::io::stdout().flush();
        res
    }

    #[cfg(not(unix))]
    {
        codex_exec::run_main(exec_cli, codex_linux_sandbox_exe).await
    }
}

fn write_schema_tempfile(prefix: &str, contents: &str) -> Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let filename = format!("{prefix}-{}.json", std::process::id());
    path.push(filename);
    fs::write(&path, contents).with_context(|| format!("Failed to write schema to {}", path.display()))?;
    Ok(path)
}

fn write_output_json<T: Serialize>(value: &T, output: Option<&PathBuf>) -> Result<()> {
    let json = serde_json::to_string(value)?;
    if let Some(path) = output {
        fs::write(path, json.as_bytes()).with_context(|| format!("Failed to write output to {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn should_skip_lexicon_term(category: &str, term_lower: &str) -> bool {
    match (category, term_lower) {
        // Extremely common English word; produces noisy false positives when treated as a dose unit.
        ("dose_unit", "each") => true,

        _ => false,
    }
}

fn scan_lexicon_terms(
    dict_scripts: &[PathBuf],
    input_text: &str,
    min_term_length: usize,
    max_mentions: usize,
) -> Result<(HashMap<MentionKey, MentionTemp>, HashSet<String>)> {
    let input_lower = input_text.to_ascii_lowercase();
    let tokens = tokenize_words_for_rword(&input_lower);

    let mut mentions: HashMap<MentionKey, MentionTemp> = HashMap::new();
    let mut concept_ids = HashSet::new();

    for script in dict_scripts {
        let file = File::open(script).with_context(|| format!("Failed to open lexicon {}", script.display()))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if !line.starts_with("INSERT INTO CUI_TERMS VALUES(") {
                continue;
            }

            let values = match parse_insert_values(&line) {
                Some(values) => values,
                None => continue,
            };
            if values.len() < 5 {
                continue;
            }

            let concept_id = values[0].trim().to_string();
            let category = values[1].trim().to_string();
            let term = values[3].trim().to_string();
            let rword = values[4].trim().to_ascii_lowercase();

            if term.len() < min_term_length {
                continue;
            }

            // Rare-word filter
            if !tokens.contains(&rword) {
                continue;
            }

            let term_lower = term.to_ascii_lowercase();
            if should_skip_lexicon_term(&category, &term_lower) {
                continue;
            }
            for start in find_term_matches(&input_lower, &term_lower) {
                let end = start + term_lower.len();
                if !input_text.is_char_boundary(start) || !input_text.is_char_boundary(end) {
                    continue;
                }

                let key = MentionKey { start, end, category: category.clone() };
                if mentions.len() >= max_mentions && !mentions.contains_key(&key) {
                    continue;
                }

                let mention = mentions.entry(key).or_insert_with(|| MentionTemp {
                    start,
                    end,
                    category: category.clone(),
                    text: input_text[start..end].to_string(),
                    candidates: HashMap::new(),
                });

                let entry = mention.candidates.entry(concept_id.clone()).or_insert(CandidateTemp {
                    concept_id: concept_id.clone(),
                    matched_term: term.clone(),
                    score: term.len(),
                });

                if term.len() > entry.score {
                    entry.matched_term = term.clone();
                    entry.score = term.len();
                }

                concept_ids.insert(concept_id.clone());
            }
        }
    }

    Ok((mentions, concept_ids))
}

fn load_lexicon_metadata(dict_scripts: &[PathBuf], concept_ids: &HashSet<String>) -> Result<LexiconMetadata> {
    let mut meta = LexiconMetadata::default();

    for script in dict_scripts {
        let file = File::open(script).with_context(|| format!("Failed to open lexicon {}", script.display()))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("INSERT INTO PREFTERM VALUES(") {
                if let Some(values) = parse_insert_values(&line) {
                    if values.len() >= 2 && concept_ids.contains(values[0].trim()) {
                        meta.prefterm
                            .insert(values[0].trim().to_string(), values[1].trim().to_string());
                    }
                }
                continue;
            }

            if line.starts_with("INSERT INTO TUI VALUES(") {
                if let Some(values) = parse_insert_values(&line) {
                    if values.len() >= 2 && concept_ids.contains(values[0].trim()) {
                        let tui = values[1]
                            .trim()
                            .parse::<u32>()
                            .ok()
                            .map(|val| format!("T{val:03}"));
                        if let Some(tui) = tui {
                            meta.tui.entry(values[0].trim().to_string()).or_default().push(tui);
                        }
                    }
                }
            }
        }
    }

    Ok(meta)
}

fn load_canonical_index(
    dict_scripts: &[PathBuf],
    allowed_categories: &HashSet<String>,
) -> Result<HashMap<String, Vec<CanonicalConcept>>> {
    let mut by_cat: HashMap<String, Vec<CanonicalConcept>> = HashMap::new();

    for script in dict_scripts {
        let file = File::open(script).with_context(|| format!("Failed to open lexicon {}", script.display()))?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if !line.starts_with("INSERT INTO PREFTERM VALUES(") {
                continue;
            }
            let values = match parse_insert_values(&line) {
                Some(v) => v,
                None => continue,
            };
            if values.len() < 2 {
                continue;
            }
            let concept_id = values[0].trim().to_string();
            let preferred = values[1].trim().to_string();
            let category = concept_id.split(':').next().unwrap_or("").to_string();

            if allowed_categories.contains("any") || allowed_categories.contains(category.as_str()) {
                by_cat.entry(category).or_default().push(CanonicalConcept { concept_id, preferred });
            }
        }
    }

    // De-duplicate by concept_id within each category
    for (_cat, vec) in by_cat.iter_mut() {
        let mut seen = HashSet::new();
        vec.retain(|c| seen.insert(c.concept_id.clone()));
        vec.sort_by(|a, b| a.preferred.cmp(&b.preferred).then_with(|| a.concept_id.cmp(&b.concept_id)));
    }

    Ok(by_cat)
}

fn build_mentions(
    mentions_map: HashMap<MentionKey, MentionTemp>,
    meta: &LexiconMetadata,
    byte_to_char: &[usize],
    max_candidates: usize,
    max_mentions: usize,
) -> Vec<Mention> {
    let mut mentions: Vec<MentionTemp> = mentions_map.into_values().collect();
    mentions.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.category.cmp(&b.category)));
    if mentions.len() > max_mentions {
        mentions.truncate(max_mentions);
    }

    mentions
        .into_iter()
        .enumerate()
        .map(|(idx, mention)| {
            let mut candidates: Vec<Candidate> = mention
                .candidates
                .into_values()
                .map(|c| {
                    let preferred = meta
                        .prefterm
                        .get(&c.concept_id)
                        .cloned()
                        .unwrap_or_else(|| c.matched_term.clone());
                    let tui = meta.tui.get(&c.concept_id).cloned().unwrap_or_default();
                    Candidate {
                        concept_id: c.concept_id,
                        preferred,
                        matched_term: c.matched_term,
                        tui,
                        score: c.score,
                    }
                })
                .collect();

            candidates.sort_by(|a, b| {
                b.score
                    .cmp(&a.score)
                    .then_with(|| a.concept_id.cmp(&b.concept_id))
            });

            if candidates.len() > max_candidates {
                candidates.truncate(max_candidates);
            }

            Mention {
                mention_id: format!("m{}", idx + 1),
                start_byte: mention.start,
                end_byte: mention.end,
                start_char: byte_to_char[mention.start],
                end_char: byte_to_char[mention.end],
                category: mention.category,
                text: mention.text,
                candidates,
            }
        })
        .collect()
}

/// Drop mentions that are strictly contained within a larger mention of the same category.
/// Example: "as needed" is contained in "as needed for".
fn prune_contained_mentions(mut mentions: Vec<Mention>) -> Vec<Mention> {
    mentions.sort_by(|a, b| {
        a.category
            .cmp(&b.category)
            .then_with(|| a.start_byte.cmp(&b.start_byte))
            .then_with(|| (b.end_byte - b.start_byte).cmp(&(a.end_byte - a.start_byte))) // longer first
    });

    let mut kept: Vec<Mention> = Vec::new();
    for m in mentions {
        let contained = kept.iter().any(|k| {
            k.category == m.category
                && k.start_byte <= m.start_byte
                && k.end_byte >= m.end_byte
                && (k.start_byte != m.start_byte || k.end_byte != m.end_byte)
        });
        if !contained {
            kept.push(m);
        }
    }

    // Reassign mention_ids to stay contiguous
    kept.sort_by(|a, b| a.start_byte.cmp(&b.start_byte).then_with(|| a.category.cmp(&b.category)));
    for (i, m) in kept.iter_mut().enumerate() {
        m.mention_id = format!("m{}", i + 1);
    }
    kept
}

/// Tokenize for rword filtering:
/// - Splits on non-alphanumeric
/// - ALSO splits on transitions between letters and digits (so q6h -> q, 6, h; 10d -> 10, d)
fn tokenize_words_for_rword(text: &str) -> HashSet<String> {
    // Base tokens: contiguous ASCII alphanumeric sequences (keeps "4x", "q6h", "10d" intact).
    // Extra tokens: also split mixed alnum tokens on digit<->alpha transitions, adding sub-tokens ("4", "x", "q", "6", "h").
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Class {
        Alpha,
        Digit,
        Other,
    }

    fn class_of(b: u8) -> Class {
        if b.is_ascii_digit() {
            Class::Digit
        } else if b.is_ascii_alphabetic() {
            Class::Alpha
        } else {
            Class::Other
        }
    }

    let mut tokens = HashSet::new();
    let bytes = text.as_bytes();
    let mut start: Option<usize> = None;

    for (idx, b) in bytes.iter().enumerate() {
        if b.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            add_token_and_splits(text, s, idx, &mut tokens);
        }
    }
    if let Some(s) = start {
        add_token_and_splits(text, s, bytes.len(), &mut tokens);
    }

    fn add_token_and_splits(text: &str, start: usize, end: usize, tokens: &mut HashSet<String>) {
        if end <= start {
            return;
        }
        let tok = text[start..end].to_string();
        tokens.insert(tok.clone());

        // Add transition splits (only if token contains both digits and letters)
        let bytes = tok.as_bytes();
        let mut has_digit = false;
        let mut has_alpha = false;
        for b in bytes {
            has_digit |= b.is_ascii_digit();
            has_alpha |= b.is_ascii_alphabetic();
        }
        if !(has_digit && has_alpha) {
            return;
        }

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum LocalClass {
            Alpha,
            Digit,
        }
        let mut s: Option<usize> = None;
        let mut prev: Option<LocalClass> = None;

        for (i, b) in bytes.iter().enumerate() {
            if b.is_ascii_alphanumeric() {
                let c = if b.is_ascii_digit() { LocalClass::Digit } else { LocalClass::Alpha };
                match (s, prev) {
                    (None, _) => {
                        s = Some(i);
                        prev = Some(c);
                    }
                    (Some(st), Some(pc)) if pc != c => {
                        tokens.insert(tok[st..i].to_string());
                        s = Some(i);
                        prev = Some(c);
                    }
                    _ => {
                        prev = Some(c);
                    }
                }
            } else if let Some(st) = s.take() {
                tokens.insert(tok[st..i].to_string());
                prev = None;
            }
        }
        if let Some(st) = s {
            tokens.insert(tok[st..].to_string());
        }
    }

    // Suppress unused warning for helper
    let _ = class_of;

    tokens
}

fn find_term_matches(haystack: &str, needle: &str) -> Vec<usize> {
    if needle.is_empty() {
        return Vec::new();
    }

    let enforce_left = needle
        .as_bytes()
        .first()
        .map(|b| b.is_ascii_alphanumeric())
        .unwrap_or(false);
    let enforce_right = needle
        .as_bytes()
        .last()
        .map(|b| b.is_ascii_alphanumeric())
        .unwrap_or(false);

    let mut matches = Vec::new();
    let mut offset = 0;
    while let Some(pos) = haystack[offset..].find(needle) {
        let start = offset + pos;
        let end = start + needle.len();
        if boundaries_ok(haystack, start, end, enforce_left, enforce_right) {
            matches.push(start);
        }
        offset = start + 1;
    }
    matches
}

/// SIG-specific boundary logic:
/// - non-alphanumeric boundaries are boundaries
/// - transitions between letters and digits are ALSO treated as boundaries (take1 => take + 1)
fn boundaries_ok(haystack: &str, start: usize, end: usize, enforce_left: bool, enforce_right: bool) -> bool {
    let bytes = haystack.as_bytes();

    let left_ok = if !enforce_left || start == 0 {
        true
    } else {
        let prev = bytes[start - 1];
        if !prev.is_ascii_alphanumeric() {
            true
        } else {
            let cur = bytes.get(start).copied().unwrap_or(b' ');
            is_digit_alpha_boundary(prev, cur)
        }
    };

    let right_ok = if !enforce_right || end >= haystack.len() {
        true
    } else {
        let next = bytes[end];
        if !next.is_ascii_alphanumeric() {
            true
        } else {
            let prev = bytes.get(end.saturating_sub(1)).copied().unwrap_or(b' ');
            is_digit_alpha_boundary(prev, next)
        }
    };

    left_ok && right_ok
}

fn is_boundary_between(prev: u8, next: u8) -> bool {
    if !prev.is_ascii_alphanumeric() || !next.is_ascii_alphanumeric() {
        return true;
    }
    (prev.is_ascii_alphabetic() && next.is_ascii_digit()) || (prev.is_ascii_digit() && next.is_ascii_alphabetic())
}

fn is_digit_alpha_boundary(prev: u8, next: u8) -> bool {
    (prev.is_ascii_alphabetic() && next.is_ascii_digit())
        || (prev.is_ascii_digit() && next.is_ascii_alphabetic())
}

/// Parse an HSQLDB INSERT line like:
/// INSERT INTO CUI_TERMS VALUES('x','y',0,'term','rword');
fn parse_insert_values(line: &str) -> Option<Vec<String>> {
    let start = line.find("VALUES(")? + "VALUES(".len();
    let bytes = line.as_bytes();
    let mut values = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut idx = start;

    while idx < bytes.len() {
        let b = bytes[idx];
        if in_string {
            if b == b'\'' {
                if idx + 1 < bytes.len() && bytes[idx + 1] == b'\'' {
                    current.push('\'');
                    idx += 2;
                } else {
                    in_string = false;
                    idx += 1;
                }
            } else {
                current.push(b as char);
                idx += 1;
            }
        } else {
            match b {
                b'\'' => {
                    in_string = true;
                    idx += 1;
                }
                b',' => {
                    values.push(current.trim().to_string());
                    current.clear();
                    idx += 1;
                }
                b')' => {
                    values.push(current.trim().to_string());
                    return Some(values);
                }
                _ => {
                    current.push(b as char);
                    idx += 1;
                }
            }
        }
    }
    None
}

/// Build byte->char index mapping so output spans use Unicode scalar indices.
fn build_byte_to_char_map(text: &str) -> Vec<usize> {
    let mut map = vec![0; text.len() + 1];
    let mut char_idx = 0;
    for (byte_idx, _) in text.char_indices() {
        map[byte_idx] = char_idx;
        char_idx += 1;
    }
    map[text.len()] = char_idx;
    map
}

fn build_char_to_byte_map(text: &str) -> Vec<usize> {
    // char index (Unicode scalar) -> byte index
    let mut map: Vec<usize> = Vec::with_capacity(text.chars().count() + 1);
    for (byte_idx, _) in text.char_indices() {
        map.push(byte_idx);
    }
    map.push(text.len());
    map
}

fn slice_chars(text: &str, char_to_byte: &[usize], start_char: usize, end_char: usize) -> String {
    if char_to_byte.is_empty() {
        return String::new();
    }
    let char_len = char_to_byte.len().saturating_sub(1);
    let s = start_char.min(char_len);
    let e = end_char.min(char_len);
    if s >= e {
        return String::new();
    }
    let sb = char_to_byte[s];
    let eb = char_to_byte[e];
    text.get(sb..eb).unwrap_or("").to_string()
}


/// Detect deterministic spans that should not be delegated to the LLM:
/// - dates (MM/DD/YYYY, MM-DD-YY, YYYY-MM-DD)
/// - numeric literals (int/decimal/fraction/range)
/// - ICD-10-like codes
///
/// Returns:
/// - output spans
/// - byte coverage ranges to exclude from residual spans
fn detect_auto_spans(sig_text: &str, byte_to_char: &[usize]) -> (Vec<SigSpanOutput>, Vec<(usize, usize)>) {
    let mut spans = Vec::new();
    let mut cov = Vec::new();

    // 1) Dates
    let date_spans = detect_date_spans(sig_text);
    let mut blocked = vec![false; sig_text.len() + 1];
    for d in &date_spans {
        mark_bytes(&mut blocked, d.start, d.end);
        cov.push((d.start, d.end));
        spans.push(SigSpanOutput {
            span_id: d.id.clone(),
            category: "aux_date".to_string(),
            role: role_to_string(SpanRole::NonCore),
            tags: Vec::new(),
            span: SpanOut {
                start: byte_to_char[d.start],
                end: byte_to_char[d.end],
                text: sig_text[d.start..d.end].to_string(),
            },
            normalized: NormalizedOut::Date { iso: d.iso.clone() },
            matched_term: sig_text[d.start..d.end].to_string(),
            confidence: 1.0,
            source: "auto".to_string(),
            candidates: None,
        });
    }

    // 2) ICD codes (token-based)
    for icd in detect_icd_spans(sig_text) {
        if overlaps_blocked(&blocked, icd.start, icd.end) {
            continue;
        }
        mark_bytes(&mut blocked, icd.start, icd.end);
        cov.push((icd.start, icd.end));
        spans.push(SigSpanOutput {
            span_id: icd.id.clone(),
            category: "aux_icd".to_string(),
            role: role_to_string(SpanRole::NonCore),
            tags: Vec::new(),
            span: SpanOut {
                start: byte_to_char[icd.start],
                end: byte_to_char[icd.end],
                text: sig_text[icd.start..icd.end].to_string(),
            },
            normalized: NormalizedOut::Icd {
                code: sig_text[icd.start..icd.end].to_string(),
            },
            matched_term: sig_text[icd.start..icd.end].to_string(),
            confidence: 1.0,
            source: "auto".to_string(),
            candidates: None,
        });
    }

    // 3) Numeric literals
    for num in detect_numeric_spans(sig_text) {
        if overlaps_blocked(&blocked, num.start, num.end) {
            continue;
        }
        mark_bytes(&mut blocked, num.start, num.end);
        cov.push((num.start, num.end));
        spans.push(SigSpanOutput {
            span_id: num.id.clone(),
            category: "numeric".to_string(),
            role: role_to_string(SpanRole::Core),
            tags: Vec::new(),
            span: SpanOut {
                start: byte_to_char[num.start],
                end: byte_to_char[num.end],
                text: sig_text[num.start..num.end].to_string(),
            },
            normalized: NormalizedOut::Numeric {
                number_kind: num.kind.clone(),
                normalized: num.normalized.clone(),
            },
            matched_term: sig_text[num.start..num.end].to_string(),
            confidence: 1.0,
            source: "auto".to_string(),
            candidates: None,
        });

        if let Some(unit) = num.unit_suffix.as_ref() {
            cov.push((unit.start, unit.end));
            spans.push(SigSpanOutput {
                span_id: unit.id.clone(),
                category: "time_unit".to_string(),
                role: role_to_string(SpanRole::Core),
                tags: Vec::new(),
                span: SpanOut {
                    start: byte_to_char[unit.start],
                    end: byte_to_char[unit.end],
                    text: sig_text[unit.start..unit.end].to_string(),
                },
                normalized: NormalizedOut::Coded {
                    concept_id: unit.concept_id.clone(),
                    preferred: unit.preferred.clone(),
                },
                matched_term: sig_text[unit.start..unit.end].to_string(),
                confidence: 1.0,
                source: "auto".to_string(),
                candidates: None,
            });
        }
    }

    // 4) Punctuation / structural markers (non-core)
    // We intentionally keep this small/high-signal (parentheses, brackets, braces, slash).
    for (bidx, ch) in sig_text.char_indices() {
        let label = match ch {
            '(' => "LPAREN",
            ')' => "RPAREN",
            '[' => "LBRACKET",
            ']' => "RBRACKET",
            '{' => "LBRACE",
            '}' => "RBRACE",
            '/' => "SLASH",
            _ => continue,
        };
        let end = bidx + ch.len_utf8();
        if overlaps_blocked(&blocked, bidx, end) {
            continue;
        }
        mark_bytes(&mut blocked, bidx, end);
        cov.push((bidx, end));
        spans.push(SigSpanOutput {
            span_id: format!("punct:{}:{}", label, bidx),
            category: "punct".to_string(),
            role: role_to_string(SpanRole::NonCore),
            tags: Vec::new(),
            span: SpanOut {
                start: byte_to_char[bidx],
                end: byte_to_char[end],
                text: sig_text[bidx..end].to_string(),
            },
            normalized: NormalizedOut::Text {
                label: label.to_string(),
            },
            matched_term: sig_text[bidx..end].to_string(),
            confidence: 1.0,
            source: "auto".to_string(),
            candidates: None,
        });
    }

    (spans, cov)
}

fn compute_residual_spans(
    sig_text: &str,
    byte_to_char: &[usize],
    lexicon_mentions: &[Mention],
    auto_cov: &[(usize, usize)],
) -> Vec<SigSpanOutput> {
    let mut covered = vec![false; sig_text.len() + 1];

    for m in lexicon_mentions {
        mark_bytes(&mut covered, m.start_byte, m.end_byte);
    }
    for (s, e) in auto_cov {
        mark_bytes(&mut covered, *s, *e);
    }

    // Residual tokenization:
    // - Skip covered and whitespace
    // - Treat ASCII punctuation as delimiter (do NOT emit "/sob" as a single unknown)
    // - Emit only alphanumeric runs as residual_text
    let bytes = sig_text.as_bytes();
    let mut residuals = Vec::new();
    let mut idx = 0usize;
    let mut rid = 1usize;

    while idx < bytes.len() {
        if covered[idx] || bytes[idx].is_ascii_whitespace() {
            idx += 1;
            continue;
        }
        if bytes[idx].is_ascii_punctuation() {
            // delimiter; ignore
            idx += 1;
            continue;
        }
        if !bytes[idx].is_ascii_alphanumeric() {
            idx += 1;
            continue;
        }

        let start = idx;
        idx += 1;
        while idx < bytes.len() && !covered[idx] && bytes[idx].is_ascii_alphanumeric() {
            idx += 1;
        }
        let end = idx;

        if end <= start {
            continue;
        }

        let text = sig_text[start..end].to_string();
        let lower = text.to_ascii_lowercase();
        let label = match lower.as_str() {
            "a" | "an" | "the" | "and" | "or" | "to" | "of" | "in" | "on" | "at" | "for" | "by" | "per" | "as" | "if" | "then" | "with" | "without" | "into" | "onto" | "from" | "during" => "STOPWORD",
            _ => "FREE_TEXT",
        };

        residuals.push(SigSpanOutput {
            span_id: format!("r{}", rid),
            category: "residual_text".to_string(),
            role: role_to_string(SpanRole::Residual),
            tags: Vec::new(),
            span: SpanOut {
                start: byte_to_char[start],
                end: byte_to_char[end],
                text,
            },
            normalized: NormalizedOut::Text { label: label.to_string() },
            matched_term: sig_text[start..end].to_string(),
            confidence: 1.0,
            source: "residual".to_string(),
            candidates: None,
        });
        rid += 1;
    }

    residuals
}

fn mark_bytes(mask: &mut [bool], start: usize, end: usize) {
    let end = end.min(mask.len().saturating_sub(1));
    let start = start.min(end);
    for i in start..end {
        mask[i] = true;
    }
}

fn overlaps_blocked(mask: &[bool], start: usize, end: usize) -> bool {
    let end = end.min(mask.len().saturating_sub(1));
    let start = start.min(end);
    for i in start..end {
        if mask[i] {
            return true;
        }
    }
    false
}

#[derive(Debug)]
struct DateSpan {
    id: String,
    start: usize,
    end: usize,
    iso: String,
}

fn detect_date_spans(text: &str) -> Vec<DateSpan> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut idx = 0usize;
    let mut did = 1usize;

    while idx < bytes.len() {
        if !bytes[idx].is_ascii_digit() {
            idx += 1;
            continue;
        }

        let start = idx;
        let (p1_end, p1) = read_digits(text, idx);
        let mut j = p1_end;
        if j >= bytes.len() {
            idx = start + 1;
            continue;
        }
        let sep1 = bytes[j];
        if sep1 != b'/' && sep1 != b'-' {
            idx = start + 1;
            continue;
        }
        j += 1;

        if j >= bytes.len() || !bytes[j].is_ascii_digit() {
            idx = start + 1;
            continue;
        }
        let (p2_end, p2) = read_digits(text, j);
        j = p2_end;
        if j >= bytes.len() {
            idx = start + 1;
            continue;
        }
        let sep2 = bytes[j];
        if sep2 != b'/' && sep2 != b'-' {
            idx = start + 1;
            continue;
        }
        j += 1;

        if j >= bytes.len() || !bytes[j].is_ascii_digit() {
            idx = start + 1;
            continue;
        }
        let (p3_end, p3) = read_digits(text, j);
        let end = p3_end;

        if !date_boundaries_ok(text, start, end) {
            idx = start + 1;
            continue;
        }

        if p1.len() > 4 || p2.len() > 2 || p3.len() > 4 {
            idx = start + 1;
            continue;
        }
        if p3.len() < 2 {
            idx = start + 1;
            continue;
        }

        if let Some(iso) = normalize_date_iso(&p1, &p2, &p3) {
            spans.push(DateSpan {
                id: format!("d{}", did),
                start,
                end,
                iso,
            });
            did += 1;
            idx = end;
            continue;
        }

        idx = start + 1;
    }

    spans
}

fn date_boundaries_ok(text: &str, start: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    let left_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
    let right_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
    left_ok && right_ok
}

fn normalize_date_iso(p1: &str, p2: &str, p3: &str) -> Option<String> {
    let a = p1.parse::<u32>().ok()?;
    let b = p2.parse::<u32>().ok()?;
    let c = p3.parse::<u32>().ok()?;

    let (year, month, day) = if p1.len() == 4 {
        (a, b, c)
    } else if p3.len() == 4 {
        (c, a, b)
    } else {
        let yy = c;
        let yyyy = if yy <= 69 { 2000 + yy } else { 1900 + yy };
        (yyyy, a, b)
    };

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(format!("{year:04}-{month:02}-{day:02}"))
}

fn read_digits(text: &str, start: usize) -> (usize, String) {
    let bytes = text.as_bytes();
    let mut idx = start;
    while idx < bytes.len() && bytes[idx].is_ascii_digit() {
        idx += 1;
    }
    (idx, text[start..idx].to_string())
}

#[derive(Debug)]
struct IcdSpan {
    id: String,
    start: usize,
    end: usize,
}

fn detect_icd_spans(text: &str) -> Vec<IcdSpan> {
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let mut idx = 0usize;
    let mut iid = 1usize;

    while idx < bytes.len() {
        if !bytes[idx].is_ascii_alphanumeric() {
            idx += 1;
            continue;
        }
        let start = idx;
        idx += 1;
        while idx < bytes.len() && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'.') {
            idx += 1;
        }
        let end = idx;
        let token = &text[start..end];

        if is_icd_token(token) {
            spans.push(IcdSpan {
                id: format!("i{}", iid),
                start,
                end,
            });
            iid += 1;
        }
    }

    spans
}

fn is_icd_token(token: &str) -> bool {
    let mut chars = token.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if !first.is_ascii_alphabetic() || !first.is_ascii_uppercase() {
        return false;
    }

    let second = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if !second.is_ascii_digit() {
        return false;
    }

    let third = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if !(third.is_ascii_digit() || third.is_ascii_uppercase()) {
        return false;
    }

    let rest: String = chars.collect();
    if rest.is_empty() {
        return true;
    }
    if !rest.starts_with('.') {
        return false;
    }
    let suffix = &rest[1..];
    if suffix.is_empty() || suffix.len() > 4 {
        return false;
    }
    suffix
        .chars()
        .all(|c| c.is_ascii_alphanumeric() && (c.is_ascii_digit() || c.is_ascii_uppercase()))
}

#[derive(Debug)]
struct TimeUnitSuffix {
    id: String,
    start: usize,
    end: usize,
    concept_id: String,
    preferred: String,
}

#[derive(Debug)]
struct NumericSpan {
    id: String,
    start: usize,
    end: usize,
    kind: String,
    normalized: String,
    unit_suffix: Option<TimeUnitSuffix>,
}

fn detect_numeric_spans(text: &str) -> Vec<NumericSpan> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut idx = 0usize;
    let mut nid = 1usize;

    // Pass 1: digit-driven literals (integers, decimals, ranges, fractions, percents)
    while idx < bytes.len() {
        if !bytes[idx].is_ascii_digit() {
            idx += 1;
            continue;
        }

        let start = idx;
        idx += 1;

        while idx < bytes.len() {
            let b = bytes[idx];
            if b.is_ascii_digit() || b == b'.' || b == b'/' || b == b'-' || b == b'%' {
                idx += 1;
            } else {
                break;
            }
        }

        let end = idx;
        if end <= start {
            continue;
        }

        let raw = &text[start..end];
        let (kind, normalized) = classify_numeric_literal(raw);
        spans.push(NumericSpan {
            id: format!("n{nid}"),
            start,
            end,
            kind,
            normalized,
            unit_suffix: None,
        });
        nid += 1;
    }

    // Pass 2: spelled-out number words ("one", "two", "half", ...)
    // NOTE: This is intentionally conservative — do NOT attempt to parse complex phrases here.
    idx = 0usize;
    while idx < bytes.len() {
        if !bytes[idx].is_ascii_alphabetic() {
            idx += 1;
            continue;
        }

        let start = idx;
        idx += 1;
        while idx < bytes.len() && bytes[idx].is_ascii_alphabetic() {
            idx += 1;
        }
        let end = idx;
        if end <= start {
            continue;
        }

        // Safety: ASCII alphabetic tokens only (SIGs are overwhelmingly ASCII).
        let token = text[start..end].to_ascii_lowercase();
        if let Some(num_raw) = number_word_to_numeric(&token) {
            let (kind, normalized) = classify_numeric_literal(num_raw);
            spans.push(NumericSpan {
                id: format!("n{nid}"),
                start,
                end,
                kind,
                normalized,
                unit_suffix: None,
            });
            nid += 1;
        }
    }

    spans
}

fn number_word_to_numeric(token: &str) -> Option<&'static str> {
    match token {
        "zero" => Some("0"),
        "one" => Some("1"),
        "two" => Some("2"),
        "three" => Some("3"),
        "four" => Some("4"),
        "five" => Some("5"),
        "six" => Some("6"),
        "seven" => Some("7"),
        "eight" => Some("8"),
        "nine" => Some("9"),
        "ten" => Some("10"),
        "half" => Some("0.5"),
        "quarter" => Some("0.25"),
        _ => None,
    }
}


fn classify_numeric_literal(raw: &str) -> (String, String) {
    if raw.contains('-') {
        let parts: Vec<&str> = raw.split('-').collect();
        if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            return ("RANGE".to_string(), format!("{}-{}", normalize_decimal(parts[0]), normalize_decimal(parts[1])));
        }
    }

    if raw.contains('/') {
        let parts: Vec<&str> = raw.split('/').collect();
        if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            let num = parts[0].trim_start_matches('0');
            let den = parts[1].trim_start_matches('0');
            let num = if num.is_empty() { "0" } else { num };
            let den = if den.is_empty() { "0" } else { den };
            return ("FRACTION".to_string(), format!("{}/{}", num, den));
        }
    }

    if raw.contains('.') {
        return ("DECIMAL".to_string(), normalize_decimal(raw));
    }

    let int_norm = raw.trim_start_matches('0');
    let int_norm = if int_norm.is_empty() { "0" } else { int_norm };
    ("INTEGER".to_string(), int_norm.to_string())
}

fn normalize_decimal(raw: &str) -> String {
    let raw = raw.trim();
    if let Some((int_part, frac_part)) = raw.split_once('.') {
        let int_norm = int_part.trim_start_matches('0');
        let int_norm = if int_norm.is_empty() { "0" } else { int_norm };
        let frac_norm = frac_part.trim_end_matches('0');
        if frac_norm.is_empty() {
            return int_norm.to_string();
        }
        format!("{}.{}", int_norm, frac_norm)
    } else {
        raw.to_string()
    }
}

#[allow(dead_code)]
fn detect_time_unit_suffix(text: &str, after: usize, id_seed: usize) -> Option<TimeUnitSuffix> {
    let bytes = text.as_bytes();
    if after >= bytes.len() {
        return None;
    }
    if !bytes[after].is_ascii_alphabetic() {
        return None;
    }

    let start = after;
    let mut end = after + 1;
    while end < bytes.len() && bytes[end].is_ascii_alphabetic() {
        end += 1;
    }

    let suffix = text[start..end].to_ascii_lowercase();
    let (concept_id, preferred) = match suffix.as_str() {
        "d" | "day" | "days" => ("time_unit:258703001".to_string(), "Day".to_string()),
        "h" | "hr" | "hrs" | "hour" | "hours" => ("time_unit:258702006".to_string(), "Hour".to_string()),
        "wk" | "wks" | "week" | "weeks" => ("time_unit:258705008".to_string(), "Week".to_string()),
        "mo" | "mos" | "month" | "months" => ("time_unit:258706009".to_string(), "Month".to_string()),
        "min" | "mins" | "minute" | "minutes" | "m" => ("time_unit:LOCAL_MINUTE".to_string(), "Minute".to_string()),
        _ => return None,
    };

    Some(TimeUnitSuffix {
        id: format!("tu{}", id_seed),
        start,
        end,
        concept_id,
        preferred,
    })
}



#[derive(Serialize)]
struct CanonicalInput {
    sig: String,
    spans: Vec<CanonicalInputSpan>,
}

fn build_canonical_input(sig_text: &str, spans: &[SigSpanOutput]) -> CanonicalInput {
    let rendered: Vec<CanonicalInputSpan> = spans
        .iter()
        .filter(|s| s.category != "residual_text")
        .filter_map(|s| {
            canonical_render_value(s).map(|render| CanonicalInputSpan {
                span_id: s.span_id.clone(),
                category: s.category.clone(),
                role: s.role.clone(),
                text: s.span.text.clone(),
                render,
                start_char: s.span.start,
                end_char: s.span.end,
                tags: s.tags.clone(),
            })
        })
        .collect();

    CanonicalInput {
        sig: sig_text.to_string(),
        spans: rendered,
    }
}


#[derive(Serialize)]
struct CanonicalInputSpan {
    span_id: String,
    category: String,
    role: String,
    text: String,
    render: String,
    start_char: usize,
    end_char: usize,
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct CanonicalModelOutput {
    doc_id: Option<String>,
    canonical: CanonicalModelCanonical,
    errors: Vec<OutputError>,
}

#[derive(Deserialize)]
struct CanonicalModelCanonical {
    text: String,
    confidence: f64,
    extra_notes: Vec<String>,
}

async fn llm_render_canonical(
    sig_text: &str,
    spans: &[SigSpanOutput],
    args: &SigConceptExtractCommand,
    codex_linux_sandbox_exe: Option<PathBuf>,
) -> Result<CanonicalSig> {
    // Build a stable deterministic baseline first. The model is asked to do minimal edits.
    let draft = render_canonical_draft(sig_text, spans);

    // Normalized spans are provided as "facts" the model may rely on.
    let input = build_canonical_input(sig_text, spans);
    let input_json = serde_json::to_string(&input)?;

    // Extra / residual segments are passed separately, so the model can place them into extra_notes.
    let extra_segments = collect_extra_segments(sig_text, spans);
    let extra_json = serde_json::to_string(&extra_segments)?;

    // A small deterministic summary helps the model preserve complex multi-clause intent
    // (tapers, repeats, max constraints, day-of-week schedules, etc.)
    let structure = compute_sig_structure(sig_text, spans);

    let steps_summary = structure
        .steps
        .iter()
        .map(|s| {
            serde_json::json!({
                "step_id": s.step_id,
                "kind": s.kind,
                "text": s.span.text,
            })
        })
        .collect::<Vec<_>>();

    let structure_summary = serde_json::json!({
        "kind": structure.kind,
        "flags": structure.flags,
        "steps": steps_summary,
    });

    let structure_json = serde_json::to_string(&structure_summary)?;

    let prompt = build_canonical_prompt(sig_text, &input_json, &draft, &extra_json, &structure_json);
    let schema_path = write_schema_tempfile("codex-sig-canonical-schema", CANONICAL_SCHEMA_JSON)?;

    let mut out_path = std::env::temp_dir();
    out_path.push(format!(
        "codex-sig-canonical-output-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));

    let mut exec_cli = ExecCli::try_parse_from(["codex", "exec"])?;
    exec_cli.prompt = Some(prompt);
    exec_cli.output_schema = Some(schema_path);
    exec_cli.last_message_file = Some(out_path.clone());
    exec_cli.model = args.model.clone();
    exec_cli.oss = args.oss;
    exec_cli.oss_provider = args.oss_provider.clone();
    exec_cli.sandbox_mode = args.sandbox_mode;
    exec_cli.config_profile = args.config_profile.clone();
    exec_cli.full_auto = args.full_auto;
    exec_cli.dangerously_bypass_approvals_and_sandbox =
        args.dangerously_bypass_approvals_and_sandbox;
    exec_cli.cwd = args.cwd.clone();
    exec_cli.skip_git_repo_check = args.skip_git_repo_check;
    exec_cli.add_dir = args.add_dir.clone();
    // IMPORTANT: internal calls must never stream JSON events to stdout; keep it off.
    exec_cli.json = false;
    exec_cli.config_overrides = args.config_overrides.clone();

    run_exec_silently(exec_cli, codex_linux_sandbox_exe).await?;

    let raw = fs::read_to_string(&out_path)
        .with_context(|| format!("Failed to read canonical output from {}", out_path.display()))?;
    let parsed: CanonicalModelOutput =
        serde_json::from_str(&raw).context("Failed to parse canonical JSON output")?;
    let _ = parsed.doc_id;

    if !parsed.errors.is_empty() {
        // Treat model-reported errors as hard failures, so caller can fall back deterministically.
        let msg = parsed
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.code, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        anyhow::bail!("Canonicalizer reported errors: {msg}");
    }

    let mut text = parsed.canonical.text.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("Canonicalizer returned empty text");
    }

    // Normalize punctuation/whitespace and enforce a trailing period.
    text = normalize_sentence(&text);

    let mut extra_notes: Vec<String> = parsed
        .canonical
        .extra_notes
        .into_iter()
        .map(|s| normalize_sentence(&s))
        .filter(|s| !s.is_empty())
        .collect();
    extra_notes = dedupe_strings(extra_notes);

    // Always surface leftover residual content for transparency to avoid misleading canonicals.
    // Only include segments that contain non-stopword content and are not already represented.
    for seg in &extra_segments {
        let norm = normalize_sentence(seg);
        if !extra_segment_is_informative(&norm) {
            continue;
        }
        let already = extra_notes
            .iter()
            .any(|e| e.eq_ignore_ascii_case(&norm) || e.to_ascii_lowercase().contains(&norm.to_ascii_lowercase()));
        let in_text = text.to_ascii_lowercase().contains(&norm.to_ascii_lowercase());
        if !already && !in_text {
            extra_notes.push(format!("Unparsed: {}", norm));
        }
    }
    extra_notes = dedupe_strings(extra_notes);

    Ok(CanonicalSig {
        text,
        confidence: parsed.canonical.confidence,
        source: "llm".to_string(),
        extra_notes,
        spans: vec![],
    })
}

fn canonical_render_value(span: &SigSpanOutput) -> Option<String> {
    match &span.normalized {
        NormalizedOut::Coded {
            concept_id,
            preferred,
        } => {
            let cat = span.category.as_str();
            match cat {
                "route" => Some(render_route_phrase(concept_id, preferred)),
                "freq" => {
                    // Expand common shorthand to reduce canonical "AI slop".
                    let p = match concept_id.as_str() {
                        "freq:BID" => "twice daily".to_string(),
                        "freq:TID" => "three times daily".to_string(),
                        "freq:QID" => "four times daily".to_string(),
                        "freq:DAILY" => "daily".to_string(),
                        "freq:NIGHTLY" => "nightly".to_string(),
                        "freq:WEEKLY" => "weekly".to_string(),
                        "freq:TWICE_WEEKLY" => "twice weekly".to_string(),
                        "freq:PRN" => "as needed".to_string(),
                        _ => preferred.to_ascii_lowercase(),
                    };
                    Some(p)
                }
                "action" => Some(preferred.clone()),
                "dose_unit" => Some(preferred.to_ascii_lowercase()),
                "dx" | "time_of_day" | "timing_event" | "food" | "site" => {
                    Some(preferred.to_ascii_lowercase())
                }
                _ => Some(preferred.clone()),
            }
        }
        NormalizedOut::Numeric { normalized, .. } => Some(normalized.clone()),
        NormalizedOut::Date { iso, .. } => Some(iso.clone()),
        NormalizedOut::Icd { code, .. } => Some(code.clone()),
        // Text/residual/rejected are intentionally excluded
        NormalizedOut::Text { .. } => None,
        NormalizedOut::Rejected { .. } => None,
    }
}

fn build_canonical_prompt(
    sig_text: &str,
    normalized_spans_json: &str,
    canonical_draft: &str,
    extra_segments_json: &str,
    structure_summary_json: &str,
) -> String {
    // NOTE: The canonicalizer is asked to do minimal edits on the draft and to
    // push anything it cannot represent (or anything clearly non-core) into extra_notes.
    //
    // This greatly improves grammar/casing consistency vs asking for a free-form canonical string.
    format!(
        "You are a medication SIG canonicalizer and copyeditor.
\

\
Your output will be consumed by a strict JSON parser.
\

\
Objective:
\
- Produce a clean, human-readable canonical instruction (canonical.text).
\
- Do not change meaning.
\
- Keep the clinical facts aligned to the normalized spans.
\
- Capture leftover / non-core content as canonical.extra_notes (cleaned).
\

\
Hard rules:
\
1) Output ONLY JSON that matches the provided JSON Schema.
\
2) Do NOT invent new clinical facts (dose, route, frequency, duration, diagnosis, device, site, timing).
\
3) Prefer sentence case. No random capitalization.
\
4) canonical.text must end with a period.
\
5) Use CANONICAL_DRAFT as your baseline. Make the smallest edits needed for grammar and clarity.
\
6) If CANONICAL_DRAFT is already good, return it verbatim.
\
7) If some information in SIG cannot be represented cleanly in one sentence, keep canonical.text focused on the core instruction and move the rest to canonical.extra_notes.
\
8) Only use information supported by NORMALIZED_SPANS_JSON and the original SIG text.
\

\
SIG (full, original context):
\
{sig}
\

\
CANONICAL_DRAFT (baseline):
\
{draft}
\

\
SIG_STRUCTURE_SUMMARY_JSON (deterministic complexity + clause segmentation hints):
\
{structure}
\

\
NORMALIZED_SPANS_JSON (facts you may rely on):
\
{json}
\

\
EXTRA_SEGMENTS_JSON (unmodeled / residual content you may optionally surface as extra_notes):
\
{extra}
",
        sig = sig_text,
        draft = canonical_draft,
        structure = structure_summary_json,
        json = normalized_spans_json,
        extra = extra_segments_json
    )
}

// ---------------------------------------------------------------------------
// Canonicalization helpers
// ---------------------------------------------------------------------------

fn normalize_sentence(s: &str) -> String {
    let mut t = s.trim().replace('\u{00A0}', " ");

    // Collapse whitespace.
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");

    // Fix common spacing issues around punctuation.
    for (from, to) in [
        (" ,", ","),
        (" .", "."),
        (" ;", ";"),
        (" :", ":"),
        (" )", ")"),
        ("( ", "("),
        (" /", "/"),
        ("/ ", "/"),
    ] {
        t = t.replace(from, to);
    }

    // Ensure the sentence ends with punctuation (prefer a period).
    if !t.ends_with('.') && !t.ends_with('!') && !t.ends_with('?') {
        t.push('.');
    }

    // Ensure first character is capitalized when it is ASCII alphabetic.
    if let Some(first) = t.chars().next() {
        if first.is_ascii_lowercase() {
            let mut chars = t.chars();
            let first_up = chars
                .next()
                .unwrap()
                .to_ascii_uppercase()
                .to_string();
            t = first_up + chars.as_str();
        }
    }

    t
}

fn dedupe_strings(items: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for s in items {
        let key = s.trim().to_ascii_lowercase();
        if key.is_empty() {
            continue;
        }
        if seen.insert(key) {
            out.push(s);
        }
    }
    out
}

fn is_stopword_token(token: &str) -> bool {
    matches!(
        token,
        "a" | "an" | "the" | "and" | "or" | "of" | "to" | "in" | "on" | "at" | "for" | "from" | "by"
            | "with" | "without" | "as" | "per" | "then" | "if" | "when" | "while" | "until"
            | "before" | "after" | "during" | "up" | "down" | "off" | "out" | "into" | "onto"
            | "each" | "every" | "any" | "all"
    )
}

fn extra_segment_is_informative(seg: &str) -> bool {
    let s = seg.trim();
    if s.is_empty() {
        return false;
    }
    let lower = s.to_ascii_lowercase();
    let mut any = false;
    let mut all_stop = true;
    for raw in lower.split_whitespace() {
        let tok = raw.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        if tok.is_empty() {
            continue;
        }
        any = true;
        if !is_stopword_token(tok) {
            all_stop = false;
        }
    }
    any && !all_stop
}

fn render_canonical_draft(sig_text: &str, spans: &[SigSpanOutput]) -> String {
    // Prefer fully-deterministic canonicalization when the SIG is simple enough.
    if let Some(det) = render_canonical_simple(spans) {
        return det.text;
    }

    // For more complex SIGs, build a *deterministic* multi-clause draft by segmenting
    // the original SIG and attempting simple-rendering per clause. This gives the LLM
    // a much cleaner starting point (reduces random capitalization / grammar drift).
    let structure = compute_sig_structure(sig_text, spans);
    if structure.steps.len() <= 1 {
        return normalize_sentence(sig_text);
    }

    let mut parts: Vec<String> = Vec::new();

    for step in structure.steps {
        // Pull spans that fall inside the step range.
        let step_spans: Vec<SigSpanOutput> = spans
            .iter()
            .filter(|s| s.span.start >= step.span.start && s.span.end <= step.span.end)
            .cloned()
            .collect();

        if let Some(det) = render_canonical_simple(&step_spans) {
            parts.push(det.text);
        } else {
            parts.push(normalize_sentence(&step.span.text));
        }
    }

    // Join as multiple sentences.
    let joined = parts
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    normalize_sentence(&joined)
}

fn collect_extra_segments(sig_text: &str, spans: &[SigSpanOutput]) -> Vec<String> {
    let char_to_byte = build_char_to_byte_map(sig_text);

    let mut residual: Vec<(usize, usize)> = spans
        .iter()
        .filter(|s| s.category == "residual_text")
        .map(|s| {
            let sb = *char_to_byte.get(s.span.start).unwrap_or(&0);
            let eb = *char_to_byte.get(s.span.end).unwrap_or(&sig_text.len());
            (sb, eb)
        })
        .collect();
    residual.sort_by_key(|(s, _)| *s);

    // Merge adjacent residual tokens into more readable "phrases".
    let mut phrases: Vec<String> = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut cur_end: usize = 0;

    for (s, e) in residual {
        if cur_start.is_none() {
            cur_start = Some(s);
            cur_end = e;
            continue;
        }
        let gap = s.saturating_sub(cur_end);
        let between = &sig_text[cur_end..s];

        // Merge if the gap is small and contains only whitespace or light punctuation.
        let mergeable = gap <= 4
            && between.chars().all(|c| {
                c.is_whitespace()
                    || matches!(c, '/' | '-' | ',' | '(' | ')' | '[' | ']' | '{' | '}' | ':')
            });

        if mergeable {
            cur_end = e;
        } else {
            let cs = cur_start.unwrap();
            let seg = sig_text[cs..cur_end].trim().to_string();
            if !seg.is_empty() {
                phrases.push(seg);
            }
            cur_start = Some(s);
            cur_end = e;
        }
    }

    if let Some(cs) = cur_start {
        let seg = sig_text[cs..cur_end].trim().to_string();
        if !seg.is_empty() {
            phrases.push(seg);
        }
    }

    // Include structured aux_* spans explicitly (they are "covered", so they won't appear in residual gaps).
    for s in spans {
        if s.category.starts_with("aux_") {
            let txt = s.span.text.trim();
            if !txt.is_empty() {
                phrases.push(format!("{}: {}", s.category, txt));
            }
        }
    }

    dedupe_strings(
        phrases
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() >= 2)
            .filter(|s| extra_segment_is_informative(s))
            .collect(),
    )
}

fn render_canonical_simple(spans: &[SigSpanOutput]) -> Option<CanonicalSig> {
    // Pull out core components.
    let action = spans
        .iter()
        .find(|s| s.category == "action")
        .and_then(extract_coded);

    let dose_unit_span = spans.iter().find(|s| s.category == "dose_unit");
    let dose_unit = dose_unit_span.and_then(extract_coded);

    // Dose amount: closest numeric immediately before the dose_unit span (common SIG pattern).
    let dose_amount_span = if let Some(du) = dose_unit_span {
        spans
            .iter()
            .filter(|s| matches!(s.normalized, NormalizedOut::Numeric { .. }))
            .filter(|n| n.span.end <= du.span.start)
            .min_by_key(|n| du.span.start.saturating_sub(n.span.end))
    } else {
        spans
            .iter()
            .find(|s| matches!(s.normalized, NormalizedOut::Numeric { .. }))
    };
    let dose_amount = dose_amount_span.and_then(extract_numeric);

    let route = spans.iter().find(|s| s.category == "route").and_then(extract_coded);

    // Optional enrichers
    let site = spans.iter().find(|s| s.category == "site").and_then(extract_coded);
    let time_of_day = spans
        .iter()
        .filter(|s| s.category == "time_of_day")
        .filter_map(extract_coded)
        .collect::<Vec<_>>();
    let timing_event = spans
        .iter()
        .filter(|s| s.category == "timing_event")
        .filter_map(extract_coded)
        .collect::<Vec<_>>();
    let food = spans
        .iter()
        .filter(|s| s.category == "food")
        .filter_map(extract_coded)
        .collect::<Vec<_>>();
    let temporal = spans
        .iter()
        .filter(|s| s.category == "temporal")
        .map(|s| s.span.text.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();

    // Frequency: attempt to build a coherent phrase.
    let freq_spans: Vec<(String, String, String)> = spans
        .iter()
        .filter(|s| s.category == "freq")
        .filter_map(|s| extract_coded(s).map(|(cid, pref)| (s.span_id.clone(), cid, pref)))
        .collect();

    let time_units: Vec<(String, String, usize, usize)> = spans
        .iter()
        .filter(|s| s.category == "time_unit")
        .filter_map(|s| {
            extract_coded(s).map(|(cid, pref)| (cid, pref, s.span.start, s.span.end))
        })
        .collect();

    let numeric_spans: Vec<(String, String, usize, usize)> = spans
        .iter()
        .filter_map(|s| {
            extract_numeric(s).map(|(_kind, norm)| {
                (
                    s.span_id.clone(),
                    norm,
                    s.span.start,
                    s.span.end,
                )
            })
        })
        .collect();

    // Exclude the dose amount numeric span from interval detection.
    let dose_num_id = dose_amount_span.map(|s| s.span_id.clone());

    let freq_phrase = render_frequency_phrase(
        &freq_spans,
        &numeric_spans,
        &time_units,
        dose_num_id.as_deref(),
    );

    // Flags / PRN
    let flags_pref = spans
        .iter()
        .filter(|s| s.category == "flags")
        .filter_map(extract_coded)
        .map(|(_cid, pref)| pref)
        .collect::<Vec<_>>();
    let has_prn = freq_spans.iter().any(|(_, cid, _)| cid == "freq:PRN")
        || flags_pref
            .iter()
            .any(|p| p.to_ascii_lowercase().contains("as needed"));
    let prn_phrase = if has_prn { Some("as needed".to_string()) } else { None };

    // Diagnosis/indication (dx)
    let dx_list = spans
        .iter()
        .filter(|s| s.category == "dx")
        .filter_map(extract_coded)
        .map(|(_cid, pref)| pref.to_ascii_lowercase())
        .collect::<Vec<_>>();
    let dx_phrase = if !dx_list.is_empty() {
        Some(format!("for {}", dx_list.join(" or ")))
    } else {
        None
    };

    // Assemble pieces.
    let mut pieces: Vec<(String, Option<String>)> = Vec::new();

    // Action
    if let Some((_, act_pref)) = action.clone() {
        pieces.push((act_pref, action.as_ref().map(|_| "action".to_string())));
    } else if dose_unit.is_some() || dose_amount.is_some() {
        pieces.push(("Take".to_string(), Some("action".to_string())));
    }

    // Dose amount + unit
    if let Some((_, qty_norm)) = dose_amount.clone() {
        pieces.push((qty_norm.clone(), dose_amount_span.map(|s| s.span_id.clone())));
    }
    if let Some((_, du_pref)) = dose_unit.clone() {
        let qty_norm = dose_amount.as_ref().map(|(_, n)| n.as_str()).unwrap_or("1");
        let du = pluralize_simple(&du_pref.to_ascii_lowercase(), qty_norm);
        pieces.push((du, dose_unit_span.map(|s| s.span_id.clone())));
    }

    // Site (often for eyes/ears/nostrils/skin). Keep very conservative wording.
    if let Some((_, site_pref)) = site.clone() {
        pieces.push((format!("to {}", site_pref.to_ascii_lowercase()), Some("site".to_string())));
    }

    // Route
    if let Some((route_cid, route_pref)) = route.clone() {
        let route_phrase = render_route_phrase(&route_cid, &route_pref);
        pieces.push((route_phrase, Some("route".to_string())));
    }

    // Frequency / timing
    if let Some(freq) = freq_phrase {
        pieces.push((freq, Some("freq".to_string())));
    }

    // Time-of-day
    if !time_of_day.is_empty() {
        let tod_phrase = render_time_of_day_phrase(&time_of_day);
        if let Some(tod) = tod_phrase {
            pieces.push((tod, Some("time_of_day".to_string())));
        }
    }

    // Timing event
    if !timing_event.is_empty() {
        let ev = timing_event
            .iter()
            .map(|(_, p)| p.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" and ");
        pieces.push((ev, Some("timing_event".to_string())));
    }

    // Food instructions
    if !food.is_empty() {
        let fp = food
            .iter()
            .map(|(_, p)| p.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" and ");
        pieces.push((fp, Some("food".to_string())));
    }

    // PRN
    if let Some(prn) = prn_phrase {
        pieces.push((prn, Some("prn".to_string())));
    }

    // Indication
    if let Some(dx) = dx_phrase {
        pieces.push((dx, Some("dx".to_string())));
    }

    // Temporal / duration fragments (leave verbatim-ish; many are already phrases like "for 7 days")
    if !temporal.is_empty() {
        let tp = temporal
            .iter()
            .map(|s| s.to_ascii_lowercase())
            .collect::<Vec<_>>()
            .join(" ");
        pieces.push((tp, Some("temporal".to_string())));
    }

    // Must have something meaningful.
    if pieces.is_empty() {
        return None;
    }

    // Join pieces and normalize.
    let out_raw = pieces
        .iter()
        .map(|(t, _)| t.clone())
        .filter(|t| !t.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let out = normalize_sentence(&out_raw);

    // Build canonical spans by locating each piece (best-effort).
    let mut canon_spans: Vec<CanonicalSpan> = Vec::new();
    let mut cursor = 0usize;
    for (piece, _) in &pieces {
        if piece.trim().is_empty() {
            continue;
        }
        if let Some(pos) = out[cursor..].find(piece) {
            let start = cursor + pos;
            let end = start + piece.len();
            canon_spans.push(CanonicalSpan {
                category: "text".to_string(),
                start,
                end,
                text: piece.to_string(),
                concept_id: None,
                preferred: None,
            });
            cursor = end;
        }
    }

    Some(CanonicalSig {
        text: out,
        confidence: 1.0,
        source: "deterministic".to_string(),
        extra_notes: vec![],
        spans: canon_spans,
    })
}

fn extract_coded(span: &SigSpanOutput) -> Option<(String, String)> {
    match &span.normalized {
        NormalizedOut::Coded {
            concept_id,
            preferred,
        } => Some((concept_id.clone(), preferred.clone())),
        _ => None,
    }
}

fn extract_numeric(span: &SigSpanOutput) -> Option<(String, String)> {
    match &span.normalized {
        NormalizedOut::Numeric {
            number_kind,
            normalized,
        } => Some((number_kind.clone(), normalized.clone())),
        _ => None,
    }
}

fn render_frequency_phrase(
    freq_spans: &[(String, String, String)],
    numeric_spans: &[(String, String, usize, usize)],
    time_units: &[(String, String, usize, usize)],
    exclude_numeric_span_id: Option<&str>,
) -> Option<String> {
    // Helper: does this set contain a freq concept?
    let has = |cid: &str| freq_spans.iter().any(|(_, c, _)| c == cid);

    // Prefer fully-specified frequency concepts first.
    if has("freq:BID") {
        return Some("twice daily".to_string());
    }
    if has("freq:TID") {
        return Some("three times daily".to_string());
    }
    if has("freq:QID") {
        return Some("four times daily".to_string());
    }
    if has("freq:TWICE_WEEKLY") {
        return Some("twice weekly".to_string());
    }
    if has("freq:WEEKLY") {
        return Some("weekly".to_string());
    }
    if has("freq:NIGHTLY") {
        return Some("nightly".to_string());
    }
    if has("freq:WITH_EACH_DOSE") {
        return Some("with each dose".to_string());
    }

    // "every <n> <unit>"
    if has("freq:EVERY") {
        if let Some((qty, unit)) =
            detect_interval_phrase(numeric_spans, time_units, exclude_numeric_span_id)
        {
            return Some(format!("every {} {}", qty, unit));
        }
        // Handle patterns like "every day" where there is a time_unit but no explicit numeric.
        if let Some((_cid, upref, _us, _ue)) = time_units.first() {
            return Some(format!("every {}", upref.to_ascii_lowercase()));
        }
        return Some("every".to_string());
    }

    if has("freq:PER") {
        if let Some((_cid, upref, _us, _ue)) = time_units.first() {
            return Some(format!("per {}", upref.to_ascii_lowercase()));
        }
        return Some("per".to_string());
    }

    // Multiplier + daily/weekly combos.
    if has("freq:DAILY") {
        if has("freq:TWICE") {
            return Some("twice daily".to_string());
        }
        if has("freq:THREE_TIMES") {
            return Some("three times daily".to_string());
        }
        if has("freq:FOUR_TIMES") {
            return Some("four times daily".to_string());
        }
        return Some("daily".to_string());
    }

    if has("freq:ONCE") {
        // Ambiguous: leave as "once" (single dose) unless another frequency is present.
        return Some("once".to_string());
    }
    if has("freq:TWICE") {
        return Some("twice".to_string());
    }
    if has("freq:THREE_TIMES") {
        return Some("three times".to_string());
    }
    if has("freq:FOUR_TIMES") {
        return Some("four times".to_string());
    }

    None
}

fn detect_interval_phrase(
    numeric_spans: &[(String, String, usize, usize)],
    time_units: &[(String, String, usize, usize)],
    exclude_numeric_span_id: Option<&str>,
) -> Option<(String, String)> {
    // Choose a numeric+unit pair that look adjacent (e.g., "6 h", "6 hours", "q6h").
    let mut best: Option<(String, String, usize)> = None;

    for (num_id, qty, _ns, ne) in numeric_spans {
        if exclude_numeric_span_id.is_some_and(|x| x == num_id) {
            continue;
        }
        for (_ucid, upref, us, _ue) in time_units {
            if *us < *ne {
                continue;
            }
            let gap = us.saturating_sub(*ne);
            // Small gaps are typical for "6h" (gap=0) or "6 h" (gap=1).
            if gap > 2 {
                continue;
            }
            let unit = pluralize_simple(&upref.to_ascii_lowercase(), qty);
            let candidate = (qty.clone(), unit, gap);
            if best.is_none() || gap < best.as_ref().unwrap().2 {
                best = Some(candidate);
            }
        }
    }

    // If we didn't find an adjacent pair but there is exactly one time_unit, fall back to it.
    if best.is_none() && time_units.len() == 1 {
        let (_ucid, upref, _us, _ue) = &time_units[0];
        if let Some((num_id, qty, _ns, _ne)) = numeric_spans
            .iter()
            .find(|(id, _, _, _)| !exclude_numeric_span_id.is_some_and(|x| x == id))
        {
            let unit = pluralize_simple(&upref.to_ascii_lowercase(), qty);
            best = Some((qty.clone(), unit, 999));
            let _ = num_id;
        }
    }

    best.map(|(q, u, _)| (q, u))
}

fn parse_numeric_value(norm: &str) -> Option<f64> {
    let s = norm.trim();
    if s.is_empty() {
        return None;
    }
    if let Some((a, b)) = s.split_once('/') {
        let na: f64 = a.trim().parse().ok()?;
        let nb: f64 = b.trim().parse().ok()?;
        if nb == 0.0 {
            return None;
        }
        return Some(na / nb);
    }
    // Range like "1-2": take the max for pluralization heuristics.
    if let Some((a, b)) = s.split_once('-') {
        let na: f64 = a.trim().parse().ok()?;
        let nb: f64 = b.trim().parse().ok()?;
        return Some(na.max(nb));
    }
    s.parse::<f64>().ok()
}

fn pluralize_simple(word: &str, qty_norm: &str) -> String {
    // Only pluralize simple alphabetic words (tablet -> tablets).
    let needs_plural = parse_numeric_value(qty_norm)
        .map(|v| v > 1.0 + 1e-9)
        .unwrap_or(false);

    if !needs_plural {
        return word.to_string();
    }

    if !word.chars().all(|c| c.is_ascii_alphabetic()) {
        return word.to_string();
    }
    if word.ends_with('s') {
        return word.to_string();
    }
    format!("{word}s")
}

fn render_time_of_day_phrase(time_of_day: &[(String, String)]) -> Option<String> {
    if time_of_day.is_empty() {
        return None;
    }
    let phrases = time_of_day
        .iter()
        .map(|(_cid, pref)| {
            let p = pref.to_ascii_lowercase();
            match p.as_str() {
                "morning" => "in the morning".to_string(),
                "afternoon" => "in the afternoon".to_string(),
                "evening" => "in the evening".to_string(),
                "night" => "at night".to_string(),
                "bedtime" => "at bedtime".to_string(),
                _ => p,
            }
        })
        .collect::<Vec<_>>();

    Some(phrases.join(" and "))
}
fn render_route_phrase(concept_id: &str, preferred: &str) -> String {
    match concept_id {
        "route:26643006" => "by mouth".to_string(),
        "route:6064005" => "topically".to_string(),
        "route:37839007" => "under the tongue".to_string(), // Sublingual
        "route:37161004" => "rectally".to_string(),         // Per rectum
        "route:16857009" => "vaginally".to_string(),        // Per vagina
        "route:46713006" => "intranasally".to_string(),
        "route:54485002" => "in the eye".to_string(),
        "route:10547007" => "in the ear".to_string(),
        "route:447694001" => "by inhalation".to_string(), // Respiratory tract route
        "route:34206005" => "subcutaneously".to_string(),
        "route:47625008" => "intravenously".to_string(),
        "route:78421000" => "intramuscularly".to_string(),
        "route:45890007" => "transdermally".to_string(),
        _ => preferred.to_ascii_lowercase(),
    }
}
