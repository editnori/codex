use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use codex_common::CliConfigOverrides;
use codex_common::SandboxModeCliArg;
use codex_exec::Cli as ExecCli;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::IsTerminal;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_DICT_SCRIPT: &str = "KidneyStone_SDOH/KidneyStone_SDOH.script";

const OUTPUT_SCHEMA_JSON: &str = r#"{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ConceptSelections",
  "type": "object",
  "additionalProperties": false,
  "required": ["doc_id", "selections", "errors"],
  "properties": {
    "doc_id": { "type": ["string", "null"] },
    "selections": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "mention_id",
          "concept_id",
          "assertion",
          "temporality",
          "experiencer",
          "confidence"
        ],
        "properties": {
          "mention_id": { "type": "string", "minLength": 1 },
          "concept_id": { "type": "string", "minLength": 1 },
          "assertion": {
            "type": "string",
            "enum": ["PRESENT", "ABSENT", "POSSIBLE", "CONDITIONAL", "HYPOTHETICAL", "GENERIC", "UNKNOWN"]
          },
          "temporality": {
            "type": "string",
            "enum": ["CURRENT", "HISTORICAL", "FUTURE", "PLANNED", "UNKNOWN"]
          },
          "experiencer": {
            "type": "string",
            "enum": ["PATIENT", "FAMILY", "OTHER", "UNKNOWN"]
          },
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
pub struct ConceptExtractCommand {
    /// Input text as a literal string. Conflicts with --input.
    #[arg(long = "text", value_name = "TEXT", conflicts_with = "input")]
    pub text: Option<String>,

    /// Path to input text file. Use '-' to read from stdin.
    #[arg(
        long = "input",
        short = 'i',
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath
    )]
    pub input: Option<PathBuf>,

    /// Path to an HSQLDB .script dictionary file.
    #[arg(
        long = "dict-script",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath
    )]
    pub dict_script: Option<PathBuf>,

    /// Max candidates per mention span.
    #[arg(long = "max-candidates", default_value_t = 5)]
    pub max_candidates: usize,

    /// Max mention spans to include in the prompt.
    #[arg(long = "max-mentions", default_value_t = 200)]
    pub max_mentions: usize,

    /// Minimum dictionary term length to match.
    #[arg(long = "min-term-length", default_value_t = 3)]
    pub min_term_length: usize,

    /// Optional output file for the final JSON result.
    #[arg(
        long = "output",
        short = 'o',
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath
    )]
    pub output: Option<PathBuf>,

    /// Output all dictionary candidates without LLM adjudication.
    #[arg(long = "all-candidates", alias = "all", default_value_t = false)]
    pub all_candidates: bool,

    /// Optional output schema override. If omitted, uses the built-in schema.
    #[arg(
        long = "schema",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath
    )]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SpanKey {
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct MentionTemp {
    start: usize,
    end: usize,
    text: String,
    candidates: HashMap<String, CandidateTemp>,
}

#[derive(Debug)]
struct CandidateTemp {
    concept_id: String,
    matched_term: String,
    score: usize,
}

#[derive(Serialize)]
struct CandidatePayload {
    mentions: Vec<MentionCandidates>,
}

#[derive(Serialize)]
struct MentionCandidates {
    mention_id: String,
    text: String,
    section_title: Option<String>,
    sentence: String,
    window: String,
    candidates: Vec<Candidate>,
}

#[derive(Serialize, Clone, Debug)]
struct Candidate {
    concept_id: String,
    matched_term: String,
    preferred: String,
    semantic_types: Vec<String>,
    rxnorm: Option<String>,
    score: usize,
}

#[derive(Debug)]
struct Mention {
    mention_id: String,
    start: usize,
    end: usize,
    text: String,
    section_title: Option<String>,
    sentence: String,
    window: String,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Default)]
struct CuiMetadata {
    prefterm: HashMap<String, String>,
    tui: HashMap<String, Vec<String>>,
    rxnorm: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SelectionOutput {
    doc_id: Option<String>,
    selections: Vec<Selection>,
    errors: Vec<SelectionError>,
}

#[derive(Debug, Deserialize)]
struct Selection {
    mention_id: String,
    concept_id: String,
    assertion: String,
    temporality: String,
    experiencer: String,
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct SelectionError {
    code: String,
    message: String,
}

pub async fn run(
    args: ConceptExtractCommand,
    codex_linux_sandbox_exe: Option<PathBuf>,
) -> Result<()> {
    let started = Instant::now();
    let input_text = read_input_text(args.text.as_ref(), args.input.as_ref())?;
    let dict_script = resolve_dict_script(args.dict_script.as_ref())?;

    eprintln!("Scanning dictionary: {}", dict_script.display());
    let (mentions_map, cuis) = scan_cui_terms(
        &dict_script,
        &input_text,
        args.min_term_length,
        args.max_mentions,
    )?;
    eprintln!(
        "Dictionary scan complete in {} ms ({} mentions, {} CUIs).",
        started.elapsed().as_millis(),
        mentions_map.len(),
        cuis.len()
    );

    let byte_to_char = build_byte_to_char_map(&input_text);

    if mentions_map.is_empty() {
        let empty = serde_json::json!({
            "doc_id": null,
            "concepts": [],
            "errors": [
                {
                    "code": "no_candidates",
                    "message": "No dictionary matches found in the input text."
                }
            ]
        });
        write_output(&empty, args.output.as_ref())?;
        return Ok(());
    }

    let meta = load_cui_metadata(&dict_script, &cuis)?;
    let sections = detect_sections(&input_text);
    let mentions = build_mentions(
        mentions_map,
        &meta,
        &byte_to_char,
        &input_text,
        &sections,
        args.max_candidates,
        args.max_mentions,
    );
    let payload = build_payload(&mentions);

    if args.all_candidates {
        let output = build_all_output(&mentions);
        write_output(&output, args.output.as_ref())?;
        return Ok(());
    }

    let payload_json = serde_json::to_string(&payload)?;
    let prompt = build_prompt(&payload_json);

    let output_schema = match args.schema.as_ref() {
        Some(path) => path.clone(),
        None => write_schema_tempfile()?,
    };

    let mut llm_output_path = std::env::temp_dir();
    llm_output_path.push(format!(
        "codex-concept-selections-{}.json",
        std::process::id()
    ));

    let mut exec_cli = ExecCli::try_parse_from(["codex", "exec"])?;
    exec_cli.prompt = Some(prompt);
    exec_cli.output_schema = Some(output_schema);
    exec_cli.last_message_file = Some(llm_output_path.clone());
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
    exec_cli.json = args.json;
    exec_cli.config_overrides = args.config_overrides;

    codex_exec::run_main(exec_cli, codex_linux_sandbox_exe).await?;

    let selections_raw = fs::read_to_string(&llm_output_path)
        .with_context(|| format!("Failed to read selections from {}", llm_output_path.display()))?;
    if selections_raw.trim().is_empty() {
        anyhow::bail!("Selection output was empty; no concepts were produced.");
    }

    let selection_output: SelectionOutput = serde_json::from_str(&selections_raw)
        .context("Failed to parse model selections JSON")?;
    let final_output = build_final_output(&mentions, selection_output);
    write_output(&final_output, args.output.as_ref())?;
    Ok(())
}

fn build_all_output(mentions: &[Mention]) -> serde_json::Value {
    let mut concepts = Vec::new();
    for mention in mentions {
        for candidate in &mention.candidates {
            concepts.push(serde_json::json!({
                "mention_id": mention.mention_id.as_str(),
                "span": {
                    "start": mention.start,
                    "end": mention.end,
                    "text": mention.text.as_str(),
                },
                "concept_id": candidate.concept_id.as_str(),
                "preferred": candidate.preferred.as_str(),
                "semantic_types": &candidate.semantic_types,
                "rxnorm": candidate.rxnorm.as_deref(),
                "matched_term": candidate.matched_term.as_str(),
                "confidence": 0.0,
                "assertion": "UNKNOWN",
                "temporality": "UNKNOWN",
                "experiencer": "UNKNOWN",
                "section_title": mention.section_title.as_deref(),
                "sentence": mention.sentence.as_str(),
                "window": mention.window.as_str()
            }));
        }
    }

    serde_json::json!({
        "doc_id": null,
        "concepts": concepts,
        "errors": []
    })
}

fn build_prompt(candidates_json: &str) -> String {
    format!(
        "You are a strict clinical concept extractor.\n\
Follow these rules:\n\
1) Output ONLY JSON that matches the provided JSON Schema.\n\
2) Use ONLY mention_id values that appear in CANDIDATES_JSON.\n\
3) For each mention_id you select, choose ONLY a concept_id from that mention's candidates.\n\
4) Do NOT output any spans or offsets. Your job is selection + attributes only.\n\
5) If a mention is ambiguous or irrelevant, omit it.\n\
6) Use the section_title/sentence/window for context.\n\
7) Always include all required fields.\n\
\n\
Allowed values:\n\
- assertion: PRESENT | ABSENT | POSSIBLE | CONDITIONAL | HYPOTHETICAL | GENERIC | UNKNOWN\n\
- temporality: CURRENT | HISTORICAL | FUTURE | PLANNED | UNKNOWN\n\
- experiencer: PATIENT | FAMILY | OTHER | UNKNOWN\n\
\n\
CANDIDATES_JSON:\n\
{candidates_json}\n"
    )
}

fn read_input_text(text: Option<&String>, input: Option<&PathBuf>) -> Result<String> {
    if let Some(literal) = text {
        return Ok(literal.to_string());
    }

    match input {
        Some(path) if path.as_path() == Path::new("-") => read_stdin(),
        Some(path) => fs::read_to_string(path)
            .with_context(|| format!("Failed to read input file {}", path.display())),
        None => {
            if std::io::stdin().is_terminal() {
                anyhow::bail!("No input provided. Use --text, --input, or pipe text via stdin.");
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
            "No dictionary provided. Use --dict-script or place {} relative to the working directory.",
            DEFAULT_DICT_SCRIPT
        );
    }
}

fn write_schema_tempfile() -> Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let filename = format!("codex-concept-schema-{}.json", std::process::id());
    path.push(filename);
    fs::write(&path, OUTPUT_SCHEMA_JSON)
        .with_context(|| format!("Failed to write schema to {}", path.display()))?;
    Ok(path)
}

fn write_output(value: &serde_json::Value, output: Option<&PathBuf>) -> Result<()> {
    let json = serde_json::to_string(value)?;
    if let Some(path) = output {
        fs::write(path, json.as_bytes())
            .with_context(|| format!("Failed to write output to {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn scan_cui_terms(
    dict_script: &Path,
    input_text: &str,
    min_term_length: usize,
    max_mentions: usize,
) -> Result<(HashMap<SpanKey, MentionTemp>, HashSet<String>)> {
    let input_lower = input_text.to_ascii_lowercase();
    let tokens = tokenize_ascii_words(&input_lower);
    let mut mentions: HashMap<SpanKey, MentionTemp> = HashMap::new();
    let mut cuis = HashSet::new();

    let file = File::open(dict_script)
        .with_context(|| format!("Failed to open dictionary {}", dict_script.display()))?;
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

        let cui = values[0].trim().to_string();
        let term = values[3].trim().to_string();
        let rword = values[4].trim().to_ascii_lowercase();

        if term.len() < min_term_length {
            continue;
        }
        if !tokens.contains(&rword) {
            continue;
        }

        let term_lower = term.to_ascii_lowercase();
        for start in find_term_matches(&input_lower, &term_lower) {
            let end = start + term_lower.len();
            if !input_text.is_char_boundary(start) || !input_text.is_char_boundary(end) {
                continue;
            }
            let key = SpanKey { start, end };
            if mentions.len() >= max_mentions && !mentions.contains_key(&key) {
                continue;
            }
            let mention = mentions.entry(key).or_insert_with(|| MentionTemp {
                start,
                end,
                text: input_text[start..end].to_string(),
                candidates: HashMap::new(),
            });
            let entry = mention
                .candidates
                .entry(cui.clone())
                .or_insert(CandidateTemp {
                    concept_id: cui.clone(),
                    matched_term: term.clone(),
                    score: term.len(),
                });
            if term.len() > entry.score {
                entry.matched_term = term.clone();
                entry.score = term.len();
            }
            cuis.insert(cui.clone());
        }
    }

    Ok((mentions, cuis))
}

fn load_cui_metadata(dict_script: &Path, cuis: &HashSet<String>) -> Result<CuiMetadata> {
    let file = File::open(dict_script)
        .with_context(|| format!("Failed to open dictionary {}", dict_script.display()))?;
    let reader = BufReader::new(file);
    let mut meta = CuiMetadata::default();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("INSERT INTO PREFTERM VALUES(") {
            if let Some(values) = parse_insert_values(&line) {
                if values.len() >= 2 && cuis.contains(values[0].trim()) {
                    meta.prefterm
                        .insert(values[0].trim().to_string(), values[1].trim().to_string());
                }
            }
            continue;
        }
        if line.starts_with("INSERT INTO TUI VALUES(") {
            if let Some(values) = parse_insert_values(&line) {
                if values.len() >= 2 && cuis.contains(values[0].trim()) {
                    let tui = values[1]
                        .trim()
                        .parse::<u32>()
                        .ok()
                        .map(|val| format!("T{val:03}"));
                    if let Some(tui) = tui {
                        meta.tui
                            .entry(values[0].trim().to_string())
                            .or_default()
                            .push(tui);
                    }
                }
            }
            continue;
        }
        if line.starts_with("INSERT INTO RXNORM VALUES(") {
            if let Some(values) = parse_insert_values(&line) {
                if values.len() >= 2 && cuis.contains(values[0].trim()) {
                    meta.rxnorm
                        .insert(values[0].trim().to_string(), values[1].trim().to_string());
                }
            }
        }
    }

    Ok(meta)
}

fn build_mentions(
    mentions_map: HashMap<SpanKey, MentionTemp>,
    meta: &CuiMetadata,
    byte_to_char: &[usize],
    input_text: &str,
    sections: &[Section],
    max_candidates: usize,
    max_mentions: usize,
) -> Vec<Mention> {
    let mut mentions: Vec<MentionTemp> = mentions_map.into_values().collect();
    mentions.sort_by_key(|mention| mention.start);
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
                .map(|candidate| {
                    let preferred = meta
                        .prefterm
                        .get(&candidate.concept_id)
                        .cloned()
                        .unwrap_or_else(|| candidate.matched_term.clone());
                    let semantic_types = meta
                        .tui
                        .get(&candidate.concept_id)
                        .cloned()
                        .unwrap_or_default();
                    let rxnorm = meta.rxnorm.get(&candidate.concept_id).cloned();
                    Candidate {
                        concept_id: candidate.concept_id,
                        matched_term: candidate.matched_term,
                        preferred,
                        semantic_types,
                        rxnorm,
                        score: candidate.score,
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

            let sentence = extract_sentence(input_text, mention.start, mention.end);
            let window = extract_window(input_text, mention.start, mention.end, 80);
            let section_title = find_section_title(sections, mention.start);

            Mention {
                mention_id: format!("m{}", idx + 1),
                start: byte_to_char[mention.start],
                end: byte_to_char[mention.end],
                text: mention.text,
                section_title,
                sentence,
                window,
                candidates,
            }
        })
        .collect()
}

fn build_payload(mentions: &[Mention]) -> CandidatePayload {
    let mentions = mentions
        .iter()
        .map(|mention| MentionCandidates {
            mention_id: mention.mention_id.clone(),
            text: mention.text.clone(),
            section_title: mention.section_title.clone(),
            sentence: mention.sentence.clone(),
            window: mention.window.clone(),
            candidates: mention.candidates.clone(),
        })
        .collect();
    CandidatePayload { mentions }
}

fn build_final_output(mentions: &[Mention], selections: SelectionOutput) -> serde_json::Value {
    let mut mention_map: HashMap<&str, &Mention> = HashMap::new();
    for mention in mentions {
        mention_map.insert(mention.mention_id.as_str(), mention);
    }

    let mut errors: Vec<serde_json::Value> = selections
        .errors
        .into_iter()
        .map(|err| serde_json::json!({ "code": err.code, "message": err.message }))
        .collect();

    let mut concepts = Vec::new();
    for selection in selections.selections {
        let mention = match mention_map.get(selection.mention_id.as_str()) {
            Some(mention) => *mention,
            None => {
                errors.push(serde_json::json!({
                    "code": "unknown_mention_id",
                    "message": format!("Mention id {} not found in candidates.", selection.mention_id)
                }));
                continue;
            }
        };

        let candidate = mention
            .candidates
            .iter()
            .find(|candidate| candidate.concept_id == selection.concept_id);
        let candidate = match candidate {
            Some(candidate) => candidate,
            None => {
                errors.push(serde_json::json!({
                    "code": "invalid_concept_id",
                    "message": format!(
                        "Concept id {} not in candidate list for mention {}.",
                        selection.concept_id, selection.mention_id
                    )
                }));
                continue;
            }
        };

        concepts.push(serde_json::json!({
            "mention_id": mention.mention_id.as_str(),
            "span": {
                "start": mention.start,
                "end": mention.end,
                "text": mention.text.as_str(),
            },
            "concept_id": candidate.concept_id.as_str(),
            "preferred": candidate.preferred.as_str(),
            "semantic_types": &candidate.semantic_types,
            "rxnorm": candidate.rxnorm.as_deref(),
            "matched_term": candidate.matched_term.as_str(),
            "confidence": selection.confidence,
            "assertion": selection.assertion,
            "temporality": selection.temporality,
            "experiencer": selection.experiencer,
            "section_title": mention.section_title.as_deref(),
            "sentence": mention.sentence.as_str(),
            "window": mention.window.as_str()
        }));
    }

    serde_json::json!({
        "doc_id": selections.doc_id,
        "concepts": concepts,
        "errors": errors
    })
}

fn tokenize_ascii_words(text: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let mut start = None;
    let bytes = text.as_bytes();
    for (idx, b) in bytes.iter().enumerate() {
        if b.is_ascii_alphanumeric() {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            tokens.insert(text[s..idx].to_string());
        }
    }
    if let Some(s) = start {
        tokens.insert(text[s..].to_string());
    }
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

fn boundaries_ok(
    haystack: &str,
    start: usize,
    end: usize,
    enforce_left: bool,
    enforce_right: bool,
) -> bool {
    let bytes = haystack.as_bytes();
    let left_ok = if enforce_left {
        start == 0 || !bytes[start - 1].is_ascii_alphanumeric()
    } else {
        true
    };
    let right_ok = if enforce_right {
        end >= bytes.len() || !bytes[end].is_ascii_alphanumeric()
    } else {
        true
    };
    left_ok && right_ok
}

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

#[derive(Debug, Clone)]
struct Section {
    start: usize,
    title: String,
}

fn detect_sections(text: &str) -> Vec<Section> {
    const SECTION_TITLES: &[&str] = &[
        "HISTORY OF PRESENT ILLNESS",
        "HPI",
        "CHIEF COMPLAINT",
        "PAST MEDICAL HISTORY",
        "PAST SURGICAL HISTORY",
        "PAST HISTORY",
        "FAMILY HISTORY",
        "SOCIAL HISTORY",
        "REVIEW OF SYSTEMS",
        "PHYSICAL EXAMINATION",
        "IMAGING STUDIES",
        "ASSESSMENT",
        "PLAN",
        "IMPRESSION",
        "DIAGNOSIS",
        "MEDICATIONS",
        "CURRENT MEDICATIONS",
        "PROBLEM LIST",
        "ALLERGIES",
        "HISTORY AND PHYSICAL",
    ];

    let upper = text.to_ascii_uppercase();
    let mut by_start: HashMap<usize, Section> = HashMap::new();
    for title in SECTION_TITLES {
        let mut search_start = 0;
        while let Some(pos) = upper[search_start..].find(title) {
            let start = search_start + pos;
            let end = start + title.len();
            if boundaries_ok(&upper, start, end, true, true)
                && looks_like_section_header(&upper, start, end)
            {
                let entry = by_start.entry(start).or_insert(Section {
                    start,
                    title: title.to_string(),
                });
                if title.len() > entry.title.len() {
                    entry.title = title.to_string();
                }
            }
            search_start = end;
        }
    }

    let mut sections: Vec<Section> = by_start.into_values().collect();
    sections.sort_by_key(|section| section.start);
    sections
}

fn looks_like_section_header(text: &str, start: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    let at_line_start = start == 0 || bytes.get(start.saturating_sub(1)) == Some(&b'\n');
    let mut idx = end;
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    let has_separator = matches!(bytes.get(idx), Some(b':') | Some(b'-'));
    at_line_start || has_separator
}

fn find_section_title(sections: &[Section], mention_start: usize) -> Option<String> {
    let mut current: Option<&Section> = None;
    for section in sections {
        if section.start <= mention_start {
            current = Some(section);
        } else {
            break;
        }
    }
    current.map(|section| section.title.clone())
}

fn extract_sentence(text: &str, start: usize, end: usize) -> String {
    let bytes = text.as_bytes();
    let mut s = start.min(bytes.len());
    while s > 0 {
        let b = bytes[s - 1];
        if b == b'.' || b == b'?' || b == b'!' || b == b'\n' {
            break;
        }
        s -= 1;
    }
    let mut e = end.min(bytes.len());
    while e < bytes.len() {
        let b = bytes[e];
        if b == b'.' || b == b'?' || b == b'!' || b == b'\n' {
            e += 1;
            break;
        }
        e += 1;
    }
    let s = clamp_to_char_boundary(text, s, false);
    let e = clamp_to_char_boundary(text, e, true);
    text[s..e].trim().to_string()
}

fn extract_window(text: &str, start: usize, end: usize, radius: usize) -> String {
    let mut s = start.saturating_sub(radius);
    let mut e = (end + radius).min(text.len());
    s = clamp_to_char_boundary(text, s, false);
    e = clamp_to_char_boundary(text, e, true);
    text[s..e].trim().to_string()
}

fn clamp_to_char_boundary(text: &str, mut idx: usize, forward: bool) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && idx < text.len() && !text.is_char_boundary(idx) {
        if forward {
            idx += 1;
        } else {
            idx -= 1;
        }
    }
    idx
}
