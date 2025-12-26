use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(about = "Run sig-concept-extract over JSONL", version)]
struct Args {
    #[arg(long = "in", value_name = "FILE")]
    in_path: PathBuf,

    #[arg(long = "out", value_name = "FILE")]
    out_path: PathBuf,

    #[arg(long = "failures-out", value_name = "FILE")]
    failures_out: Option<PathBuf>,

    #[arg(long = "codex", value_name = "FILE")]
    codex: Option<PathBuf>,

    #[arg(long = "dict-script", value_name = "FILE")]
    dict_script: PathBuf,

    #[arg(long = "overlay-script", value_name = "FILE")]
    overlay_script: PathBuf,

    #[arg(long = "sig-key", default_value = "sig")]
    sig_key: String,

    #[arg(long = "doc-id-key", default_value = "doc_id")]
    doc_id_key: String,

    #[arg(long = "learn-unknowns", default_value_t = true)]
    learn_unknowns: bool,

    #[arg(long = "learn-categories", default_value = "dx,aux_warning,aux_auxiliary,aux_provider,aux_misc")]
    learn_categories: String,

    #[arg(long = "learn-min-confidence", default_value_t = 0.95)]
    learn_min_confidence: f64,

    #[arg(long = "learn-tag-min-confidence", default_value_t = 0.90)]
    learn_tag_min_confidence: f64,

    #[arg(long = "canonical", default_value_t = true)]
    canonical: bool,

    #[arg(long = "limit", default_value_t = 0)]
    limit: usize,

    #[arg(long = "timeout", default_value_t = 120)]
    timeout: u64,

    #[arg(long = "retries", default_value_t = 1)]
    retries: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let codex_path = args.codex.unwrap_or_else(|| {
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("codex"));
        exe.with_file_name("codex")
    });

    let failures_path = args
        .failures_out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("{}.fail.jsonl", args.out_path.display())));

    let in_file = File::open(&args.in_path)
        .with_context(|| format!("Failed to open input {}", args.in_path.display()))?;
    let mut out_file = File::create(&args.out_path)
        .with_context(|| format!("Failed to open output {}", args.out_path.display()))?;
    let mut fail_file = File::create(&failures_path)
        .with_context(|| format!("Failed to open failures {}", failures_path.display()))?;

    let reader = BufReader::new(in_file);
    let mut count = 0usize;
    let mut row_idx = 0usize;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let sig = match &obj {
            Value::String(s) => Some(s.clone()),
            Value::Object(map) => map
                .get(&args.sig_key)
                .and_then(|v| v.as_str().map(|s| s.to_string())),
            _ => None,
        };
        if sig.is_none() {
            continue;
        }
        let sig = sig.unwrap();

        let doc_id = match &obj {
            Value::Object(map) => map
                .get(&args.doc_id_key)
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .or_else(|| map.get("id").and_then(|v| v.as_str().map(|s| s.to_string()))),
            _ => None,
        };

        row_idx += 1;

        let mut last_err = None::<String>;
        let mut output = None::<String>;

        for attempt in 0..=args.retries {
            let mut cmd = if args.timeout > 0 {
                let mut c = Command::new("timeout");
                c.arg(format!("{}s", args.timeout));
                c
            } else {
                Command::new(&codex_path)
            };

            if args.timeout > 0 {
                cmd.arg(&codex_path);
            }

            cmd.arg("sig-concept-extract")
                .arg("--sig")
                .arg(&sig)
                .arg("--dict-script")
                .arg(&args.dict_script)
                .arg("--overlay-script")
                .arg(&args.overlay_script)
                .arg(format!("--learn-unknowns={}", args.learn_unknowns))
                .arg("--learn-categories")
                .arg(&args.learn_categories)
                .arg("--learn-min-confidence")
                .arg(args.learn_min_confidence.to_string())
                .arg("--learn-tag-min-confidence")
                .arg(args.learn_tag_min_confidence.to_string())
                .arg(format!("--canonical={}", args.canonical))
                .arg("--skip-git-repo-check");

            let res = cmd.output();
            match res {
                Ok(out) if out.status.success() => {
                    output = Some(String::from_utf8_lossy(&out.stdout).trim().to_string());
                    last_err = None;
                    break;
                }
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    last_err = Some(if err.is_empty() { format!("exit_code={}", out.status) } else { err });
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                }
            }

            if attempt < args.retries {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }

        let out_line = if let Some(raw) = output {
            if raw.is_empty() {
                None
            } else {
                match serde_json::from_str::<Value>(&raw) {
                    Ok(mut v) => {
                        if let Value::Object(map) = &mut v {
                            if !map.contains_key("doc_id") || map.get("doc_id") == Some(&Value::Null) {
                                let fallback = doc_id.clone().unwrap_or_else(|| format!("row_{}", row_idx));
                                map.insert("doc_id".to_string(), Value::String(fallback));
                            }
                        }
                        Some(serde_json::to_string(&v).unwrap_or(raw))
                    }
                    Err(_) => Some(raw),
                }
            }
        } else {
            None
        };

        if let Some(line) = out_line {
            writeln!(out_file, "{}", line)?;
            count += 1;
        } else {
            let err = last_err.unwrap_or_else(|| "unknown_error".to_string());
            let fail = serde_json::json!({
                "row_index": row_idx,
                "doc_id": doc_id.clone(),
                "sig": sig,
                "error": err,
            });
            writeln!(fail_file, "{}", fail)?;
        }

        if args.limit > 0 && count >= args.limit {
            break;
        }
    }

    println!("Wrote {} ({} rows)", args.out_path.display(), count);
    if failures_path.exists() && failures_path.metadata()?.len() > 0 {
        println!("Failures: {}", failures_path.display());
    }

    Ok(())
}
