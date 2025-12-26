use anyhow::{Context, Result};
use clap::Parser;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Shard a SIG JSONL file (round-robin)", version)]
struct Args {
    #[arg(long = "in", value_name = "FILE")]
    in_path: PathBuf,

    #[arg(long = "out-dir", value_name = "DIR")]
    out_dir: PathBuf,

    #[arg(long = "shards", default_value_t = 4)]
    shards: usize,

    #[arg(long = "prefix", default_value = "shard")]
    prefix: String,

    #[arg(long = "limit", default_value_t = 0)]
    limit: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.shards == 0 {
        anyhow::bail!("--shards must be >= 1");
    }

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("Failed to create {}", args.out_dir.display()))?;

    let mut outs: Vec<File> = Vec::with_capacity(args.shards);
    for i in 0..args.shards {
        let path = args
            .out_dir
            .join(format!("{}-{:02}.jsonl", args.prefix, i + 1));
        let f = File::create(&path)
            .with_context(|| format!("Failed to create {}", path.display()))?;
        outs.push(f);
    }

    let in_file = File::open(&args.in_path)
        .with_context(|| format!("Failed to open {}", args.in_path.display()))?;
    let reader = BufReader::new(in_file);

    let mut count = 0usize;
    let mut idx = 0usize;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let shard = idx % args.shards;
        writeln!(outs[shard], "{}", line)?;
        idx += 1;
        count += 1;
        if args.limit > 0 && count >= args.limit {
            break;
        }
    }

    println!(
        "Wrote {} records into {} shards at {}",
        count,
        args.shards,
        args.out_dir.display()
    );

    Ok(())
}
