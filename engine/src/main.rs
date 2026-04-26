use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Parser, Debug)]
#[command(name = "engine", about = "video-ai engine: orchestrate analysis pipeline")]
struct Args {
    video: PathBuf,

    #[arg(long, default_value = "storage")]
    storage: PathBuf,

    #[arg(long, default_value = "python")]
    python: String,

    #[arg(long)]
    skip_normalize: bool,

    #[arg(long)]
    no_faces: bool,

    #[arg(long)]
    no_objects: bool,

    #[arg(long)]
    embeddings: bool,

    #[arg(long)]
    asr: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct Highlight {
    t0: f64,
    t1: f64,
    score: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Features {
    video_id: String,
    duration: f64,
    #[serde(default)]
    highlights: Vec<Highlight>,
    #[serde(default)]
    global_decisions: Vec<String>,
}

fn video_id(path: &Path) -> Result<String> {
    let canon = path.canonicalize().context("canonicalize input")?;
    let mtime = std::fs::metadata(&canon)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow!("bad mtime: {e}"))?
        .as_secs();
    let raw = format!("{}|{}", canon.display(), mtime);
    let mut h = Sha256::new();
    h.update(raw.as_bytes());
    Ok(hex(&h.finalize())[..16].to_string())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn ensure_ffmpeg() -> Result<()> {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .context("ffmpeg not found in PATH")?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure_ffmpeg()?;

    let vid = video_id(&args.video)?;
    let cache_dir = args.storage.join("cache").join(&vid);
    std::fs::create_dir_all(&cache_dir)?;

    eprintln!("[engine] video_id={vid} cache={}", cache_dir.display());

    let mut cmd = Command::new(&args.python);
    cmd.args(["-m", "analysis"])
        .arg(&args.video)
        .arg("--storage")
        .arg(&args.storage)
        .arg("--out")
        .arg(cache_dir.join("features.json"));
    if args.skip_normalize {
        cmd.arg("--no-normalize");
    }
    if args.no_faces {
        cmd.arg("--no-faces");
    }
    if args.no_objects {
        cmd.arg("--no-objects");
    }
    if args.embeddings {
        cmd.arg("--embeddings");
    }
    if args.asr {
        cmd.arg("--asr");
    }

    let status = cmd.status().context("spawn python analysis")?;
    if !status.success() {
        return Err(anyhow!("analysis failed with status {status}"));
    }

    let json_path = cache_dir.join("features.json");
    let raw = std::fs::read_to_string(&json_path)
        .with_context(|| format!("read {}", json_path.display()))?;
    let feats: Features = serde_json::from_str(&raw)?;
    eprintln!(
        "[engine] duration={:.2}s highlights={} global_decisions={:?}",
        feats.duration,
        feats.highlights.len(),
        feats.global_decisions,
    );
    println!("{}", json_path.display());
    Ok(())
}
