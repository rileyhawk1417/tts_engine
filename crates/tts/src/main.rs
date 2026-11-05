//! `tts synth` — the seam CLI (docs/seam.md §2):
//!
//!     tts synth --lane LANE --model MODEL [--lang en] [--voice NAME]
//!               --text TEXT -o out.wav [--jsonl out.jsonl | --bench]
//!               [--seed N] [--allow-hf]
//!
//! Exit codes: 0 ok · 2 usage · 3 load · 4 synth · 5 io. Telemetry is the
//! seam's job: `RunRecorder` drives `chunks.jsonl` while chunks flow through.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tts_engine::lanes::make_lane;
use tts_engine::telemetry::RunRecorder;
use tts_engine::types::{LaneError, Lang, ModelRef, SynthRequest, VoiceRef};
use tts_engine::{chunking, wav};

static PROC_START: OnceLock<Instant> = OnceLock::new();

fn proc_start() -> Instant {
    *PROC_START.get_or_init(Instant::now)
}

#[derive(Parser)]
#[command(name = "tts", version, about = "Model-agnostic TTS engine (seam CLI)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Synthesize text through a lane, writing WAV + telemetry.
    Synth {
        /// Lane name: smoke | kokoro (feature graph-ort) | flow-ptts | flow-candle
        /// | ggml-ffi | streaming-graph (later ones land per task)
        #[arg(long)]
        lane: String,
        /// Model: a local path (mmap-loaded per the seam mandate) or, with
        /// --allow-hf, a hub id like hexgrad/Kokoro-82M
        #[arg(long)]
        model: String,
        /// Language tag (base token decides: en, en_US, fr-FR, ...)
        #[arg(long, default_value = "en")]
        lang: String,
        /// Voice: named voice id or a path to an embedding file
        #[arg(long)]
        voice: Option<String>,
        /// Input text, or `-` for stdin, or `@path/to/file.txt`
        #[arg(long)]
        text: String,
        /// Output WAV path (mono 16-bit)
        #[arg(short = 'o', long)]
        out: PathBuf,
        /// Telemetry sink (one row: header, N chunks, summary)
        #[arg(long, conflicts_with = "bench")]
        jsonl: Option<PathBuf>,
        /// Stream telemetry rows to stdout (one row per line)
        #[arg(long)]
        bench: bool,
        /// Seed for seeded/deterministic lanes (also TTS_ENGINE_SEED)
        #[arg(long)]
        seed: Option<u64>,
        /// Allow explicit hub ids (first use downloads into the HF cache)
        #[arg(long)]
        allow_hf: bool,
        /// Synthesize N times in-process (warm-pool measurement; rep 0 = cold,
        /// the only rep that pays load). Outputs rep-indexed when N > 1.
        #[arg(long)]
        repeat: Option<u32>,
    },
}

fn main() -> ExitCode {
    let _ = proc_start(); // anchor t_synth_start_ms at the earliest point we can
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Synth {
            lane,
            model,
            lang,
            voice,
            text,
            out,
            jsonl,
            bench,
            seed,
            allow_hf,
            repeat,
        } => run_synth(
            lane,
            model,
            lang,
            voice,
            text,
            out,
            jsonl,
            bench,
            seed,
            allow_hf,
            repeat.unwrap_or(1),
        ),
    }
}

fn usage_fail(msg: &str) -> ExitCode {
    eprintln!("usage: {msg}");
    ExitCode::from(2)
}

fn load_fail(msg: &str) -> ExitCode {
    eprintln!("load error: {msg}");
    ExitCode::from(3)
}

fn synth_fail(msg: &str) -> ExitCode {
    eprintln!("synth error: {msg}");
    ExitCode::from(4)
}

fn io_fail(msg: &str) -> ExitCode {
    eprintln!("io error: {msg}");
    ExitCode::from(5)
}

fn lane_exit(e: &LaneError) -> ExitCode {
    match e {
        LaneError::Load(_) => load_fail(&e.to_string()),
        LaneError::Synth(_) => synth_fail(&e.to_string()),
        LaneError::Io(_) => io_fail(&e.to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_synth(
    lane_name: String,
    model_arg: String,
    lang_arg: String,
    voice_arg: Option<String>,
    text_arg: String,
    out: PathBuf,
    jsonl: Option<PathBuf>,
    bench: bool,
    seed_arg: Option<u64>,
    allow_hf: bool,
    repeat: u32,
) -> ExitCode {
    // --- resolve text -------------------------------------------------------
    let text = match resolve_text(&text_arg) {
        Ok(t) => t,
        Err(e) => return io_fail(&format!("text input: {e}")),
    };
    if text.trim().is_empty() {
        return usage_fail("empty text; pass --text \"...\", - for stdin, or @file.txt");
    }

    // --- resolve lang -------------------------------------------------------
    let Some(lang) = Lang::parse(&lang_arg) else {
        return usage_fail(format!("unsupported lang {lang_arg:?} (en fr de es pt)").as_str());
    };

    // --- resolve model ------------------------------------------------------
    let model_ref = resolve_model(&lane_name, &model_arg, allow_hf);
    let Some(model_ref) = model_ref else {
        return load_fail(
            format!("model not found on disk: {model_arg} (hub ids need --allow-hf)").as_str(),
        );
    };

    // --- resolve lane -------------------------------------------------------
    let Some(mut lane) = make_lane(&lane_name) else {
        return usage_fail(
            format!("unknown lane {lane_name:?} (smoke is registered; others arrive per task)")
                .as_str(),
        );
    };

    // --- env dials (advisory logging only; lanes read what they need) -------
    if let Ok(t) = std::env::var("TTS_ENGINE_THREADS") {
        eprintln!("dials: TTS_ENGINE_THREADS={t}");
    }
    let seed = seed_arg.or_else(|| {
        std::env::var("TTS_ENGINE_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
    });

    // --- load ---------------------------------------------------------------
    let load_info = match lane.load(&model_ref) {
        Ok(info) => info,
        Err(e) => return lane_exit(&e),
    };

    // --- chunk (seam-side per T013 law) -------------------------------------
    let chunks = chunking::chunk_text(&text, chunking::ChunkConfig::default());

    // --- synthesize ---------------------------------------------------------
    let req = SynthRequest {
        text: text.clone(),
        chunks,
        lang,
        voice: voice_arg.map(|v| resolve_voice(&v)),
        seed,
        lane_options: tts_engine::types::LaneOptions::default(),
    };

    for rep in 0..repeat {
        let synth_start = Instant::now();
        let stream = match lane.synthesize(&req) {
            Ok(s) => s,
            Err(e) => return lane_exit(&e),
        };
        debug_assert_eq!(
            stream.meta.load_ms, load_info.load_ms,
            "lane meta must restate the load wall"
        );

        // --- telemetry sink -----------------------------------------------------
        if bench && jsonl.is_some() {
            eprintln!("note: --bench wins over --jsonl");
        }
        let suffix = if repeat > 1 {
            format!(".rep{rep}")
        } else {
            String::new()
        };
        let out_rep = suffixed_path(&out, &suffix);
        let jsonl_rep = jsonl.as_ref().map(|p| suffixed_path(p, &suffix));

        let sink: Box<dyn Write> = if bench {
            Box::new(std::io::stdout().lock())
        } else if let Some(p) = &jsonl_rep {
            match open_sink_file(p) {
                Ok(f) => Box::new(f),
                Err(e) => return io_fail(&format!("jsonl {:?}: {e}", p)),
            }
        } else {
            Box::new(std::io::sink())
        };

        let run_id = if repeat > 1 {
            format!("{}-{}-r{}", lane_name, epoch_secs(), rep)
        } else {
            format!("{}-{}", lane_name, epoch_secs())
        };
        let mut recorder = match RunRecorder::begin(
            &stream.meta,
            proc_start(),
            synth_start,
            sink,
            run_id,
            "not-assessed",
        ) {
            Ok(r) => r,
            Err(e) => return io_fail(&e.to_string()),
        };

        let mut pcm_all: Vec<i16> = Vec::new();
        for item in stream.chunks {
            match item {
                Ok(chunk) => {
                    if let Err(e) = recorder.on_chunk(&chunk) {
                        return io_fail(&e.to_string());
                    }
                    pcm_all.extend_from_slice(&chunk.pcm);
                }
                Err(e) => return lane_exit(&e),
            }
        }
        let stats = match recorder.finish() {
            Ok(s) => s,
            Err(e) => return io_fail(&e.to_string()),
        };

        // --- write wav ----------------------------------------------------------
        if let Err(e) = wav::write_wav(&out_rep, stream.meta.sample_rate, &pcm_all) {
            return io_fail(&format!("write {:?}: {e}", out_rep));
        }

        // --- human summary on stderr (rows went to jsonl/stdout) ---------------
        let rep_tag = if repeat > 1 {
            format!(" rep={rep}")
        } else {
            String::new()
        };
        eprintln!(
            "synth ok: lane={} model={} voice={} chunks={} audio={:.3}s wall={}ms rtf={:.3} ttfa={}ms peak_rss={}KB{} -> {}",
            lane_name,
            stream.meta.model,
            stream.meta.voice,
            stats.chunks,
            stats.audio_seconds,
            stats.wall_ms,
            stats.rtf_final.unwrap_or(f64::NAN),
            stats.ttfa_ms.unwrap_or(0),
            stats.peak_rss_kb,
            rep_tag,
            out_rep.display(),
        );
    }
    ExitCode::SUCCESS
}

fn suffixed_path(p: &Path, suffix: &str) -> PathBuf {
    if suffix.is_empty() {
        return p.to_path_buf();
    }
    match p.extension() {
        Some(ext) => p.with_extension(format!("{}{suffix}", ext.to_string_lossy())),
        None => p.with_extension(suffix.trim_start_matches('.')),
    }
}

fn resolve_text(arg: &str) -> std::io::Result<String> {
    if arg == "-" {
        if std::io::stdin().is_terminal() {
            eprintln!("reading text from stdin (^D to end)");
        }
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else if let Some(p) = arg.strip_prefix('@') {
        std::fs::read_to_string(p)
    } else {
        Ok(arg.to_string())
    }
}

fn resolve_model(lane: &str, arg: &str, allow_hf: bool) -> Option<ModelRef> {
    // Diagnostic pseudo-models (no weights): `tts synth --lane smoke --model smoke`
    if lane == "smoke" && arg == "smoke" {
        return Some(ModelRef::Pseudo(arg.to_string()));
    }
    let p = Path::new(arg);
    if p.exists() {
        return Some(ModelRef::Path(p.to_path_buf()));
    }
    if allow_hf && arg.contains('/') && !arg.is_empty() {
        return Some(ModelRef::Hub {
            id: arg.to_string(),
            revision: None,
        });
    }
    None
}

fn resolve_voice(arg: &str) -> VoiceRef {
    let p = Path::new(arg);
    if p.exists() {
        VoiceRef::Path(p.to_path_buf())
    } else {
        VoiceRef::Id(arg.to_string())
    }
}

fn open_sink_file(p: &Path) -> std::io::Result<std::fs::File> {
    if let Some(parent) = p.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::File::create(p)
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
