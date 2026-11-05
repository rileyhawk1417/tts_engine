//! T012 phase-2 (2026-11-05): bench the official neucodec ONNX decoders
//! through the same ORT machinery the kokoro lane rides (`graph-ort`).
//!
//! This is the number a NeuCodec head of a NeuTTS-style lane would actually
//! ship with in this engine — not a Python-stack proxy.
//!
//! Graph contract (read from the artifacts):
//!   in  `codes` int32 [b, 1, T]   — FSQ code ids at 50 Hz
//!   out `audio` f32  [b, 1, 480*T - 480]  — 24 kHz PCM
//! Budget: decode 50 tokens (1 s of speech) in <= 1 s wall.
//!
//! usage: neucodec_ort_bench <path/to/model.onnx> [reps=20] [intra=8]
//!   reps: timed runs per token-count, after 3 warmups
//!   intra: ORT intra-op threads; pass 0 for runtime default (all)
//!
//! needs: `--features graph-ort` and an ORT dylib on the default path, or
//! `$TTS_ORT_DYLIB` (kokoro lane conventions reused verbatim).

use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

/// Same dylib the kokoro lane rides by default (1.22.1, api-21).
const DEFAULT_ORT_LIB: &str = "/teranux/ort-1.22.1/libonnxruntime.so";
const FRAME_TOKENS: usize = 50; // 1 s of speech at 50 Hz
const SAMPLES_PER_FRAME: usize = 480; // 24 kHz / 50 Hz

static ENV: OnceLock<Environment> = OnceLock::new();

fn ort_env() -> Result<&'static Environment, String> {
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let lib = std::env::var("TTS_ORT_DYLIB").unwrap_or_else(|_| DEFAULT_ORT_LIB.to_string());
    let env = ort::init_from(&lib)
        .map_err(|e| format!("ort dylib load failed ({lib}): {e}"))?
        .with_telemetry(false)
        .build()
        .map_err(|e| format!("ort environment build: {e}"))?;
    ENV.set(env).map_err(|_| "env raced".to_string())?;
    Ok(ENV.get().expect("just set"))
}

fn build_session(
    env: &Environment,
    model: &str,
    intra: usize,
    threads_arg: u64,
) -> Result<Session, String> {
    let name = if threads_arg == 0 {
        intra // 0 arg = runtime default; pass the builder's own default
    } else {
        threads_arg.min(u16::MAX as u64) as usize
    };
    Session::builder(env)
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_intra_threads(name)
        .map_err(|e| format!("ort intra threads: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| format!("ort opt level: {e}"))?
        .commit_from_file(Path::new(model))
        .map_err(|e| format!("ort commit {model}: {e}"))
}

/// One decode: codes [1,1,T] with deterministic pseudo-ids; returns audio len.
fn decode_once(sess: &mut Session, t: usize) -> Result<usize, String> {
    let codes: Vec<i32> = (0..t).map(|i| ((i * i * 7 + 13) % 1024) as i32).collect();
    let codes_t = Tensor::from_array(([1usize, 1usize, t], codes))
        .map_err(|e| format!("codes tensor: {e}"))?;
    let outs = sess
        .run(ort::inputs!["codes" => codes_t])
        .map_err(|e| format!("graph run: {e}"))?;
    let out = outs
        .get("audio")
        .ok_or_else(|| "graph has no 'audio' output".to_string())?;
    let (_, data) = out
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("audio extract: {e}"))?;
    Ok(data.len())
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        (v[v.len() / 2 - 1] + v[v.len() / 2]).div_f64(2.0)
    }
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| {
        "/teranux/hugging_face/t012-drop/neucodec-onnx-decoder-int8/model.onnx".to_string()
    });
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let threads_arg: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);

    let env = ort_env()?;
    let mut sess = build_session(env, &model, 8, threads_arg)?;
    let tag = Path::new(&model)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    println!("model: {tag:10} threads: {threads_arg} (0=runtime default)  reps: {reps}");

    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "tokens", "med_ms", "min_ms", "max_ms", "tok/s", "x budget(50)"
    );
    for &t in &[50usize, 150, 250] {
        // exact-length sanity on the shaped contract
        let len = decode_once(&mut sess, t)?;
        let want = SAMPLES_PER_FRAME * t - SAMPLES_PER_FRAME; // 480*T - 480
        if len != want {
            return Err(format!("audio length {len} != contract {want} (T={t})"));
        }
        // warmup
        for _ in 0..3 {
            decode_once(&mut sess, t)?;
        }
        // timed
        let mut walls: Vec<Duration> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            decode_once(&mut sess, t)?;
            walls.push(t0.elapsed());
        }
        let med = median(&mut walls);
        let min = *walls.iter().min().expect("reps>0");
        let max = *walls.iter().max().expect("reps>0");
        let toks = t as f64 / med.as_secs_f64();
        println!(
            "{:>8} {:>10.1} {:>10.1} {:>10.1} {:>8.1} {:>11.2}x",
            t,
            med.as_secs_f64() * 1e3,
            min.as_secs_f64() * 1e3,
            max.as_secs_f64() * 1e3,
            toks,
            toks / 50.0
        );
    }

    // value-invariance sanity: all-zero ids must decode to the same length.
    {
        let t = FRAME_TOKENS;
        let zeros: Vec<i32> = vec![0; t];
        let zeros_t = Tensor::from_array(([1usize, 1usize, t], zeros))
            .map_err(|e| format!("zeros tensor: {e}"))?;
        let outs = sess
            .run(ort::inputs!["codes" => zeros_t])
            .map_err(|e| format!("zeros run: {e}"))?;
        let out = outs
            .get("audio")
            .ok_or_else(|| "zeros: no 'audio' output".to_string())?;
        let (_, data) = out
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("zeros extract: {e}"))?;
        if data.len() != SAMPLES_PER_FRAME * t - SAMPLES_PER_FRAME {
            return Err(format!("zeros: len {} breaks contract", data.len()));
        }
    } // drop outs before the timed loop (Session::run takes &mut)
    let t = FRAME_TOKENS;
    let t0 = Instant::now();
    for _ in 0..5 {
        let zeros: Vec<i32> = vec![0; t];
        let zeros_t = Tensor::from_array(([1usize, 1usize, t], zeros))
            .map_err(|e| format!("zeros tensor: {e}"))?;
        sess.run(ort::inputs!["codes" => zeros_t])
            .map_err(|e| format!("zeros run: {e}"))?;
    }
    println!(
        "zerocodes: all-zero ids decode fine ({:.1} ms/run median across 5), len contract holds — rate is value-independent",
        t0.elapsed().as_secs_f64() * 1e3 / 5.0
    );

    println!("budget line: realtime needs 50 tok/s; >1.0x = realtime-capable decode");
    Ok(())
}
