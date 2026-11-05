//! T007 — Session-primer measurement: cold `synth.stream_with` (voice
//! conditioning paid per call) vs `synth.session_default` (conditioned once,
//! cloned per call). Prints per-rep TTFA + total wall so the lane decides
//! whether to wire `Session` in.
//!
//! Run: cargo run --release -p tts-engine --features flow-ptts \
//!      --example ptts_session_probe -- models/phonon-31m-en

use std::time::Instant;

use ptts::checkpoint::Checkpoint;
use ptts::preprocess::{Lang as PttsLang, Normalize};
use ptts::synth::{SpeechOptions, Synth};

const TEXT: &str = "The engine loads its weights directly from disk and streams audio frames.";
const SEED: u64 = 42;
const REPS: usize = 4;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("pass the checkpoint dir, e.g. models/phonon-31m-en");
    let text = std::env::args().nth(2).unwrap_or_else(|| TEXT.to_string());
    let dial = std::env::args()
        .nth(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    xn::set_num_threads(dial); // gate dial default: honest timing, mode-comparable
    println!("# dial xn_threads={dial} text=\"{text}\"");

    let t0 = Instant::now();
    // Q80: apples-to-apples with the lane (LANE_QUANT). Default is F32 —
    // 2.4x faster per frame on Zen 2 (T009 q-law datapoint) and different audio.
    let cp = Checkpoint::resolve(
        std::path::Path::new(&dir),
        ptts::checkpoint::ResolveOptions {
            quant: ptts::synth::Quant::Q80,
            weights: None,
        },
    )
    .expect("checkpoint resolve");
    let mut synth: Synth = cp
        .builder(Normalize::from(PttsLang::En))
        .build()
        .expect("synth build");
    cp.register_voices(&mut synth);
    let load_wall = t0.elapsed();
    let voice = synth
        .voices()
        .first()
        .cloned()
        .expect("checkpoint must ship a voice");

    // Warmup (discarded): page-in weights + allocator.
    let wt = Instant::now();
    let warm = synth
        .stream_with(
            &text,
            &SpeechOptions {
                voice: Some(voice.clone()),
                seed: Some(SEED),
                ..Default::default()
            },
        )
        .expect("warmup");
    let warm_ttfa = wt.elapsed().as_millis();
    let wsz = warm
        .into_iter()
        .map(|i| i.map(|p| p.len()).unwrap_or(0))
        .sum::<usize>();
    println!("# warmup_ms {warm_ttfa} warmup_samples {wsz}");

    // ---- cold: conditioning inside every call (lane's current path) ----
    // RAW text: stream_with normalizes inside plan_chunks; pre-normalizing
    // here double-applies and alters generation (measured T007).
    let normalized = text.clone();
    let cold: Vec<(u128, u128)> = (0..REPS)
        .map(|_| {
            let t0 = Instant::now();
            let s = synth
                .stream_with(
                    &normalized,
                    &SpeechOptions {
                        voice: Some(voice.clone()),
                        seed: Some(SEED),
                        ..Default::default()
                    },
                )
                .expect("cold stream");
            // per-item counts reveal chunk granularity (frame vs buffer)
            let mut n_items = 0usize;
            let ttfa: u128;
            let mut it = s;
            if let Some(item) = it.next() {
                n_items = 1;
                ttfa = t0.elapsed().as_millis();
                drop(item);
            } else {
                ttfa = t0.elapsed().as_millis();
            }
            let start = Instant::now();
            let mut rest = 0usize;
            for item in it {
                n_items += 1;
                rest += item.expect("cold pcm").len();
            }
            let post = start.elapsed().as_millis();
            let total_samples = rest;
            println!(
                "cold      ttfa_ms {ttfa} post_first_ms {post} samples {total_samples} n_items {n_items}"
            );
            (ttfa, post)
        })
        .collect();

    // ---- primed: conditioning once at session build ----
    let opts = SpeechOptions {
        voice: Some(voice.clone()),
        seed: Some(SEED),
        ..Default::default()
    };
    let sp = Instant::now();
    let session = synth.session(&opts, 4096).expect("session 4096");
    let prime_ms = sp.elapsed().as_millis();
    println!(
        "load+build_ms {} | session_prime_ms {} | seq_budget {}",
        load_wall.as_millis(),
        prime_ms,
        session.seq_budget()
    );
    let primed: Vec<(u128, u128)> = (0..REPS)
        .map(|_| {
            let t0 = Instant::now();
            let mut it = session.stream_seeded(&text, SEED).expect("primed stream");
            let mut first_len = 0usize;
            let ttfa: u128;
            if let Some(item) = it.next() {
                first_len = item.expect("primed pcm").len();
                ttfa = t0.elapsed().as_millis();
            } else {
                ttfa = t0.elapsed().as_millis();
            }
            let start = Instant::now();
            let mut rest = 0usize;
            for item in it {
                rest += item.expect("primed pcm").len();
            }
            let post = start.elapsed().as_millis();
            println!(
                "primed    ttfa_ms {ttfa} post_first_ms {post} samples {} first_item_samples {first_len}",
                first_len + rest
            );
            (ttfa, post)
        })
        .collect();

    let avg = |xs: &[(u128, u128)]| xs.iter().map(|x| x.0).sum::<u128>() / REPS as u128;
    let (c, p) = (avg(&cold), avg(&primed));
    println!("AVG ttfa_ms cold {c} primed {p}");
    println!(
        "TTFA delta cold->primed: -{} ms ({:.2}x faster)",
        c - p,
        c as f64 / p as f64
    );
}
