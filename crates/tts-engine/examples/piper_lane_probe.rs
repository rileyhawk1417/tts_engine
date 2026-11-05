//! T011 session-2 probe for the piper lane: in-process load + synthesize
//! over a patched (caller-supplied seeded noise) fused VITS graph, with a
//! 3× byte-identity check (the session-1 python-ORT acceptance, restated
//! through the Rust seam) and a stage-time breakdown.
//!
//! Run (amy or ryan — pass the *-rng.onnx graph):
//!   PIPER_ESPEAKNG_DATA_DIRECTORY=/usr/share \
//!     cargo run --release -p tts-engine --features piper \
//!     --example piper_lane_probe -- /teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx

use std::time::Instant;

use tts_engine::lanes::piper::PiperLane;
use tts_engine::telemetry::pcm_sha;
use tts_engine::types::{Lang, ModelRef, SynthRequest, TtsLane, VoiceRef};

fn main() {
    let onnx = std::env::args()
        .nth(1)
        .expect("pass the patched piper graph (*-rng.onnx)");
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The engine loads its weights straight from disk.".to_string());

    let t0 = Instant::now();
    let mut lane = PiperLane::new();
    lane.load(&ModelRef::Path(onnx.into())).expect("lane load");
    println!("# load_wall_ms {}", t0.elapsed().as_millis());

    // Warmup (discarded) — lazy voice-config parse lands in the first TTFA.
    let warm = lane
        .synthesize(&synreq("Warmup sentence.", Some(42)))
        .expect("warm synth");
    let warm_meta = warm.meta.clone();
    let _ = warm.chunks.count();
    println!(
        "# meta lane={} sr={} voice={} determinism={:?} chunk_policy={}",
        warm_meta.lane,
        warm_meta.sample_rate,
        warm_meta.voice,
        warm_meta.determinism,
        warm_meta.chunk_policy
    );

    let mut shas: Vec<String> = Vec::new();
    for rep in 0..3 {
        let ta = Instant::now();
        let stream = lane.synthesize(&synreq(&text, Some(42))).expect("synth");
        let meta = stream.meta.clone();
        let setup = ta.elapsed();
        let mut total_samples = 0usize;
        let mut ttfa_ms = 0u128;
        let mut first_done = false;
        let mut pcm_all: Vec<i16> = Vec::new();
        let tb = Instant::now();
        for chunk in stream.chunks {
            let c = chunk.expect("chunk");
            total_samples += c.pcm.len();
            pcm_all.extend_from_slice(&c.pcm);
            if !first_done {
                ttfa_ms = tb.elapsed().as_millis();
                first_done = true;
            }
        }
        let drain_ms = tb.elapsed().as_millis();
        let sha = pcm_sha(&pcm_all);
        shas.push(sha.clone());
        let rep_setup_ms = setup.as_millis();
        let audio_s = total_samples as f64 / f64::from(meta.sample_rate);
        let rtf = audio_s / (drain_ms as f64 / 1000.0);
        println!(
            "# rep{rep} setup_ms {rep_setup_ms} ttfa_ms {ttfa_ms} drain_ms {drain_ms} \
             samples {total_samples} audio_s {audio_s:.3} rtf(audio/wall) {rtf:.3} \
             pcm_sha {sha}",
        );
        if rep == 0 {
            // Stage split of the first chunk (g2p / acoustic / vocoder / io).
            let stream2_setup = Instant::now();
            let stream2 = lane
                .synthesize(&synreq(&text, Some(42)))
                .expect("stage synth");
            let mut stage_printed = false;
            for chunk in stream2.chunks {
                let c = chunk.expect("stage chunk");
                if !stage_printed {
                    let s = c.stages;
                    println!(
                        "# stages g2p {}ms acoustic {}ms vocoder {}ms io {}ms (setup {}ms)",
                        s.g2p.as_millis(),
                        s.acoustic.as_millis(),
                        s.vocoder.as_millis(),
                        s.io.as_millis(),
                        stream2_setup.elapsed().as_millis(),
                    );
                    stage_printed = true;
                }
            }
        }
    }
    if shas.iter().all(|s| *s == shas[0]) {
        println!(
            "# PASS 3× byte-identical at seed 42 ({})",
            &shas[0][..16.min(shas[0].len())]
        );
    } else {
        println!("# FAIL byte-identity: {shas:?}");
        std::process::exit(1);
    }
}

fn synreq(text: &str, seed: Option<u64>) -> SynthRequest {
    SynthRequest {
        text: text.into(),
        chunks: Vec::new(),
        lang: Lang::En,
        voice: Some(VoiceRef::Default),
        seed,
        lane_options: Default::default(),
    }
}
