//! T010 layer-B probe for the ggml-ffi (Chatterbox Turbo) lane: in-process
//! load + synthesize through the llama.cpp FFI shim, with a 3× byte-identity
//! check, a stage-time breakdown, and an optional voice-cloning pass.
//!
//! Precondition: the pinned llama.cpp static build
//! (`/teranux/third_party/llama.cpp`, commit 186eab1) carries
//! `build/**.a`; see `docs/notes/chatterbox.md` for the build recipe.
//!
//! Run (default shipped voice):
//!   cargo run --release -p tts-engine --features ggml-ffi --example ggml_ffi_lane_probe -- \
//!     [[/teranux/hugging_face/t012-drop/chatterbox/chatterbox-turbo-Q8_0.gguf]]
//! Voice cloning:
//!   GGML_FFI_PROBE_SPEAKER=/tmp/t011-long.wav cargo run ... (same args)

use std::time::Instant;

use tts_engine::lanes::ggml_ffi::GgmlFfiLane;
use tts_engine::telemetry::pcm_sha;
use tts_engine::types::{Lang, ModelRef, SynthRequest, TtsLane, VoiceRef};

const DEFAULT_MODEL: &str = "/teranux/hugging_face/t012-drop/chatterbox/chatterbox-turbo-Q8_0.gguf";

fn synreq(text: &str, seed: Option<u64>, speaker: Option<&str>) -> SynthRequest {
    SynthRequest {
        text: text.into(),
        chunks: Vec::new(),
        lang: Lang::En,
        voice: match speaker {
            Some(p) => Some(VoiceRef::Path(p.into())),
            None => Some(VoiceRef::Default),
        },
        seed,
        lane_options: Default::default(),
    }
}

fn main() {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let text = std::env::args().nth(2).unwrap_or_else(|| {
        "Hello from the chatterbox native port, running on a laptop CPU.".to_string()
    });
    let speaker = std::env::var("GGML_FFI_PROBE_SPEAKER").ok();

    let t0 = Instant::now();
    let mut lane = GgmlFfiLane::new();
    println!("# lane {}", lane.lane_name());
    lane.load(&ModelRef::Path(model.into()))
        .expect("lane load (mmproj twin must sit beside the backbone gguf)");
    let load_wall_ms = t0.elapsed().as_millis();

    // Warmup (discarded) — llama.cpp compute-buffer warmup lands in TTFA.
    let warm = lane
        .synthesize(&synreq("Warmup sentence.", Some(42), None))
        .expect("warm synth");
    let warm_meta = warm.meta.clone();
    let _ = warm.chunks.count();
    println!(
        "# load_wall_ms {load_wall_ms} (load_ms {}) sr={} voice={} determinism={:?} chunk_policy={} runtime={}",
        warm_meta.load_ms,
        warm_meta.sample_rate,
        warm_meta.voice,
        warm_meta.determinism,
        warm_meta.chunk_policy,
        warm_meta.runtime,
    );

    // 3× byte-identity at the default dial (threads from TTS_CBX_THREADS or
    // the lane's t4 default). Shipped voice; cloning is gated separately
    // below so the two identity classes never share a sha baseline.
    let mut shas: Vec<String> = Vec::new();
    let mut first_meta = None::<tts_engine::types::RunMeta>;
    for rep in 0..3 {
        let ta = Instant::now();
        let stream = lane
            .synthesize(&synreq(&text, Some(42), None))
            .expect("synth");
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
                let s = c.stages;
                println!(
                    "# stages g2p {}ms acoustic {}ms vocoder {}ms io {}ms (ttfa {ttfa_ms}ms, setup {}ms)",
                    s.g2p.as_millis(),
                    s.acoustic.as_millis(),
                    s.vocoder.as_millis(),
                    s.io.as_millis(),
                    setup.as_millis(),
                );
            }
        }
        let drain_ms = tb.elapsed().as_millis();
        let sha = pcm_sha(&pcm_all);
        shas.push(sha.clone());
        let audio_s = total_samples as f64 / f64::from(meta.sample_rate);
        let rtf = audio_s / (drain_ms as f64 / 1000.0);
        println!(
            "# rep{rep} setup_ms {} ttfa_ms {ttfa_ms} drain_ms {drain_ms} samples {total_samples} \
             audio_s {audio_s:.3} rtf(audio/wall) {rtf:.3} pcm_sha {sha}",
            setup.as_millis()
        );
        if let Some(m) = first_meta.take() {
            assert_eq!(m.sample_rate, meta.sample_rate, "Rate must be stable");
            assert_eq!(m.voice, meta.voice, "Voice label must be stable");
        } else {
            first_meta = Some(meta);
        }

        if rep == 0 {
            // Persist the first byte-identity output for the round-trip WER
            // gate (te transcribe).
            let out = "/tmp/ggml_ffi_probe_rep0.wav";
            tts_engine::wav::write_wav(
                out,
                first_meta.as_ref().map(|m| m.sample_rate).unwrap_or(0),
                &pcm_all,
            )
            .expect("write probe wav");
            println!("# wav {out} (rep0 bytes, for te round-trip WER)");
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

    // Optional: cloned-voice determinism pair (the layer-A law, restated
    // through the seam).
    if let Some(sp) = &speaker {
        let mut shas_c: Vec<String> = Vec::new();
        for _ in 0..2 {
            let stream = lane
                .synthesize(&synreq(&text, Some(42), Some(sp)))
                .expect("cloned synth");
            let mut pcm_all: Vec<i16> = Vec::new();
            for chunk in stream.chunks {
                pcm_all.extend_from_slice(&chunk.expect("chunk").pcm);
            }
            shas_c.push(pcm_sha(&pcm_all));
        }
        if shas_c[0] == shas_c[1] {
            println!(
                "# PASS 2× byte-identical cloned voice ({})",
                &shas_c[0][..16.min(shas_c[0].len())]
            );
        } else {
            println!("# FAIL cloned-voice byte-identity");
            std::process::exit(1);
        }

        // Sticky-voice regression (the reset law): Default AFTER Path must
        // reproduce the shipped-voice audio byte-for-byte.
        let stream = lane
            .synthesize(&synreq(&text, Some(42), None))
            .expect("post-clone default synth");
        let mut pcm_all: Vec<i16> = Vec::new();
        for chunk in stream.chunks {
            pcm_all.extend_from_slice(&chunk.expect("chunk").pcm);
        }
        if pcm_sha(&pcm_all) == shas[0] {
            println!("# PASS shipped voice restored after cloning (no sticky speaker)");
        } else {
            println!("# FAIL sticky voice: Default after Path diverged");
            std::process::exit(1);
        }
    }
}
