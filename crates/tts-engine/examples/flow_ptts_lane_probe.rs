//! T007 bisect: run the actual FlowPttsLane path and print where the wall time
//! goes (load / synthesize setup / first chunk pull / full drain), vs the raw
//! ptts probe. CLI showed 2.7 s between synthesize() and the first pull.
//!
//! Run: TTS_XN_THREADS=1 cargo run --release -p tts-engine --features flow-ptts \
//!      --example flow_ptts_lane_probe -- models/phonon-31m-en

use std::time::Instant;

use tts_engine::lanes::flow_ptts::FlowPttsLane;
use tts_engine::types::{Lang, ModelRef, SynthRequest, TtsLane};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("pass the checkpoint dir, e.g. models/phonon-31m-en");

    let t0 = Instant::now();
    let mut lane = FlowPttsLane::new();
    lane.load(&ModelRef::Path(std::path::Path::new(&dir).into()))
        .expect("lane load");
    println!("# load_wall_ms {}", t0.elapsed().as_millis());

    // Warmup (discarded) mirrors the CLI gate contract: synth build is lazy
    // and lands in the FIRST request's TTFA.
    let t1 = Instant::now();
    let warm_req = synreq("warmup sentence.");
    let warm_stream = lane.synthesize(&warm_req).expect("warm synthesize");
    let setup = t1.elapsed();
    let t2 = Instant::now();
    let mut first = 0usize;
    let n = warm_stream.chunks.count();
    first = n;
    println!(
        "# warmup setup_ms {} first_pull_first_chunk_ms {} samples {first}",
        setup.as_millis(),
        t2.elapsed().as_millis()
    );

    for rep in 0..3 {
        let ta = Instant::now();
        let stream = lane.synthesize(&synreq("The engine loads its weights straight from disk.")).expect("synth");
        let setup = ta.elapsed();
        let mut total_samples = 0usize;
        let mut ttfa_ms = 0u128;
        let mut first_done = false;
        let tb = Instant::now();
        for chunk in stream.chunks {
            let pcm = chunk.expect("chunk").pcm;
            total_samples += pcm.len();
            if !first_done {
                let t = tb.elapsed().as_millis();
                println!(
                    "# rep{rep} setup_ms {} ttfa_ms {t} setup+ttfa {} samples {total_samples}",
                    setup.as_millis(),
                    (setup + std::time::Duration::from_millis(t as u64)).as_millis(),
                );
                ttfa_ms = t;
                first_done = true;
            }
        }
        let drain_ms = tb.elapsed().as_millis();
        println!(
            "# rep{rep} done: ttfa {ttfa_ms} ms drain {drain_ms} ms total_samples {total_samples} rtf {:.3}",
            total_samples as f64 / 24_000.0 / (drain_ms as f64 / 1000.0)
        );
    }
}

fn synreq(text: &str) -> SynthRequest {
    SynthRequest {
        text: text.into(),
        chunks: Vec::new(),
        lang: Lang::En,
        voice: Some(tts_engine::types::VoiceRef::Default),
        seed: Some(42),
        lane_options: Default::default(),
    }
}
