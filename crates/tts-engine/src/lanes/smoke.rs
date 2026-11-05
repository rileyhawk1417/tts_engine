//! The smoke lane — silence through the seam.
//!
//! Purpose: prove the telemetry plumbing end-to-end (header/chunk/summary
//! rows, TTFA, running RTF, RSS sampling, deterministic byte-identity) with a
//! lane that has zero model weight. It synthesizes 150 ms of true silence
//! (0-sample i16) per input word at 24 kHz, with honest measured stage times:
//! real tokenization (g2p), a real zero-fill (acoustic), a real fold pass
//! (vocoder). Sub-millisecond work floors to 0 in `chunk_ms` — that is the
//! honest reading, not a bug. `load` performs a fixed 5 ms pause so `load_ms`
//! is measurable (a stub lane's honest load work is otherwise ~0).

use std::time::{Duration, Instant};

use crate::types::{
    ChunkOut, Determinism, LaneError, LoadInfo, ModelRef, RunMeta, StageTimings, SynthRequest,
    SynthStream, TtsLane,
};

pub struct SmokeLane {
    model: String,
    model_sha: Option<String>,
    load_ms: u64,
}

const SAMPLE_RATE: u32 = 24_000;
const MILLIS_PER_WORD: u64 = 150;

impl SmokeLane {
    pub fn new() -> Self {
        Self {
            model: "smoke".into(),
            model_sha: None,
            load_ms: 0,
        }
    }
}

impl Default for SmokeLane {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsLane for SmokeLane {
    fn lane_name(&self) -> &'static str {
        "smoke"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        match model {
            ModelRef::Path(p) => {
                if !p.exists() {
                    return Err(LaneError::Load(format!(
                        "model path not found: {}",
                        p.display()
                    )));
                }
                self.model = p
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string());
                self.model_sha = crate::lanes::model_sha16(p);
            }
            ModelRef::Hub { id, .. } => {
                self.model = id.clone();
                self.model_sha = Some(model_sha_placeholder(id));
            }
            ModelRef::Pseudo(id) => {
                self.model = id.clone();
                self.model_sha = Some(model_sha_placeholder(id));
            }
        }
        // Fixed measurable pause (see module docs); honest timing around it.
        std::thread::sleep(Duration::from_millis(5));
        self.load_ms = t0.elapsed().as_millis() as u64;
        Ok(LoadInfo {
            load_ms: self.load_ms,
        })
    }

    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError> {
        if req.text.trim().is_empty() {
            return Err(LaneError::Synth("empty text".into()));
        }
        let seed = req.seed;
        let model = self.model.clone();
        let model_sha = self.model_sha.clone();
        let load_ms = self.load_ms;
        let chunks_in = if req.chunks.is_empty() {
            vec![crate::types::TextChunk {
                text: req.text.clone(),
                byte_range: [0, req.text.len() as u32],
            }]
        } else {
            req.chunks.clone()
        };
        let voice_label = req
            .voice
            .as_ref()
            .map(|v| v.label())
            .unwrap_or_else(|| "default".to_string());

        let meta = RunMeta {
            lane: "smoke",
            model,
            model_sha,
            voice: voice_label,
            sample_rate: SAMPLE_RATE,
            lang: req.lang,
            determinism: Determinism::Deterministic,
            seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "smoke-stub".to_string(),
            load_ms,
            chunk_policy: "seam-provided".to_string(),
            restart_voice: false,
        };

        let iter = chunks_in.into_iter().enumerate().map(move |(idx, chunk)| {
            let t_g = Instant::now();
            let words = chunk.text.split_whitespace().count().max(1);
            let g2p = t_g.elapsed();

            let t_a = Instant::now();
            let n = (words as u64 * SAMPLE_RATE as u64 * MILLIS_PER_WORD / 1000) as usize;
            let pcm = vec![0i16; n];
            let acoustic = t_a.elapsed();

            let t_v = Instant::now();
            let mut sum = 0i64;
            for (k, &s) in pcm.iter().enumerate() {
                sum += s as i64 + (k % 2) as i64; // real pass over the data
            }
            std::hint::black_box(sum);
            let vocoder = t_v.elapsed();

            Ok(ChunkOut {
                index: idx as u32,
                stages: StageTimings {
                    g2p,
                    acoustic,
                    vocoder,
                    io: Duration::ZERO,
                },
                pcm,
            })
        });

        Ok(SynthStream {
            meta,
            chunks: Box::new(iter),
        })
    }
}

/// Deterministic placeholder sha for hub-resolved stub models (real lanes
/// hash the actual mmap'd weights per the seam law).
fn model_sha_placeholder(id: &str) -> String {
    crate::telemetry::sha256_hex(id)[..16].to_string()
}
