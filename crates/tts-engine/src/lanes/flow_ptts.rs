//! T006 — Phonon flow-latent lane (`gradium/phonon-31m-en`, Apache-2.0) behind
//! the seam, running on the `ptts` 0.6 crate (pure-Rust `xn` tensor stack: no
//! ggml, no FFI, no embedding runtime).
//!
//! Checkpoint layout on disk (gitignored): `models/phonon-31m-en/` with
//! `config.json`, `tokenizer.json`, `model.q8.gguf` (64.5 MB, Q80) and
//! `default-voice.safetensors`. The checkpoint resolver picks `model.q8.gguf`
//! first (Quant::Q80) and falls back to `model.safetensors`.
//!
//! Stage mapping per the fixed 4-slot vocabulary (`docs/seam.md` §4, kokoro
//! fused-graph rule applies):
//! - `g2p` = ptts per-language text normalization (`preprocess::Normalize`),
//!   FUSED: plan_chunks applies the synth's normalizer inside `stream_with`
//!   (lane-side pre-normalization double-applies and stretches generation —
//!   measured T007). ptts ships its own per-language normalizer —
//!   no espeak subprocess (unlike kokoro). Tokenizer + sentence split stay
//!   fused into the first acoustic pull.
//! - `acoustic` = SpeechStream chunk pulls: flow-LM generation + Mimi decode
//!   fused (Mimi IS the vocoder).
//! - `vocoder` = fused into acoustic; reported as zero, never a lie of
//!   omission — the mapping table in `docs/seam.md` §4 says so.
//! - `io` = f32 → i16 PCM assembly in the lane.
//!
//! Determinism: `SpeechOptions::seed` per request (`Determinism::Seeded`).
//! The byte-identity gate runs at a fixed seed.
//!
//! Warm-pool law (measured in T007): voice conditioning is paid per stream
//! call on `Synth`; `Session` primes it once — TTFA delta rows come from the
//! session dial, not the lane default.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ptts::checkpoint::{Checkpoint, ResolveOptions};
use ptts::preprocess::{Lang as PttsLang, Normalize};
use ptts::synth::{Quant, SpeechOptions, Synth};

use crate::types::{
    ChunkOut, Determinism, LaneError, Lang, LoadInfo, ModelRef, RunMeta, StageTimings,
    SynthRequest, SynthStream, VoiceRef,
};

/// Weight quantization this lane loads (`model.q8.gguf`), per the T006
/// contract. Higher-fidelity quants ride later tasks (T009 q-law).
const LANE_QUANT: Quant = Quant::Q80;
const QUANT_LABEL: &str = "q80";

/// Experimental dial: `TTS_PTTS_QUANT=f32` resolves `model.safetensors`
/// (f32 weights, ~99 MB) instead of the q8 gguf. Measured on Zen 2 (T009):
/// TTFA 54 vs 393 ms, drain ~1.6x faster, same audio, deterministic at
/// threads=1. Default stays q80 (gate baseline); f32 pending identity-gate
/// certification before any default flip.
fn lane_quant() -> Quant {
    match std::env::var("TTS_PTTS_QUANT").as_deref() {
        Ok("f32") => Quant::F32,
        _ => LANE_QUANT,
    }
}

struct FlowRuntime {
    synth: Synth,
    sample_rate: u32,
    voice_names: Vec<String>,
}

pub struct FlowPttsLane {
    checkpoint: Option<Checkpoint>,
    model_dir: Option<PathBuf>,
    weights_sha: Option<String>,
    weights_label: String,
    /// Synth is built at first synthesize (normalize bakes the request lang);
    /// its build wall ms is folded into `load_ms` before the first meta row.
    runtime: Option<Arc<Mutex<FlowRuntime>>>,
    built_lang: Option<Lang>,
    /// Effective xn/rayon thread count (TTS_XN_THREADS or the estimate) —
    /// decided once at load, reported in `runtime`, and used for the
    /// determinism class.
    xn_threads: usize,
    load_ms: u64,
    synth_build_ms: u64,
}

/// The xn/rayon thread dial — read once process-wide so the value reported in
/// telemetry is the value that actually sized the pool.
pub(crate) fn xn_thread_dial() -> usize {
    static DIAL: OnceLock<usize> = OnceLock::new();
    *DIAL.get_or_init(|| {
        std::env::var("TTS_XN_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(crate::lanes::physical_core_estimate)
    })
}

impl FlowPttsLane {
    pub fn new() -> Self {
        Self {
            checkpoint: None,
            model_dir: None,
            weights_sha: None,
            weights_label: String::new(),
            runtime: None,
            built_lang: None,
            xn_threads: 0,
            load_ms: 0,
            synth_build_ms: 0,
        }
    }

    /// `xn::set_num_threads` sizes rayon's global pool and must happen before
    /// the first tensor op. Guarded: a later re-dial with a different value
    /// would be ignored by rayon anyway; surface that instead of lying.
    fn apply_thread_dial() {
        static APPLIED: AtomicBool = AtomicBool::new(false);
        static SET: OnceLock<usize> = OnceLock::new();
        let want = xn_thread_dial();
        if APPLIED.swap(true, Ordering::SeqCst) {
            if let Some(prev) = SET.get()
                && *prev != want
            {
                eprintln!(
                    "[flow-ptts] TTS_XN_THREADS={want} ignored — rayon pool already sized \
                     to {prev}; set the env before the run starts"
                );
            }
            return;
        }
        let _ = SET.set(want);
        xn::set_num_threads(want);
    }

    fn build_synth(&mut self, lang: Lang, voice_path: Option<&Path>) -> Result<(), LaneError> {
        let cp = self
            .checkpoint
            .as_ref()
            .ok_or_else(|| LaneError::Synth("flow-ptts lane not loaded".into()))?;
        let t0 = Instant::now();
        let normalize = Normalize::from(ptts_lang(lang));
        let mut builder = cp.builder(normalize);
        let path_voice_name: Option<String> = voice_path.map(|p| {
            p.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string())
        });
        if let (Some(p), Some(name)) = (voice_path, path_voice_name.as_ref()) {
            builder = builder.add_voice(name.clone(), p.to_path_buf());
        }
        let mut synth: Synth = builder.build().map_err(|e| {
            LaneError::Load(format!(
                "ptts build ({} {QUANT_LABEL}): {e}",
                self.weights_label
            ))
        })?;
        // Registers voices sitting next to config.json, incl. `default` from
        // default-voice.safetensors.
        cp.register_voices(&mut synth);
        let voice_names = synth.voices();
        if voice_names.is_empty() && path_voice_name.is_none() {
            return Err(LaneError::Load(format!(
                "checkpoint {} has no voice — phonon-31m-en ships default-voice.safetensors; \
                 unconditional generation is out of seam scope",
                self.model_dir
                    .as_deref()
                    .unwrap_or(Path::new("?"))
                    .display()
            )));
        }
        let sample_rate = synth.sample_rate();
        let build = t0.elapsed().as_millis() as u64;
        self.runtime = Some(Arc::new(Mutex::new(FlowRuntime {
            synth,
            sample_rate,
            voice_names,
        })));
        self.built_lang = Some(lang);
        self.synth_build_ms = build;
        // Deliberately NOT folded into `load_ms`: the seam contract is
        // load_ms == wall of load() (kokoro precedent). The one-time synth
        // build is lazy warmup — it lands in the request's TTFA and is
        // reported separately in the `runtime` string.
        Ok(())
    }
}

impl Default for FlowPttsLane {
    fn default() -> Self {
        Self::new()
    }
}

fn ptts_lang(lang: Lang) -> PttsLang {
    match lang {
        Lang::En => PttsLang::En,
        Lang::Fr => PttsLang::Fr,
        Lang::De => PttsLang::De,
        Lang::Es => PttsLang::Es,
        Lang::Pt => PttsLang::Pt,
    }
}

impl crate::types::TtsLane for FlowPttsLane {
    fn lane_name(&self) -> &'static str {
        "flow-ptts"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        let p = match model {
            ModelRef::Path(p) => p,
            // seam law: real lanes reject Pseudo; hub download is not in scope
            // (ptts is built without the `hub` feature on purpose — weights are
            // gitignored local artifacts, fetched once).
            ModelRef::Hub { id, .. } => {
                return Err(LaneError::Load(format!(
                    "flow-ptts lane requires a disk checkpoint dir, got hub model '{id}' (fetch \
                     the gradium/phonon-31m-en artifacts into models/phonon-31m-en/)"
                )));
            }
            ModelRef::Pseudo(id) => {
                return Err(LaneError::Load(format!(
                    "flow-ptts lane requires a disk checkpoint, got pseudo model '{id}'"
                )));
            }
        };
        // Accept the checkpoint dir or any file inside it (mirror kokoro's
        // file-or-dir tolerance: dir must hold config.json).
        let dir: std::borrow::Cow<Path> = if p.is_dir() {
            std::borrow::Cow::Borrowed(p.as_path())
        } else {
            match p.parent() {
                Some(parent) => std::borrow::Cow::Owned(parent.to_path_buf()),
                None => std::borrow::Cow::Borrowed(Path::new(".")),
            }
        };
        if !dir.join("config.json").is_file() {
            return Err(LaneError::Load(format!(
                "no config.json at {} — flow-ptts loads a phonon checkpoint dir \
                 (config.json, tokenizer.json, model.q8.gguf, default-voice.safetensors)",
                dir.display()
            )));
        }
        Self::apply_thread_dial();
        let cp = Checkpoint::resolve(
            dir.as_ref(),
            ResolveOptions {
                quant: lane_quant(),
                weights: None,
            },
        )
        .map_err(|e| LaneError::Load(format!("ptts checkpoint resolve {}: {e}", dir.display())))?;
        let weights_label = cp
            .weights
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| cp.weights.display().to_string());
        let sha = crate::lanes::model_sha16(&cp.weights);
        self.model_dir = Some(dir.as_ref().to_path_buf());
        self.weights_label = weights_label;
        self.weights_sha = sha;
        self.checkpoint = Some(cp);
        self.xn_threads = xn_thread_dial();
        self.load_ms = t0.elapsed().as_millis() as u64;
        self.synth_build_ms = 0;
        self.runtime = None;
        self.built_lang = None;
        Ok(LoadInfo {
            load_ms: self.load_ms,
        })
    }

    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError> {
        if req.text.trim().is_empty() {
            return Err(LaneError::Synth("empty text".into()));
        }
        if self.checkpoint.is_none() {
            return Err(LaneError::Synth("flow-ptts lane not loaded".into()));
        }
        // Build (or rebuild) the Synth when the request language changes —
        // ptts bakes text normalization at build time. Path voices ride the
        // build (builder.add_voice) for the same reason.
        let voice_path = match req.voice.as_ref() {
            Some(VoiceRef::Path(p)) => Some(p.clone()),
            _ => None,
        };
        let path_voice_name = voice_path.as_ref().map(|p| {
            p.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string())
        });
        let rebuild = self.built_lang != Some(req.lang) || voice_path.is_some();
        if rebuild {
            self.build_synth(req.lang, voice_path.as_deref())?;
        }
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| LaneError::Synth("flow-ptts runtime missing".into()))?;
        let sample_rate = {
            let rt = runtime
                .lock()
                .map_err(|_| LaneError::Synth("flow-ptts runtime poisoned".into()))?;
            rt.sample_rate
        };
        let voice_label = match req.voice.as_ref().unwrap_or(&VoiceRef::Default) {
            VoiceRef::Default => "default".to_string(),
            VoiceRef::Id(id) => id.clone(),
            VoiceRef::Path(p) => p
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
        };
        let voice_name: Option<String> = match req.voice.as_ref().unwrap_or(&VoiceRef::Default) {
            VoiceRef::Default => None, // synth default: checkpoint's `default`
            VoiceRef::Id(id) => {
                let rt = runtime
                    .lock()
                    .map_err(|_| LaneError::Synth("flow-ptts runtime poisoned".into()))?;
                if rt.voice_names.iter().any(|v| v == id) {
                    Some(id.clone())
                } else {
                    return Err(LaneError::Synth(format!(
                        "voice '{id}' not in checkpoint voices {:?}",
                        rt.voice_names
                    )));
                }
            }
            VoiceRef::Path(_) => path_voice_name.clone(),
        };

        // ---- g2p slot: per-language text normalization (ptts built-in) ----
        // FUSED into `stream_with`: plan_chunks runs the synth's own
        // Normalize over the raw text. Lane-side pre-normalization
        // double-applies and measurably stretches generation (T007: same
        // sentence+seed, pre-normalized 103,680 samples vs raw 65,280) —
        // the raw text goes in, the g2p stage reports the fusion.
        let g2p = Duration::ZERO;

        let meta = RunMeta {
            lane: "flow-ptts",
            model: self.weights_label.clone(),
            model_sha: self.weights_sha.clone(),
            voice: voice_label,
            sample_rate,
            lang: req.lang,
            determinism: if self.xn_threads <= 1 {
                Determinism::Seeded
            } else {
                Determinism::NonDeterministic(format!(
                    "xn cpu parallel FP reductions are order-nondeterministic at \
                     {} threads; TTS_XN_THREADS=1 restores bit-identity at seed",
                    self.xn_threads
                ))
            },
            seed: req.seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: format!(
                "ptts-0.6 ({QUANT_LABEL}, cpu, xn-threads={}) load {}ms + synth {}ms",
                self.xn_threads, self.load_ms, self.synth_build_ms
            ),
            load_ms: self.load_ms,
            // Grain = the oracle SpeechStream channel item: the decode thread
            // drains up to LATENT_QUEUE=16 queued latents per decode call, so
            // batch sizes (and the chunk count) depend on thread timing — not
            // a fixed policy and not reproducible row-for-row.
            chunk_policy: "ptts-decode-batch(max16)".to_string(),
            restart_voice: false,
        };

        // Per-request seed (Determinism::Seeded); unseeded runs draw noise.
        let opts = SpeechOptions {
            voice: voice_name,
            temperature: None,
            seed: req.seed,
            cfg_coef: None,
            max_tokens_per_chunk: None,
            post_process: None,
        };

        let mut stream = {
            let rt = runtime
                .lock()
                .map_err(|_| LaneError::Synth("flow-ptts runtime poisoned".into()))?;
            rt.synth
                .stream_with(&req.text, &opts)
                .map_err(|e| LaneError::Synth(format!("ptts stream: {e}")))?
        };

        let mut idx: u32 = 0;
        let iter = std::iter::from_fn(move || {
            // ---- acoustic slot: flow-LM + Mimi decode (fused) ----
            let t_a = Instant::now();
            let item = stream.next();
            let acoustic = t_a.elapsed();
            match item {
                None => None,
                Some(Err(e)) => Some(Err(LaneError::Synth(format!("ptts stream chunk: {e}")))),
                Some(Ok(pcm_f)) => {
                    // ---- io slot: f32 -> i16 ----
                    let t_i = Instant::now();
                    let pcm: Vec<i16> = pcm_f
                        .iter()
                        .map(|&s| {
                            let v = (s.clamp(-1.0, 1.0) * 32767.0).round();
                            v as i16
                        })
                        .collect();
                    let io = t_i.elapsed();
                    let out = ChunkOut {
                        index: idx,
                        stages: StageTimings {
                            g2p: if idx == 0 { g2p } else { Duration::ZERO },
                            acoustic,
                            // Mimi decode fused into the pull (seam §4 table).
                            vocoder: Duration::ZERO,
                            io,
                        },
                        pcm,
                    };
                    idx += 1;
                    Some(Ok(out))
                }
            }
        });

        Ok(SynthStream {
            meta,
            chunks: Box::new(iter),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LaneOptions, TtsLane};

    #[test]
    fn ptts_lang_maps_every_seam_lang() {
        assert_eq!(ptts_lang(Lang::En), PttsLang::En);
        assert_eq!(ptts_lang(Lang::Fr), PttsLang::Fr);
        assert_eq!(ptts_lang(Lang::De), PttsLang::De);
        assert_eq!(ptts_lang(Lang::Es), PttsLang::Es);
        assert_eq!(ptts_lang(Lang::Pt), PttsLang::Pt);
    }

    #[test]
    fn xn_dial_is_at_least_one() {
        // May come from TTS_XN_THREADS or the physical-core estimate; either
        // way the pool sizing and the telemetry label must agree via this one
        // function (OnceLock: first caller wins process-wide).
        assert!(xn_thread_dial() >= 1);
    }

    #[test]
    fn load_rejects_hub_and_pseudo_models() {
        let mut lane = FlowPttsLane::new();
        for m in [
            ModelRef::Pseudo("pseudo-phonon".to_string()),
            ModelRef::Hub {
                id: "gradium/phonon-31m-en".to_string(),
                revision: None,
            },
        ] {
            let err = lane.load(&m).expect_err("must reject non-disk model");
            let msg = format!("{err}");
            assert!(
                msg.contains("flow-ptts lane requires a disk checkpoint"),
                "{msg}"
            );
        }
    }

    #[test]
    fn load_fails_cleanly_on_missing_checkpoint() {
        let mut lane = FlowPttsLane::new();
        let err = lane
            .load(&ModelRef::Path("/tmp/definitely-not-a-checkpoint".into()))
            .expect_err("missing dir must fail");
        let msg = format!("{err}");
        assert!(msg.contains("no config.json"), "{msg}");
    }

    #[test]
    fn synthesize_rejects_empty_text() {
        let mut lane = FlowPttsLane::new();
        let req = SynthRequest {
            text: "   ".to_string(),
            chunks: vec![],
            lang: Lang::En,
            voice: None,
            seed: Some(0),
            lane_options: LaneOptions::default(),
        };
        let err = lane.synthesize(&req).expect_err("empty text must fail");
        assert!(matches!(err, LaneError::Synth(_)));
    }
}
