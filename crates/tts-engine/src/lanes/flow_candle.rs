//! `flow-candle` lane — T008 from-scratch Phonon-31m in candle.
//!
//! Session-3: end-to-end — `load()` builds the real model — remap key map
//! into owned f32 tensors (documented deviation: candle 0.11 has no zero-copy
//! path), `VarBuilder` over that map, `TTSModel::load` (conditioners, flow
//! head, streaming transformer, Mimi decoder) with the checkpoint's
//! `tokenizer.json` — registers the checkpoint's voices
//! (`default-voice.safetensors` + `voices|embeddings/*.safetensors`), and
//! `synthesize()` drives the one-shot oracle path: `plan_chunks` (normalize +
//! sentence split + token-frame budget) → voice-conditioned state priming →
//! lazy per-frame pull (flow-LM step + fused Mimi decode = 1920 PCM samples).
//! Oracle for parity: the `flow-ptts` lane (ptts-0.6.0), per
//! docs/plan/tasks/T008-candle-phonon-port.md.
//!
//! Telemetry contract is identical to flow-ptts: raw checkpoint text into the
//! g2p slot (normalize is fused into the plan port at `Duration::ZERO` —
//! lane-side pre-normalization double-applies, measured T007), fused
//! acoustic+vocoder, f32→i16 in io.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ptts::preprocess::{Lang as PttsLang, Normalize};

use crate::types::{
    ChunkOut, Determinism, LaneError, Lang, LoadInfo, ModelRef, RunMeta, StageTimings,
    SynthRequest, SynthStream, TtsLane, VoiceRef,
};

/// ptts per-language normalizer — the same mapping flow-ptts rides at
/// `Synth::build`. The candle plan port runs the identical `Normalize`
/// implementation from the ptts crate (delegated, not re-ported: identical
/// output by construction).
fn normalize_for(lang: Lang) -> Normalize {
    match lang {
        Lang::En => Normalize::for_lang(PttsLang::En),
        Lang::Fr => Normalize::for_lang(PttsLang::Fr),
        Lang::De => Normalize::for_lang(PttsLang::De),
        Lang::Es => Normalize::for_lang(PttsLang::Es),
        Lang::Pt => Normalize::for_lang(PttsLang::Pt),
    }
}

/// Everything `load` banks; the TTSModel owns its materialized weights.
pub struct CandleCheckpoint {
    pub dir: PathBuf,
    pub cfg: crate::candle_ptts::cfg::TTSConfig,
    pub model: std::sync::Arc<crate::candle_ptts::model::TTSModel>,
    pub weights_label: String,
    pub weights_sha: Option<String>,
    pub load_ms: u64,
    /// Registered voices (oracle `Synth::load`): `default-voice.safetensors`
    /// as `default`, plus every `voices|embeddings/*.safetensors` by stem.
    pub voices: Vec<crate::candle_ptts::synth::Voice>,
    pub tensor_count: usize,
    pub used_count: usize,
    pub total_bytes: usize,
}

pub struct FlowCandleLane {
    loaded: Option<CandleCheckpoint>,
}

impl FlowCandleLane {
    pub fn new() -> Self {
        Self { loaded: None }
    }
}

impl Default for FlowCandleLane {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsLane for FlowCandleLane {
    fn lane_name(&self) -> &'static str {
        "flow-candle"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        let p = match model {
            ModelRef::Path(p) => p,
            // seam law: real lanes reject Pseudo; hub download is not in scope
            ModelRef::Hub { id, .. } => {
                return Err(LaneError::Load(format!(
                    "flow-candle lane requires a disk checkpoint dir, got hub model '{id}' (fetch \
                     the gradium/phonon-31m-en artifacts into models/phonon-31m-en/)"
                )));
            }
            ModelRef::Pseudo(id) => {
                return Err(LaneError::Load(format!(
                    "flow-candle lane requires a disk checkpoint, got pseudo model '{id}'"
                )));
            }
        };
        // Accept the checkpoint dir or any file inside it (mirror flow-ptts).
        let dir: std::borrow::Cow<Path> = if p.is_dir() {
            std::borrow::Cow::Borrowed(p.as_path())
        } else {
            match p.parent() {
                Some(parent) => std::borrow::Cow::Owned(parent.to_path_buf()),
                None => std::borrow::Cow::Borrowed(Path::new(".")),
            }
        };
        let config_path = dir.join("config.json");
        if !config_path.is_file() {
            return Err(LaneError::Load(format!(
                "no config.json at {} — flow-candle loads a phonon checkpoint dir \
                 (config.json, tokenizer.model, model.safetensors, default-voice.safetensors)",
                dir.display()
            )));
        }
        let cfg =
            crate::candle_ptts::cfg::from_checkpoint(&config_path).map_err(LaneError::Load)?;
        let weights_path = dir.join(cfg.weights_file());
        if !weights_path.is_file() {
            return Err(LaneError::Load(format!(
                "no {} weights at {} — candle lane loads the checkpoint's own \
                 weights_name from config.json (f32 safetensors contract)",
                cfg.weights_file(),
                dir.display()
            )));
        }
        let inv = crate::candle_ptts::loader::inventory(&weights_path).map_err(LaneError::Load)?;
        // Oracle `Synth::load` semantics remapped onto owned tensors (see the
        // DEVIATION note in loader.rs): remap_key per entry, drop-unknowns,
        // VarBuilder on top. `load_ms` therefore includes the one-time
        // materialization (~98 MiB f32) — that's the honest load cost.
        let weights = crate::candle_ptts::loader::Weights::from_safetensors(&weights_path)
            .map_err(LaneError::Load)?;
        let vb = weights.var_builder();
        // Oracle Checkpoint::open takes dir/tokenizer.json when it exists and
        // leaves the tokenizer unset otherwise (config's tokenizer_name is
        // informational on phonon).
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer: crate::candle_ptts::tok::Tok = if tokenizer_path.is_file() {
            crate::candle_ptts::tok::Tok::open(&tokenizer_path).map_err(LaneError::Load)?
        } else {
            return Err(LaneError::Load(format!(
                "no tokenizer.json next to {} — phonon loads its tokenizer from the checkpoint dir",
                dir.display()
            )));
        };
        let conditions = HashMap::new();
        let model =
            crate::candle_ptts::model::TTSModel::load(&vb, Box::new(tokenizer), &cfg, &conditions)
                .map_err(|e| LaneError::Load(e.to_string()))?;
        // Oracle `Synth::load`: voices load with the checkpoint. The default
        // voice file registers as `default`; directory voices register by
        // file stem (`voices/` wins over `embeddings/`).
        let mut voices = Vec::new();
        for (name, path) in crate::candle_ptts::loader::checkpoint_voice_files(&dir) {
            let emb = crate::candle_ptts::loader::load_voice_emb(
                &path,
                cfg.embeddings_ext.as_deref(),
                model.speaker_proj(),
            )
            .map_err(LaneError::Load)?;
            voices.push(crate::candle_ptts::synth::Voice { name, emb });
        }
        let weights_label = weights_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| weights_path.display().to_string());
        let weights_sha = crate::lanes::model_sha16(&weights_path);
        let load_ms = t0.elapsed().as_millis() as u64;
        self.loaded = Some(CandleCheckpoint {
            dir: dir.to_path_buf(),
            cfg,
            model: std::sync::Arc::new(model),
            weights_label,
            weights_sha,
            load_ms,
            voices,
            tensor_count: inv.tensors.len(),
            used_count: inv.used_count,
            total_bytes: inv.total_bytes,
        });
        Ok(LoadInfo {
            load_ms: t0.elapsed().as_millis() as u64,
        })
    }

    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError> {
        let cp = self
            .loaded
            .as_ref()
            .ok_or_else(|| LaneError::Synth("flow-candle checkpoint not loaded".into()))?;

        let voice_label = match req.voice.as_ref().unwrap_or(&VoiceRef::Default) {
            VoiceRef::Default => "default".to_string(),
            VoiceRef::Id(id) => id.clone(),
            VoiceRef::Path(p) => p
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
        };
        // Voice resolution mirrors flow-ptts: Default rides the synth default
        // (checkpoint's `default` voice), Id must be a registered voice.
        let voice_name: Option<String> = match req.voice.as_ref().unwrap_or(&VoiceRef::Default) {
            VoiceRef::Default => None,
            VoiceRef::Id(id) => {
                if cp.voices.iter().any(|v| v.name == *id) {
                    Some(id.clone())
                } else {
                    let mut known: Vec<&str> = cp.voices.iter().map(|v| v.name.as_str()).collect();
                    known.sort();
                    return Err(LaneError::Synth(format!(
                        "voice '{id}' not in checkpoint voices {known:?}"
                    )));
                }
            }
            VoiceRef::Path(_) => Some(voice_label.clone()),
        };

        // ---- g2p slot: fused into the plan port (see module header) ----
        let g2p = Duration::ZERO;

        let opts = crate::candle_ptts::synth::SynthOptions {
            voice: voice_name.as_deref(),
            temperature: None,
            seed: req.seed,
        };
        // Oracle `stream_with`: plan chunks + prime the state up front; pull
        // frames lazily. Driver owns an Arc'd model — the lane stays free for
        // other callers.
        let mut driver = crate::candle_ptts::synth::SynthDriver::new(
            &cp.model,
            &cp.voices,
            &req.text,
            &opts,
            normalize_for(req.lang),
        )
        .map_err(|e| LaneError::Synth(format!("candle stream: {e}")))?;

        let meta = RunMeta {
            lane: "flow-candle",
            model: cp.weights_label.clone(),
            model_sha: cp.weights_sha.clone(),
            voice: voice_label,
            sample_rate: cp.model.sample_rate() as u32,
            lang: req.lang,
            // Candle + matrixmultiply is deterministic per shape (T008
            // session-2 probe); the gate verifies rep-identity for real.
            determinism: Determinism::Seeded,
            seed: req.seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: format!("candle-0.11 (f32, cpu) load {}ms", cp.load_ms),
            load_ms: cp.load_ms,
            // Grain = one decoded frame (1920 samples): the single-threaded
            // inline decode path (the port runs no decode queue, so each frame
            // is its own ChunkOut). The ptts lane's grain is the oracle decode
            // thread's drain-16 batch — timing-dependent, not reproducible.
            chunk_policy: "frame:1920".to_string(),
            restart_voice: false,
        };

        let mut idx: u32 = 0;
        let iter = std::iter::from_fn(move || {
            // ---- acoustic slot: flow-LM + Mimi decode (fused) ----
            let t_a = Instant::now();
            let item = driver.next();
            let acoustic = t_a.elapsed();
            match item {
                None => None,
                Some(Err(e)) => Some(Err(LaneError::Synth(format!("candle stream chunk: {e}")))),
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
