//! `flow-candle` lane — T008 from-scratch Phonon-31m in candle.
//!
//! Session-1 scaffold: `load()` parses the checkpoint config and inventories
//! the mmap'd safetensors weights (mmap mandate: nothing materialized), and
//! `synthesize()` fails cleanly — the transformer / flow head / Mimi decode
//! port in later sessions. Oracle for parity: the `flow-ptts` lane
//! (ptts-0.6.0), per docs/plan/tasks/T008-candle-phonon-port.md.
//!
//! Telemetry contract is identical to flow-ptts (raw checkpoint text into the
//! g2p slot, fused normalize at `Duration::ZERO` once the plan port lands).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::types::{LaneError, LoadInfo, ModelRef, TtsLane};

/// Everything `load` banks for later sessions; no mmap handle is retained
/// (re-open lazily at use time — mmap mandate).
pub struct CandleCheckpoint {
    pub dir: PathBuf,
    pub cfg: crate::candle_ptts::cfg::TTSConfig,
    pub weights_path: PathBuf,
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
        let cfg = crate::candle_ptts::cfg::from_checkpoint(&config_path)
            .map_err(LaneError::Load)?;
        let weights_path = dir.join(cfg.weights_file());
        if !weights_path.is_file() {
            return Err(LaneError::Load(format!(
                "no {} weights at {} — candle lane loads the checkpoint's own \
                 weights_name from config.json (f32 safetensors contract)",
                cfg.weights_file(),
                dir.display()
            )));
        }
        let inv = crate::candle_ptts::loader::inventory(&weights_path)
            .map_err(LaneError::Load)?;
        self.loaded = Some(CandleCheckpoint {
            dir: dir.to_path_buf(),
            cfg,
            weights_path,
            tensor_count: inv.tensors.len(),
            used_count: inv.used_count,
            total_bytes: inv.total_bytes,
        });
        Ok(LoadInfo {
            load_ms: t0.elapsed().as_millis() as u64,
        })
    }

    fn synthesize(
        &mut self,
        req: &crate::types::SynthRequest,
    ) -> Result<crate::types::SynthStream, LaneError> {
        // Session-1 honesty: no fabricated audio. The port lands module by
        // module behind this clean failure.
        let _ = req;
        Err(LaneError::Synth(
            "flow-candle port in progress (T008): transformer/flow/EOS/Mimi decode arrive in \
             later sessions — see docs/plan/tasks/T008-candle-phonon-port.md"
                .into(),
        ))
    }
}
