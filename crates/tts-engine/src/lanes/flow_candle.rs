//! `flow-candle` lane — T008 from-scratch Phonon-31m in candle.
//!
//! Session-2: `load()` builds the real model — remap key map into owned f32
//! tensors (documented deviation: candle 0.11 has no zero-copy path),
//! `VarBuilder` over that map, `TTSModel::load` (conditioners, flow head,
//! streaming transformer) with the checkpoint's `tokenizer.json`. Synthesis
//! still fails cleanly until the decode loop / Mimi module land. Oracle for
//! parity: the `flow-ptts` lane (ptts-0.6.0), per
//! docs/plan/tasks/T008-candle-phonon-port.md.
//!
//! Telemetry contract is identical to flow-ptts (raw checkpoint text into the
//! g2p slot, fused normalize at `Duration::ZERO` once the plan port lands).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::types::{LaneError, LoadInfo, ModelRef, TtsLane};

/// Everything `load` banks; the TTSModel owns its materialized weights.
pub struct CandleCheckpoint {
    pub dir: PathBuf,
    pub cfg: crate::candle_ptts::cfg::TTSConfig,
    pub model: crate::candle_ptts::model::TTSModel,
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
        let model = crate::candle_ptts::model::TTSModel::load(
            &vb,
            Box::new(tokenizer),
            &cfg,
            &conditions,
        )
        .map_err(|e| LaneError::Load(e.to_string()))?;
        self.loaded = Some(CandleCheckpoint {
            dir: dir.to_path_buf(),
            cfg,
            model,
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
        // Session-2 honesty: the model loads (flow-LM + conditioner +
        // transformer behind it), but no audio comes out until the Mimi
        // decode module lands. Same clean-failure contract as session 1.
        let _ = req;
        Err(LaneError::Synth(
            "flow-candle port in progress (T008): the model loads; the decode loop and Mimi \
             module arrive in later sessions — see docs/plan/tasks/T008-candle-phonon-port.md"
                .into(),
        ))
    }
}
