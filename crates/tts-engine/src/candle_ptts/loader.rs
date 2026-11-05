//! Checkpoint tensor keys — a faithful port of the ptts-0.6.0 oracle's
//! `loader.rs` remap table (see ~/.cache/tts-engine-ref/ptts-0.6.0/src/loader.rs),
//! plus an mmap'd safetensors inventory for the loader mandate: weights are
//! never fully resident unless evidenced as a win.

use candle_core::{Device, Tensor};

/// Maps upstream checkpoint names onto this crate's module names, dropping the
/// tensors the runtime has no use for. Verbatim port of the oracle's
/// `remap_key`; keep both in lockstep — a divergence is a parity bug.
pub fn remap_key(name: &str) -> Option<String> {
    // Skip keys we don't need.
    if name.contains("flow.w_s_t")
        || name.contains("quantizer.vq")
        || name.contains("quantizer.logvar_proj")
    {
        return None;
    }

    let mut name = name.to_string();

    // Order matters: more specific replacements first.
    name = name.replace(
        "flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
        "flow_lm.speaker_proj_weight",
    );
    name = name.replace(
        "flow_lm.condition_provider.conditioners.transcript_in_segment.",
        "flow_lm.conditioner.",
    );
    name = name.replace("flow_lm.backbone.", "flow_lm.transformer.");
    name = name.replace("flow_lm.flow.", "flow_lm.flow_net.");
    name = name.replace("mimi.model.", "mimi.");

    Some(name)
}

/// Tensors the TTSModel load legitimately leaves untouched: the encoder side
/// is only pulled in by voice-prompt encoding later, and the quantizer is
/// replaced. Verbatim port of the oracle's `is_unused_by_tts_model`.
pub fn is_unused_by_tts_model(name: &str) -> bool {
    name == "flow_lm.condition_provider.conditioners.speaker_wavs.learnt_padding"
        || name.starts_with("mimi.quantizer")
        || name.starts_with("mimi.encoder")
        || name.starts_with("speaker_mimi")
        // A prefix, not the single `conv.conv.weight` the examples used to
        // name: the ws-server matched it this way and permissiveness is part
        // of the contract.
        || name.starts_with("mimi.downsample.")
}

pub type LoadError = String;

/// One tensor as declared by the safetensors header (no data touched).
#[derive(Clone, Debug)]
pub struct TensorDecl {
    /// Remapped key (None ⇒ the tensor is known-unused); used for inventory
    /// audit stats.
    pub name: String,
    pub dtype: candle_core::DType,
    pub shape: Vec<usize>,
    pub data_len: usize,
}

#[derive(Clone, Debug)]
pub struct Inventory {
    pub tensors: Vec<TensorDecl>,
    /// Sum of data_offsets spans (weights bytes actually on disk).
    pub total_bytes: usize,
    /// Tensors retained by remap (not known-unused).
    ///
    /// Semantics note (T008): this is "renamed by remap OR not flagged
    /// unused-by-tts-model", which is NOT the same set as the loader's kept
    /// map. Kept-map = `remap_key(..) == Some` (e.g. 177 on phonon-31m-en:
    /// 185 raw − 8 `flow.w_s_t` drops). `used_count` can be higher (183 here:
    /// 185 − 2) because a few keys pass remap but stay flagged unused until
    /// voice-prompt encoding pulls the encoder side in.
    pub used_count: usize,
}

impl Inventory {
    pub fn bytes_by_dtype(&self) -> Vec<(candle_core::DType, usize)> {
        let mut acc: Vec<(candle_core::DType, usize)> = Vec::new();
        for t in &self.tensors {
            match acc.iter_mut().find(|(d, _)| *d == t.dtype) {
                Some((_, n)) => *n += t.data_len,
                None => acc.push((t.dtype, t.data_len)),
            }
        }
        acc.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        acc
    }
}

/// Mmap the safetensors file and inventory its tensors WITHOUT materializing
/// any data (mmap mandate: headers only; the runtime memory-maps what it
/// reads lazily at use time).
pub fn inventory(path: &std::path::Path) -> Result<Inventory, LoadError> {
    use candle_core::safetensors::MmapedSafetensors;
    // SAFETY: read-only mmap of a checkpoint file; data is never mutated under
    // the map and the header is validated before any tensor view escapes.
    let st = unsafe { MmapedSafetensors::new(path) }
        .map_err(|e| format!("mmap {}: {e}", path.display()))?;
    let mut tensors = Vec::new();
    let mut used_count = 0usize;
    let mut total_bytes = 0usize;
    for (name, view) in st.tensors() {
        let dtype: candle_core::DType = view
            .dtype()
            .try_into()
            .map_err(|e| format!("{path:?}: dtype {e:?} not supported in candle"))?;
        let decl = TensorDecl {
            name: remap_key(&name).unwrap_or_else(|| name.clone()),
            dtype,
            shape: view.shape().to_vec(),
            data_len: view.data().len(),
        };
        if decl.name != name || !is_unused_by_tts_model(&name) {
            used_count += 1;
        }
        total_bytes += decl.data_len;
        tensors.push(decl);
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Inventory {
        tensors,
        total_bytes,
        used_count,
    })
}


/// The tensor map a checkpoint loads into: every key remaps to Some(name)
/// except those the remap table drops; shape/dtype checks happen per-request
/// at `VarBuilder::get` (candle-nn's `SimpleBackend for HashMap<String,
/// Tensor>` semantics).
///
/// DEVIATION (documented, T008): candle 0.11 has no zero-copy safetensors
/// path — even the mmap route copies into owned CPU storage — so weights are
/// materialized once here (~98 MiB f32 for phonon-31m-en), unlike the ptts
/// lane's mmap views. `inventory` (headers only) remains the mmap-mandate
/// surface; the candle lane's load must be revisited if residency matters.
pub struct Weights {
    pub map: std::collections::HashMap<String, Tensor>,
    /// Count of remap-dropped tensors (informational, for the probe audit).
    pub dropped: usize,
}

impl Weights {
    pub fn from_safetensors(path: &std::path::Path) -> Result<Self, LoadError> {
        let map = candle_core::safetensors::load(path, &Device::Cpu)
            .map_err(|e| format!("safetensors load {}: {e}", path.display()))?;
        let mut remapped =
            std::collections::HashMap::with_capacity(map.len());
        let mut dropped = 0usize;
        for (k, v) in map {
            match remap_key(&k) {
                Some(name) => {
                    if remapped.insert(name.clone(), v).is_some() {
                        return Err(format!(
                            "remap table maps multiple checkpoints entries to {name:?} (came from {k:?})"
                        ));
                    }
                }
                None => dropped += 1,
            }
        }
        Ok(Self { map: remapped, dropped })
    }

    /// A `VarBuilder` over the owned map, f32 on CPU.
    pub fn var_builder(&self) -> crate::candle_ptts::nns::VB {
        candle_nn::VarBuilder::from_tensors(
            self.map.clone(),
            candle_core::DType::F32,
            &Device::Cpu,
        )
    }
}
