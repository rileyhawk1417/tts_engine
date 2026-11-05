//! Checkpoint tensor keys — a faithful port of the ptts-0.6.0 oracle's
//! `loader.rs` remap table (see ~/.cache/tts-engine-ref/ptts-0.6.0/src/loader.rs),
//! plus an mmap'd safetensors inventory for the loader mandate: weights are
//! never fully resident unless evidenced as a win.

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
