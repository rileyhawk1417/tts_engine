//! Checkpoint tensor keys — a faithful port of the ptts-0.6.0 oracle's
//! `loader.rs` remap table (see ~/.cache/tts-engine-ref/ptts-0.6.0/src/loader.rs),
//! plus an mmap'd safetensors inventory for the loader mandate: weights are
//! never fully resident unless evidenced as a win.

use candle_core::{Device, Tensor};
use candle_nn::Module as _;

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
    // Pocket-tts names its speaker_wavs sink tensor after the training-side
    // bos-before-voice frame (`insert_bos_before_voice: true`): the [1, 1,
    // d_model] frame that precedes the voice segment. The KV-cache voices in
    // that checkpoint family bake it into their caches, and the tensor feeds
    // PrefixSink as `...speaker_wavs.sink` when a prompt-audio path still runs.
    name = name.replace(
        "flow_lm.bos_before_voice",
        "flow_lm.condition_provider.conditioners.speaker_wavs.sink",
    );
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
        let mut remapped = std::collections::HashMap::with_capacity(map.len());
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
        Ok(Self {
            map: remapped,
            dropped,
        })
    }

    /// A `VarBuilder` over the owned map, f32 on CPU.
    pub fn var_builder(&self) -> crate::candle_ptts::nns::VB {
        candle_nn::VarBuilder::from_tensors(self.map.clone(), candle_core::DType::F32, &Device::Cpu)
    }
}

// ---- voice assets (port of the oracle's voice halves of `loader.rs`) ----

pub const VOICE_DIRS: [&str; 2] = ["voices", "embeddings"];

/// A checkpoint's own voice, which [`checkpoint_voice_files`] registers as
/// `default`.
pub const DEFAULT_VOICE_FILE: &str = "default-voice.safetensors";

/// Oracles' `SPEAKER_PROJ_WEIGHT`, for error messages in [`project_latents`].
pub const SPEAKER_PROJ: &str = "flow_lm.speaker_proj_weight";

/// Directory and voice name of a safetensors file relative to a checkpoint.
/// Only files directly inside `voices/` or `embeddings/` are voice assets.
pub fn voice_file_name(path: &str) -> Option<(&str, &str)> {
    let (dir, file) = path.split_once('/')?;
    let name = file.strip_suffix(".safetensors")?;
    (VOICE_DIRS.contains(&dir) && !name.is_empty() && !name.contains('/')).then_some((dir, name))
}

/// Select voice names from paths listed by a checkpoint transport (the
/// oracle's `voice_paths`; ours is dir-based so only the name filter ships).
/// `voices/` takes precedence over `embeddings/` for the same voice name.
fn voice_names_in(dir: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    let mut names = std::collections::BTreeMap::new();
    for d in VOICE_DIRS {
        let sub = dir.join(d);
        let Ok(entries) = std::fs::read_dir(&sub) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") && !name.is_empty()
            {
                // BTreeMap: first dir wins ("voices" precedes "embeddings").
                names.entry(name.to_string()).or_insert(path);
            }
        }
    }
    names.into_iter().collect()
}

/// Voice files a checkpoint ships: `default-voice.safetensors` at the root as
/// `default`, plus every `voices|embeddings/*.safetensors` by file stem.
pub fn checkpoint_voice_files(dir: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    let mut out = Vec::new();
    let default = dir.join(DEFAULT_VOICE_FILE);
    if default.is_file() {
        out.push(("default".to_string(), default));
    }
    out.extend(voice_names_in(dir));
    out
}

/// Verbatim port of the oracle's `check_model_ext`: a voice asset's header
/// metadata must either omit `model_ext` or carry one of `embeddings_ext`'s
/// values (format epoch, e.g. `"ba7a5a45@300"`).
fn check_model_ext(
    header: &std::collections::HashMap<String, String>,
    label: &str,
    embeddings_ext: &[String],
) -> Result<(), LoadError> {
    // A file that records no `model_ext` is accepted: older voices predate the metadata.
    let Some(model_ext) = header.get("model_ext") else {
        return Ok(());
    };
    if !embeddings_ext.iter().any(|ext| ext == model_ext) {
        return Err(format!(
            "voice embedding model_ext '{model_ext}' in {label} is not one of config \
             embeddings_ext {embeddings_ext:?}"
        ));
    }
    Ok(())
}

/// Read a safetensors file's header: the raw `__metadata__` map. The header is
/// length-prefixed little-endian, exactly as the oracle's `read_safetensors_header`
/// reads it.
fn read_safetensors_header(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<String, String>, LoadError> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| format!("cannot read header length from {}: {e}", path.display()))?;
    let n = u64::from_le_bytes(len_bytes);
    // A file that is not safetensors can claim an absurd header length; refuse
    // to allocate on its word (oracle: 100 MiB cap).
    if n > 100 * 1024 * 1024 {
        return Err(format!(
            "{} does not look like a safetensors file",
            path.display()
        ));
    }
    let mut buf = vec![0u8; n as usize];
    file.read_exact(&mut buf)
        .map_err(|e| format!("cannot read header from {}: {e}", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&buf).map_err(|e| {
        format!(
            "cannot parse safetensors header of voice file {}: {e}",
            path.display()
        )
    })?;
    let metadata = header.get("__metadata__").cloned().unwrap_or_default();
    serde_json::from_value::<std::collections::HashMap<String, String>>(metadata)
        .map_err(|e| format!("bad __metadata__ map in {}: {e}", path.display()))
}

/// What a voice file holds, told apart by tensor name / tensor keys.
enum VoiceTensor {
    /// A voice embedding, `[T, dim]` or `[1, T, dim]`.
    Emb,
    /// Speaker-Mimi latents, `[C, T]` or `[1, C, T]`.
    Latents,
}

pub const EMB_TENSOR: &str = "emb";
pub const SPEAKER_WAVS_TENSOR: &str = "speaker_wavs";

/// A voice file's payload after [`load_voice`]: what prompts the base state
/// with. Pocket-tts ships pre-baked attention caches instead of embeddings.
#[derive(Clone)]
pub enum LoadedVoice {
    /// The flow-LM conditioning embedding `[1, T, d_model]` (phonon-style).
    Emb(Tensor),
    /// Per-layer `(k, v)` planes `[1, F, H, D]` plus the layer offset the
    /// file records, in layer order.
    Kv(Vec<(Tensor, Tensor, usize)>),
}

impl LoadedVoice {
    /// Prompt length in frames: the embedding's T, or the voice cache's end
    /// offset (pocket keeps them aligned across layers, verified at load).
    pub fn frames(&self) -> Result<usize, candle_core::Error> {
        match self {
            LoadedVoice::Emb(t) => t.dim(1),
            LoadedVoice::Kv(layers) => Ok(layers.iter().map(|(_, _, off)| *off).max().unwrap_or(0)),
        }
    }
}

/// Read a `.safetensors` voice asset and return its payload. A file holding
/// `...self_attn/cache` entries is a pocket-tts pre-baked voice cache; else an
/// `emb` / first-key embedding; else `speaker_wavs` latents.
pub fn load_voice(
    path: &std::path::Path,
    embeddings_ext: Option<&[String]>,
    speaker_proj: Option<&candle_nn::Linear>,
) -> Result<LoadedVoice, LoadError> {
    let label = format!("voice file {}", path.display());
    // The header first: a mismatch is refused before any tensor is decoded, and
    // a file that is not safetensors gets a clearer error than the map gives.
    if let Some(embeddings_ext) = embeddings_ext {
        check_model_ext(&read_safetensors_header(path)?, &label, embeddings_ext)?;
    }
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)
        .map_err(|e| format!("voice load {}: {e}", path.display()))?;
    voice_from_map(&tensors, &label, speaker_proj)
}

/// The map part of [`load_voice`]. `label` names the file in error messages.
fn voice_from_map(
    tensors: &std::collections::HashMap<String, Tensor>,
    label: &str,
    speaker_proj: Option<&candle_nn::Linear>,
) -> Result<LoadedVoice, LoadError> {
    let layer = |k: &str| -> Option<usize> {
        // `transformer.layers.{i}.self_attn/cache` — parse the layer index.
        k.strip_prefix("transformer.layers.")
            .and_then(|rest| rest.split('.').next()?.parse::<usize>().ok())
    };
    if tensors.keys().any(|k| k.ends_with("self_attn/cache")) {
        let mut cache_layers = tensors
            .keys()
            .filter_map(|k| {
                if k.ends_with("self_attn/cache") {
                    layer(k)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();
        cache_layers.sort_unstable();
        cache_layers.dedup();
        return voice_cache_from_map(tensors, &cache_layers, label);
    }
    let (name, kind) = if tensors.contains_key(EMB_TENSOR) {
        (EMB_TENSOR, VoiceTensor::Emb)
    } else if tensors.contains_key(SPEAKER_WAVS_TENSOR) {
        (SPEAKER_WAVS_TENSOR, VoiceTensor::Latents)
    } else {
        let first = tensors
            .keys()
            .next()
            .ok_or_else(|| format!("no tensors found in {label}"))?;
        (first.as_str(), VoiceTensor::Emb)
    };
    let tensor = tensors.get(name).expect("tensor name just looked up");
    let dims = tensor.dims().to_vec();
    let tensor = match dims.as_slice() {
        [a, b] => tensor
            .reshape((1, *a, *b))
            .map_err(|e| format!("voice reshape `{name}` in {label}: {e}"))?,
        [_, _, _] => tensor.clone(),
        _ => {
            return Err(format!(
                "voice tensor `{name}` in {label} has shape {dims:?}, expected two or \
                 three dimensions"
            ));
        }
    };
    let emb = match kind {
        VoiceTensor::Emb => tensor,
        VoiceTensor::Latents => project_latents(&tensor, label, speaker_proj)?,
    };
    Ok(LoadedVoice::Emb(emb))
}

/// Parse a pocket-tts pre-baked voice cache: per layer
/// `transformer.layers.{i}.self_attn/cache` `[2, 1, F, H, D]` f32 (plane 0 =
/// keys, plane 1 = values) and `.../offset` `[1]` i64 = that layer's filled
/// prefix length. Layer offsets must agree (pocket's `complete_kv` keeps the
/// voice segment aligned across layers; the base state's prompt length is one
/// number shared by all layers).
fn voice_cache_from_map(
    tensors: &std::collections::HashMap<String, Tensor>,
    layers: &[usize],
    label: &str,
) -> Result<LoadedVoice, LoadError> {
    if layers.is_empty() {
        return Err(format!("no cache planes found in {label}"));
    }
    let mut out = Vec::with_capacity(layers.len());
    let mut offset0: Option<usize> = None;
    for &l in layers {
        let cache = tensors
            .get(&format!("transformer.layers.{l}.self_attn/cache"))
            .ok_or_else(|| format!("layer {l} cache missing in {label}"))?;
        let offset_t = tensors
            .get(&format!("transformer.layers.{l}.self_attn/offset"))
            .ok_or_else(|| format!("layer {l} offset missing in {label}"))?;
        let [2, 1, f, h, d] = cache.dims() else {
            return Err(format!(
                "layer {l} cache in {label} has shape {:?}, expected [2, 1, F, H, D]",
                cache.dims()
            ));
        };
        let _ = (f, h, d);
        // Plane 0 = keys, plane 1 = values; squeeze the size-1 batch dim.
        let (k, v) = match (
            cache.narrow(0, 0, 1).and_then(|t| t.squeeze(0)),
            cache.narrow(0, 1, 1).and_then(|t| t.squeeze(0)),
        ) {
            (Ok(k), Ok(v)) => (k, v),
            (Err(e), _) | (_, Err(e)) => {
                return Err(format!("layer {l} cache planes in {label}: {e}"));
            }
        };
        let offsets = offset_t
            .to_vec1::<i64>()
            .map_err(|e| format!("layer {l} offset in {label}: {e}"))?;
        let [offset] = offsets.as_slice() else {
            return Err(format!(
                "layer {l} offset in {label} has {} entries, expected 1",
                offsets.len()
            ));
        };
        if *offset < 0 {
            return Err(format!("layer {l} offset is negative in {label}"));
        }
        let offset = *offset as usize;
        let filled = k
            .dim(1)
            .map_err(|e| format!("layer {l} cache k dims in {label}: {e}"))?;
        if offset > filled {
            return Err(format!(
                "layer {l} cache in {label} records {offset} filled frames but only holds {filled}"
            ));
        }
        match offset0 {
            None => offset0 = Some(offset),
            Some(prev) if prev != offset => {
                return Err(format!(
                    "layer {l} offset {offset} disagrees with layer {}'s \
                     {prev} in {label}: pocket voice caches keep the voice \
                     segment aligned across layers",
                    layers[0]
                ));
            }
            _ => {}
        }
        out.push((k, v, offset));
    }
    Ok(LoadedVoice::Kv(out))
}

/// Port of the oracle's `load_voice_emb`: read a `.safetensors` voice asset,
/// check its `model_ext` against `embeddings_ext`, and return the embedding as
/// `[1, T, dim]` f32 on CPU. A `speaker_wavs` file needs the `speaker_proj`
/// linear; `None` for such a file is an error (and ignored for `emb` files).
pub fn load_voice_emb(
    path: &std::path::Path,
    embeddings_ext: Option<&[String]>,
    speaker_proj: Option<&candle_nn::Linear>,
) -> Result<Tensor, LoadError> {
    match load_voice(path, embeddings_ext, speaker_proj)? {
        LoadedVoice::Emb(t) => Ok(t),
        LoadedVoice::Kv(_) => Err(format!(
            "voice file {} holds pre-baked KV caches (pocket-tts family); this \
             caller only supports embedding voices",
            path.display()
        )),
    }
}

/// Port of the oracle's `project_latents`: voice latents arrive in the encoder
/// layout `[1, C, T]`; the flow LM wants `[1, T, C]` through `speaker_proj`.
pub fn project_latents(
    tensor: &Tensor,
    label: &str,
    speaker_proj: Option<&candle_nn::Linear>,
) -> Result<Tensor, LoadError> {
    // [1, C, T] -> [1, T, C]
    let latents = tensor
        .transpose(1, 2)
        .and_then(|t| t.contiguous())
        .map_err(|e| format!("voice latents transpose in {label}: {e}"))?;
    let Some(proj) = speaker_proj else {
        return Err(format!(
            "{label} holds `{SPEAKER_WAVS_TENSOR}` latents, but this checkpoint has no \
             speaker projection (`{SPEAKER_PROJ}`) to turn them into a voice \
             embedding. A GGUF written by an older `quantize --no-mimi-encoder` dropped \
             it: regenerate the GGUF from the safetensors checkpoint, or use a \
             precomputed `{EMB_TENSOR}` voice."
        ));
    };
    let channels = latents
        .dim(2)
        .map_err(|e| format!("voice latents dims in {label}: {e}"))?;
    let in_dim = proj.weight().dims()[1];
    if channels != in_dim {
        return Err(format!(
            "`{SPEAKER_WAVS_TENSOR}` in {label} has {channels} channels but the speaker \
             projection takes {in_dim}"
        ));
    }
    proj.forward(&latents)
        .map_err(|e| format!("speaker projection: {e}"))
}

#[cfg(test)]
mod voice_tests {
    use super::*;

    #[test]
    fn voice_file_name_rejects_escape_attempts() {
        assert_eq!(
            voice_file_name("voices/a.safetensors"),
            Some(("voices", "a"))
        );
        for bad in [
            "../a.safetensors",
            "/etc/a.safetensors",
            "voices/../../a.safetensors",
            "C:a",
        ] {
            assert_eq!(voice_file_name(bad), None, "{bad}");
        }
    }
}
