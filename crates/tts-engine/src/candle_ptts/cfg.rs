//! Checkpoint `config.json` — a faithful serde port of the ptts-0.6.0 oracle's
//! `TTSConfig` graph (oracle source cached at
//! `~/.cache/tts-engine-ref/ptts-0.6.0/src/{tts_model,flow_lm,mimi,conditioners}.rs`).
//!
//! T008 law: every hyperparameter is sourced from the checkpoint; nothing is
//! hardwired by accident. Field-for-field with the oracle (including serde
//! defaults) so both lanes parse the same configs; unknown keys are ignored
//! exactly as the oracle does (no `deny_unknown_fields`).

// ---- enums shared with the transformer/flow modules (ported from
// ---- transformer.rs / flow_lm.rs / mlp.rs) -------------------------------

/// FFN activation. `erf` is the exact GELU; Pocket TTS uses `gelu(approximate="tanh")`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gelu {
    Tanh,
    #[default]
    Erf,
}

/// Flow-net time-embedding RMS norm (mlp.rs): `Rms` is the uncentred
/// audiocraft form (every phonon model sets it); `RmsVar` is the variance
/// centred form pocket-tts computes (default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeRmsNorm {
    Rms,
    #[default]
    RmsVar,
}

/// The flow net's sampler (flow_lm.rs): `Lsd` = lagrangian self-distillation,
/// `lsd_decode_steps` steps each conditioned on two times; `Drifting`/`
/// Pointwise` are pocket-tts variants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlowType {
    #[default]
    Lsd,
    Drifting,
    Pointwise,
}

// ---- conditioners.rs ----------------------------------------------------

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LutConditionerConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub tokenizer: String,
    #[serde(default)]
    pub possible_values: Option<Vec<String>>,
}

fn default_max_period() -> f32 {
    10000.0
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContinuousConditionerConfig {
    pub scale_factor: f32,
    pub dim: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f32,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct FuserConfig {
    #[serde(default)]
    pub sum: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ConditionerConfig {
    Lut {
        name: String,
        lut: LutConditionerConfig,
    },
    Continuous {
        name: String,
        continuous: ContinuousConditionerConfig,
    },
}

impl ConditionerConfig {
    pub fn name(&self) -> &str {
        match self {
            Self::Lut { name, .. } | Self::Continuous { name, .. } => name,
        }
    }
}

// ---- tts_model.rs -------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ModelId {
    // NOTE oracle keeps {sig, epoch}; phonon configs add mimi_sig/mimi_epoch,
    // which serde ignores like the oracle until speaker-mimi port needs them.
    pub sig: String,
    pub epoch: usize,
}

/// Optional separate Mimi codec used only for voice-prompt encoding (not set
/// on phonon-31m-en; pocket-tts may set it — port kept for the shared decoder).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpeakerMimiConfig {
    pub prefix: String,
    pub mimi: MimiConfig,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VoiceConfig {
    pub name: String,
    #[serde(default)]
    pub conditions: std::collections::HashMap<String, String>,
    pub prefix: String,
}

/// Learnt tokens in front of the `speaker_wavs` / `transcript_in_segment`
/// conditionings.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PrefixSinkConfig {
    #[serde(default)]
    pub dropped_token: bool,
}

fn default_audio_prompt_min_duration() -> f32 {
    10.0
}

fn default_audio_prompt_max_duration() -> f32 {
    10.0
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TTSConfig {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    pub lsd_decode_steps: usize,
    pub eos_threshold: f32,
    pub model_id: Option<ModelId>,
    #[serde(default = "default_audio_prompt_min_duration")]
    pub audio_prompt_min_duration: f32,
    #[serde(default = "default_audio_prompt_max_duration")]
    pub audio_prompt_max_duration: f32,
    /// CFG null branch: no text at all vs the text conditioner's learnt
    /// padding row (historical behavior). See oracle comment; unused until the
    /// CFG path is ported.
    #[serde(default)]
    pub cfg_null_audio_empty: bool,
    #[serde(default)]
    pub cfg_null_text_empty: bool,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub prefix_sinks: std::collections::BTreeMap<String, PrefixSinkConfig>,
    #[serde(default)]
    pub speaker_mimi: Option<SpeakerMimiConfig>,
    #[serde(default)]
    pub conditioners: Vec<ConditionerConfig>,
    #[serde(default)]
    pub fuser: Option<FuserConfig>,
    #[serde(default)]
    pub voices: Vec<VoiceConfig>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub embeddings_ext: Option<Vec<String>>,
    /// Checkpoint-only extras the oracle reads elsewhere (or ignores by
    /// contract): `temp`, `weights_name`, `tokenizer_name`. We need all three
    /// (decode temp comes from the checkpoint here, not a hardwire), so they
    /// are captured explicitly — oracle keeps them out of TTSConfig but the
    /// checkpoint file sets them; deviation is documented, not silent.
    pub temp: Option<f32>,
    #[serde(default)]
    pub weights_name: Option<String>,
    #[serde(default)]
    pub tokenizer_name: Option<String>,
}

impl TTSConfig {
    /// The FFN activation when the config names none: tanh for Pocket TTS
    /// checkpoints (no `model_id`), erf otherwise (oracle tts_model.rs).
    pub fn default_gelu(&self) -> Gelu {
        if self.model_id.is_some() {
            Gelu::Erf
        } else {
            Gelu::Tanh
        }
    }

    /// Weights artifact to load, per the checkpoint (fallback matches the
    /// oracle default). f32 safetensors is the candle-lane contract; gguf
    /// stays the ptts lane's business.
    pub fn weights_file(&self) -> &str {
        self.weights_name.as_deref().unwrap_or("model.safetensors")
    }
}

// ---- flow_lm.rs ---------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlowLMConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub max_period: f32,
    pub n_bins: usize,
    pub lut_dim: usize,
    pub flow_dim: usize,
    pub flow_depth: usize,
    pub ldim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu: Option<Gelu>,
    #[serde(default)]
    pub time_rms_norm: TimeRmsNorm,
    #[serde(default)]
    pub flow_type: FlowType,
    /// Width of the backbone output the flow net and the EOS head read: a
    /// bias-free `z_proj` from `d_model` when set, the transformer output
    /// itself when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_dim: Option<usize>,
    /// Width of the attention's queries and output, `d_model` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_dim: Option<usize>,
    /// Key/value heads, each shared by `num_heads / num_kv_heads` query heads
    /// (grouped-query attention), `num_heads` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_heads: Option<usize>,
}

// ---- mimi.rs ------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: f64,
    pub dimension: usize,
    pub quantizer_dimension: usize,
    pub quantizer_output_dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    pub transformer_d_model: usize,
    pub transformer_num_heads: usize,
    pub transformer_num_layers: usize,
    pub transformer_layer_scale: f64,
    pub transformer_context: usize,
    pub transformer_max_period: f32,
    pub transformer_dim_feedforward: usize,
    #[serde(default)]
    pub downsample_channel_wise: bool,
    /// Width of the SEANet decoder / decoder transformer output when a small
    /// decoder was distilled from a larger codec. Unset: `dimension`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_dimension: Option<usize>,
    /// Width of the encoder's latents; `dimension` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu: Option<Gelu>,
}

impl MimiConfig {
    /// Samples per output frame: `round(sample_rate / frame_rate)` — the
    /// oracle's `frame_size` (mimi.rs 134); 24000/12.5 = 1920. The seanet
    /// hop (ratios product = 120 -> 200 fps) is subdivided by the
    /// pre-quantizer downsample (stride = encoder_fps / frame_rate = 16,
    /// mimi.rs 118-133), hence `mimi.downsample.*` weights exist.
    pub fn frame_samples(&self) -> usize {
        (self.sample_rate as f64 / self.frame_rate).round() as usize
    }
}

// ---- parser -------------------------------------------------------------

pub type ParseError = String;

/// Load a checkpoint's `config.json` (dir or parent-of-file tolerance at the
/// lane level; this takes a file path).
pub fn from_checkpoint(config_json: &std::path::Path) -> Result<TTSConfig, ParseError> {
    let bytes = std::fs::read(config_json)
        .map_err(|e| format!("read {}: {e}", config_json.display()))?;
    let cfg: TTSConfig = serde_json::from_slice(&bytes)
        .map_err(|e| format!("parse {}: {e}", config_json.display()))?;
    Ok(cfg)
}
