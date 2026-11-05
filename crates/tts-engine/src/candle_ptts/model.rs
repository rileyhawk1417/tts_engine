//! Faithful port of the ptts-0.6.0 oracle's `flow_lm.rs`, `conditioners.rs`
//! and the flow-LM half of `tts_model.rs` — f32/candle only. The Mimi decoder
//! half of TTSModel lands with the mimi module in a later session.
//!
//! Port-fidelity pins (from the oracle):
//! - The text conditioner is a LUT (`flow_lm.conditioner`, after remap) with
//!   `n_bins + 1` rows; when the checkpoint has a `learnt_padding` row it is
//!   cat'ed onto the table as the LAST row and ids must never reach it.
//! - Condition providers sum into one `[1, 1, d_model]` tensor at load-time
//!   (fuser `sum`); phonon sums num_speakers("1") + duration_delta(0.0).
//! - Sampling: eos logit BEFORE noise; noise is b·ldim samples in row-major
//!   order; Lsd decode with s/t per step.
//! - `empty_text` is a zero-token prefix; no prefix sinks on phonon.

use candle_core::{Device, Result, Tensor};

/// Flatten a small rank-1/2 tensor (e.g. an [b, 1] eos logit head) to f32s.
fn to_vec1(t: &Tensor) -> Result<Vec<f32>> {
    if t.rank() <= 1 {
        t.to_vec1()
    } else {
        Ok(t.to_vec2()?.into_iter().flatten().collect())
    }
}
use candle_nn::{Linear, Module};
use std::collections::HashMap;

use super::attention::{StreamingTransformer, StreamingTransformerState};
use super::cfg as ccfg;
use super::cfg::TTSConfig;
use super::mlp::{lsd_decode, SimpleMLPAdaLN};
use super::nns::{linear, VB};
use super::rng::Rng;
use super::tok::Tokenizer;

// ---- Conditioners (oracle conditioners.rs) ----

/// A checkpoint's LUT conditioner — the text conditioner plus any LUT
/// condition provider.
pub struct LutConditioner {
    pub tokenizer: Option<Box<dyn Tokenizer>>,
    embed: Tensor, // [rows, output_dim]
    learnt_padding: Option<Tensor>, // [1, 1, output_dim]
    learnt_padding_id: Option<u32>,
    pub dim: usize,
    pub output_dim: usize,
}

impl LutConditioner {
    /// Oracle `LUTConditioner::load`: embed table (n_bins+1, dim), optional
    /// output_proj, optional learnt padding appended as the last row.
    pub fn load(
        vb: &VB,
        n_bins: usize,
        tokenizer: Option<Box<dyn Tokenizer>>,
        dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let embed = vb.get((n_bins + 1, dim), "embed.weight")?;
        let learnt_padding = if vb.contains_tensor("learnt_padding") {
            Some(vb.get((1, 1, output_dim), "learnt_padding")?)
        } else {
            None
        };
        let embed = if vb.contains_tensor("output_proj.weight") {
            let proj = linear(&vb.pp("output_proj"), dim, output_dim, false)?;
            proj.forward(&embed)?
        } else {
            embed
        };
        let (embed, learnt_padding_id) = match learnt_padding.as_ref() {
            Some(learnt_padding) => {
                let learnt_padding = learnt_padding.squeeze(0)?;
                let embed = Tensor::cat(&[&embed, &learnt_padding], 0)?;
                (embed, Some(n_bins as u32 + 1))
            }
            None => (embed, None),
        };
        Ok(Self { tokenizer, embed, learnt_padding, learnt_padding_id, dim, output_dim })
    }

    pub fn learnt_padding_id(&self) -> Option<u32> {
        self.learnt_padding_id
    }

    pub fn tokenize(&self, text: &str) -> std::result::Result<Vec<u32>, String> {
        match self.tokenizer.as_ref() {
            Some(tokenizer) => tokenizer.encode(text),
            None => Err("No tokenizer available for LutConditioner".to_string()),
        }
    }

    /// Get embeddings for token ids. Returns [1, num_tokens, output_dim].
    pub fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            let dev = self.embed.device();
            return Tensor::zeros((1, 0, self.output_dim), candle_core::DType::F32, dev);
        }
        let rows = self.learnt_padding_id.map_or(self.embed.dims()[0], |id| id as usize);
        if let Some(&id) = token_ids.iter().find(|&&id| id as usize >= rows) {
            return Err(candle_core::Error::Msg(format!(
                "token id {id} is past this checkpoint's {rows}-entry embedding table: the \
                 tokenizer.json does not belong to this checkpoint"
            )));
        }
        let ids_t = Tensor::from_vec(
            token_ids.iter().map(|&x| x as i64).collect(),
            token_ids.len(),
            self.embed.device(),
        )?;
        let emb = self.embed.index_select(&ids_t, 0)?;
        let emb = emb.reshape((1, token_ids.len(), self.output_dim))?;
        Ok(emb)
    }

    pub fn learnt_padding(&self) -> Option<&Tensor> {
        self.learnt_padding.as_ref()
    }
}

fn default_condition(name: &str) -> Option<&'static str> {
    match name {
        "num_speakers" => Some("1"),
        "duration_delta" | "padding_bonus" => Some("0.0"),
        _ => None,
    }
}

enum Condition {
    Lut { possible: Vec<String>, lut: LutConditioner },
    Continuous { cfg: ccfg::ContinuousConditionerConfig, proj: Option<Linear> },
}

fn sin_embedding(position: f32, dim: usize, max_period: f32) -> Result<Vec<f32>> {
    if dim < 4 || !dim.is_multiple_of(2) {
        return Err(candle_core::Error::Msg(format!(
            "a sinusoidal embedding needs an even dim of at least 4, got {dim}"
        )));
    }
    let half = dim / 2;
    let phases: Vec<f32> =
        (0..half).map(|i| position / max_period.powf(i as f32 / (half - 1) as f32)).collect();
    Ok(phases.iter().map(|p| p.cos()).chain(phases.iter().map(|p| p.sin())).collect())
}

/// The `condition_provider.conditioners` set, fused per the checkpoint fuser.
pub struct Conditioners {
    conditions: Vec<(String, Condition)>,
    device: Device,
}

impl Conditioners {
    /// Oracle `Conditioners::load`.
    pub fn load(vb: &VB, config: &TTSConfig) -> Result<Self> {
        let output_dim = config.flow_lm.d_model;
        let mut conditions = Vec::with_capacity(config.conditioners.len());
        for cfg in config.conditioners.iter() {
            let name = cfg.name();
            if let Some(fuser) = &config.fuser && !fuser.sum.iter().any(|s| s == name) {
                return Err(candle_core::Error::Msg(format!(
                    "conditioner {name:?} is not in the fuser's sum, the only fusion supported"
                )));
            }
            let vb = vb.pp(name);
            let condition = match cfg {
                ccfg::ConditionerConfig::Lut { lut, .. } => load_lut_condition(&vb, lut, output_dim)?,
                ccfg::ConditionerConfig::Continuous { continuous, .. } => {
                    load_continuous_condition(&vb, continuous, output_dim)?
                }
            };
            conditions.push((name.to_string(), condition));
        }
        Ok(Self { conditions, device: vb.device().clone() })
    }

    /// Oracle `Conditioners::sum` — the fuser's `[1, 1, d_model]` tensor.
    pub fn sum(&self, values: &HashMap<String, String>) -> Result<Option<Tensor>> {
        if let Some(name) =
            values.keys().find(|k| !self.conditions.iter().any(|(n, _)| n == *k))
        {
            return Err(candle_core::Error::Msg(format!(
                "no conditioner named {name:?} in this checkpoint"
            )));
        }
        let mut sum: Option<Tensor> = None;
        for (name, condition) in self.conditions.iter() {
            let Some(value) = values.get(name).map(String::as_str).or_else(|| default_condition(name)) else {
                return Err(candle_core::Error::Msg(format!(
                    "conditioner {name:?} has no default, a value has to be given for it"
                )));
            };
            let cond = match condition {
                Condition::Lut { possible, lut } => {
                    let Some(id) = possible.iter().position(|v| v == value) else {
                        return Err(candle_core::Error::Msg(format!(
                            "{value:?} is not one of the conditioner's values {possible:?}"
                        )));
                    };
                    lut.embed_tokens(&[id as u32])?
                }
                Condition::Continuous { cfg, proj } => {
                    let value: f32 = value.parse().map_err(|_| {
                        candle_core::Error::Msg(format!("{value:?} is not a number"))
                    })?;
                    let emb = sin_embedding(cfg.scale_factor * value, cfg.dim, cfg.max_period)?;
                    let emb = Tensor::from_vec(emb, (1, 1, cfg.dim), &self.device)?;
                    match proj {
                        Some(proj) => proj.forward(&emb)?,
                        None => emb,
                    }
                }
            };
            sum = Some(match sum {
                Some(s) => s.add(&cond)?,
                None => cond,
            });
        }
        Ok(sum)
    }
}

fn load_lut_condition(
    vb: &VB,
    cfg: &ccfg::LutConditionerConfig,
    output_dim: usize,
) -> Result<Condition> {
    if cfg.tokenizer != "noop" {
        return Err(candle_core::Error::Msg(format!(
            "unsupported tokenizer {:?} for a lut conditioner", cfg.tokenizer
        )));
    }
    let Some(possible) = cfg.possible_values.clone() else {
        return Err(candle_core::Error::Msg(
            "a lut conditioner without possible_values is not supported".to_string(),
        ));
    };
    let lut = LutConditioner::load(vb, cfg.n_bins, None, cfg.dim, output_dim)?;
    Ok(Condition::Lut { possible, lut })
}

fn load_continuous_condition(
    vb: &VB,
    cfg: &ccfg::ContinuousConditionerConfig,
    output_dim: usize,
) -> Result<Condition> {
    if vb.contains_tensor("learnt_padding") {
        vb.get((1, 1, output_dim), "learnt_padding")?;
    }
    let proj = if vb.contains_tensor("output_proj.weight") {
        Some(linear(&vb.pp("output_proj"), cfg.dim, output_dim, false)?)
    } else if cfg.dim == output_dim {
        None
    } else {
        return Err(candle_core::Error::Msg(format!(
            "conditioner of dim {} has no output_proj to {output_dim}", cfg.dim
        )));
    };
    Ok(Condition::Continuous { cfg: cfg.clone(), proj })
}

fn load_summed_conditions(
    vb: &VB,
    config: &TTSConfig,
    values: &HashMap<String, String>,
) -> Result<Option<Tensor>> {
    Conditioners::load(vb, config)?.sum(values)
}

// ---- FlowLM (oracle flow_lm.rs) ----

/// What a sampling step is conditioned on: the sequence start (bos) or a latent.
pub enum StepInput<'a> {
    Bos { batch: usize },
    Latent(&'a Tensor),
}

pub struct FlowLM {
    pub conditioner: LutConditioner,
    condition_providers: Option<Tensor>, // summed providers, broadcast into every step
    flow_net: SimpleMLPAdaLN,
    transformer: StreamingTransformer,
    emb_std: Tensor,
    emb_mean: Tensor,
    bos_emb: Tensor,
    pub input_linear: Linear,
    out_norm_weight: Tensor,
    out_norm_bias: Tensor,
    out_eos: Linear,
    z_proj: Option<Linear>,
    flow_type: ccfg::FlowType,
    /// Width of the backbone output the flow net and the EOS head read.
    pub dim: usize,
    pub ldim: usize,
}

#[derive(Clone)]
pub struct FlowLMState {
    pub transformer_state: StreamingTransformerState,
    pub conditions: Option<Tensor>,
}

impl FlowLM {
    pub fn load(
        vb: &VB,
        tokenizer: Box<dyn Tokenizer>,
        config: &TTSConfig,
        conditions: &HashMap<String, String>,
    ) -> Result<Self> {
        let cfg = &config.flow_lm;
        let conditioner = LutConditioner::load(
            &vb.pp("conditioner"),
            cfg.n_bins,
            Some(tokenizer),
            cfg.lut_dim,
            cfg.d_model,
        )?;
        let condition_providers = load_summed_conditions(
            &vb.pp("condition_provider.conditioners"),
            config,
            conditions,
        )?;

        let dim = cfg.z_dim.unwrap_or(cfg.d_model);
        let num_time_conds = match cfg.flow_type {
            ccfg::FlowType::Lsd => 2,
            ccfg::FlowType::Drifting | ccfg::FlowType::Pointwise => 0,
        };
        let flow_net = SimpleMLPAdaLN::load(
            &vb.pp("flow_net"),
            cfg.ldim,       // in_channels
            cfg.flow_dim,   // model_channels
            cfg.ldim,       // out_channels
            dim,            // cond_channels
            cfg.flow_depth, // num_res_blocks
            num_time_conds,
            cfg.time_rms_norm,
        )?;
        let z_proj = cfg.z_dim.map(|z| linear(vb, cfg.d_model, z, false)).transpose()?;

        let gelu_tanh = matches!(cfg.gelu.unwrap_or(config.default_gelu()), ccfg::Gelu::Tanh);
        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            cfg.d_model,
            cfg.num_heads,
            cfg.num_layers,
            None,
            cfg.dim_feedforward,
            cfg.max_period,
            cfg.attn_dim.unwrap_or(cfg.d_model),
            cfg.num_kv_heads.unwrap_or(cfg.num_heads),
            gelu_tanh,
        )?;

        let emb_std = vb.get(cfg.ldim, "emb_std")?;
        let emb_mean = vb.get(cfg.ldim, "emb_mean")?;
        let bos_emb = vb.get(cfg.ldim, "bos_emb")?;
        let input_linear = linear(&vb.pp("input_linear"), cfg.ldim, cfg.d_model, false)?;
        let out_norm_weight = vb.get(cfg.d_model, "out_norm.weight")?;
        let out_norm_bias = vb.get(cfg.d_model, "out_norm.bias")?;
        let out_eos = linear(&vb.pp("out_eos"), dim, 1, true)?;

        Ok(Self {
            conditioner,
            condition_providers,
            flow_net,
            transformer,
            emb_std,
            emb_mean,
            bos_emb,
            input_linear,
            out_norm_weight,
            out_norm_bias,
            out_eos,
            z_proj,
            flow_type: cfg.flow_type,
            dim,
            ldim: cfg.ldim,
        })
    }

    pub fn device(&self) -> &Device {
        self.emb_std.device()
    }

    pub fn init_state(&self, batch_size: usize, sequence_length: usize) -> Result<FlowLMState> {
        let transformer_state =
            self.transformer.init_state(batch_size, sequence_length, self.emb_std.device())?;
        Ok(FlowLMState { transformer_state, conditions: None })
    }

    /// Run the backbone: concat text_embeddings + input, run transformer, strip prefix.
    fn backbone(
        &self,
        input: &Tensor,
        text_embeddings: &Tensor,
        seq_len: usize,
        state: &mut FlowLMState,
    ) -> Result<Tensor> {
        let input = match state.conditions.as_ref().or(self.condition_providers.as_ref()) {
            Some(ns) => input.broadcast_add(ns)?,
            None => input.clone(),
        };
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let out = self.transformer.forward(&input, &mut state.transformer_state)?;
        let out = super::nns::layer_norm(
            &out,
            &self.out_norm_weight,
            &self.out_norm_bias,
            1e-5,
        )?;
        // Remove prefix, keep only last seq_len positions.
        let total = out.dim(1)?;
        let start = total - seq_len;
        let out = out.narrow(1, start, seq_len)?.contiguous()?;
        match &self.z_proj {
            Some(z_proj) => z_proj.forward(&out),
            None => Ok(out),
        }
    }

    fn step_embedding(&self, input: StepInput<'_>) -> Result<Tensor> {
        match input {
            StepInput::Latent(t) => Ok(t.clone()),
            StepInput::Bos { batch } => self
                .bos_emb
                .reshape((1, 1, self.ldim))?
                .broadcast_as((batch, 1, self.ldim))?
                .contiguous(),
        }
    }

    /// The tail both samplers share. `t_out` is the backbone's last position, [batch, dim].
    fn sample_from(
        &self,
        t_out: &Tensor,
        b: usize,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, Tensor)> {
        let eos_logit = self.out_eos.forward(t_out)?;
        let noise_data: Vec<f32> = (0..b * self.ldim).map(|_| rng.sample()).collect();
        let noise = Tensor::from_vec(noise_data, (b, self.ldim), t_out.device())?;
        let latent = match self.flow_type {
            ccfg::FlowType::Lsd => lsd_decode(&self.flow_net, t_out, &noise, lsd_decode_steps)?,
            ccfg::FlowType::Drifting => self.flow_net.forward(t_out, &[], &noise)?,
            ccfg::FlowType::Pointwise => {
                noise.add(&self.flow_net.forward(t_out, &[], &noise)?)?
            }
        };
        Ok((latent.reshape((b, 1, self.ldim))?, eos_logit))
    }

    /// Records a sampling step, returning the latent and the *raw* eos logit.
    pub fn sample_next_latent_parts(
        &self,
        input: StepInput<'_>,
        text_embeddings: &Tensor,
        state: &mut FlowLMState,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, Tensor)> {
        let sequence = self.step_embedding(input)?;
        let (b, s, _) = sequence.dims3()?;

        let input = self.input_linear.forward(&sequence)?;
        let t_out = self.backbone(&input, text_embeddings, s, state)?;

        let t_len = t_out.dim(1)?;
        let t_out = t_out.narrow(1, t_len - 1, 1)?.contiguous()?.reshape((b, self.dim))?;
        self.sample_from(&t_out, b, lsd_decode_steps, rng)
    }

    /// As [`Self::sample_next_latent_parts`], with classifier-free guidance:
    /// the same input runs through the conditional and the null state, and the
    /// last position combines as `t_out = (t_out - null_out)·cfg + null_out`.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn sample_next_latent_cfg(
        &self,
        input: StepInput<'_>,
        text_embeddings: &Tensor,
        state: &mut FlowLMState,
        null_state: &mut FlowLMState,
        cfg_coef: f32,
        lsd_decode_steps: usize,
        rng: &mut impl Rng,
        eos_threshold: f32,
    ) -> Result<(Tensor, bool)> {
        let sequence = self.step_embedding(input)?;
        let (b, s, _) = sequence.dims3()?;

        let x = self.input_linear.forward(&sequence)?;
        let t_out = self.backbone(&x, text_embeddings, s, state)?;
        let t_len = t_out.dim(1)?;
        let t_out = t_out.narrow(1, t_len - 1, 1)?.contiguous()?.reshape((b, self.dim))?;
        let null_out = self.backbone(&x, text_embeddings, s, null_state)?;
        let null_out =
            null_out.narrow(1, t_len - 1, 1)?.contiguous()?.reshape((b, self.dim))?;
        let t_out = t_out.sub(&null_out)?.affine(cfg_coef as f64, 0.0)?.add(&null_out)?;

        let (latent, eos_logit) = self.sample_from(&t_out, b, lsd_decode_steps, rng)?;
        Ok((latent, Self::eos_from_logit(&to_vec1(&eos_logit)?, eos_threshold)))
    }

    /// Threshold an eos logit already on the host. Only row 0 decides; an
    /// empty slice is never eos.
    pub fn eos_from_logit(eos_val: &[f32], eos_threshold: f32) -> bool {
        eos_val.first().is_some_and(|v| *v > eos_threshold)
    }

    /// Set the state's conditions (voice override goes through here later; the
    /// oracle copies `voice.conditions.or(self.conditions)` into the state).
    pub fn set_state_conditions(&self, state: &mut FlowLMState, conditions: Option<Tensor>) {
        state.conditions = conditions;
    }
}

// ---- TTSModel, flow-LM half (oracle tts_model.rs) ----

/// A prefix sink (transcript_in_segment); phonon-31m-en has none (speaker
/// sinks land with the mimi module).
pub struct PrefixSink {
    sink: Tensor, // [1, 1, d_model]
    dropped: Option<Tensor>,
}

impl PrefixSink {
    pub fn load(vb: &VB, prefix: &str, dropped_token: bool, d_model: usize) -> Result<Self> {
        let sink = vb.get((1, 1, d_model), &format!("{prefix}.sink"))?;
        let dropped = dropped_token
            .then(|| vb.get((1, 1, d_model), &format!("{prefix}.sink_dropped")))
            .transpose()?;
        Ok(Self { sink, dropped })
    }

    fn token(&self, present: bool) -> &Tensor {
        match (&self.dropped, present) {
            (Some(dropped), false) => dropped,
            _ => &self.sink,
        }
    }
}

pub struct TTSModel {
    pub flow_lm: FlowLM,
    lsd_decode_steps: usize,
    eos_threshold: f32,
    cfg_null_text_empty: bool,
    pub text_sink: Option<PrefixSink>,
}

#[derive(Clone)]
pub struct TTSState {
    pub flow_lm_state: FlowLMState,
}

impl TTSModel {
    pub fn load(
        vb: &VB,
        tokenizer: Box<dyn Tokenizer>,
        config: &TTSConfig,
        conditions: &HashMap<String, String>,
    ) -> Result<Self> {
        let flow_lm = FlowLM::load(&vb.pp("flow_lm"), tokenizer, config, conditions)?;
        // Mimi decoder + speaker_proj + speaker sinks land with the mimi module.
        let mut text_sink = None;
        for (name, sink) in config.prefix_sinks.iter() {
            match name.as_str() {
                "transcript_in_segment" => {
                    let prefix = "flow_lm.conditioner";
                    text_sink =
                        Some(PrefixSink::load(vb, prefix, sink.dropped_token, config.flow_lm.d_model)?);
                }
                other => {
                    return Err(candle_core::Error::Msg(format!(
                        "unsupported prefix sink {other:?} (speaker sinks land with the mimi module)"
                    )));
                }
            }
        }
        if !config.prefix_sinks.is_empty() && !config.cfg_null_audio_empty {
            return Err(candle_core::Error::Msg(
                "a checkpoint with prefix_sinks needs cfg_null_audio_empty: true".to_string(),
            ));
        }

        Ok(Self {
            flow_lm,
            lsd_decode_steps: config.lsd_decode_steps,
            eos_threshold: config.eos_threshold,
            cfg_null_text_empty: config.cfg_null_text_empty,
            text_sink,
        })
    }

    /// The checkpoint's eos threshold, for parity harnesses.
    pub fn eos_threshold(&self) -> f32 {
        self.eos_threshold
    }

    /// Initialize flow LM state with the given sequence length budget.
    pub fn init_flow_lm_state(&self, batch_size: usize, sequence_length: usize) -> Result<TTSState> {
        Ok(TTSState { flow_lm_state: self.flow_lm.init_state(batch_size, sequence_length)? })
    }

    /// Set the state's conditions from the loaded fuser sum (voice override
    /// later: `voice.conditions.or(defaults)`).
    pub fn set_conditions(&self, state: &mut TTSState, conditions: Option<Tensor>) {
        self.flow_lm.set_state_conditions(&mut state.flow_lm_state, conditions);
    }

    /// Run flow LM step with text tokens. Increments state.
    pub fn prompt_text(&self, state: &mut TTSState, text_tokens: &[u32]) -> Result<()> {
        let text_embeddings = self.flow_lm.conditioner.embed_tokens(text_tokens)?;
        let text_embeddings = match &self.text_sink {
            Some(sink) => Tensor::cat(&[sink.token(!text_tokens.is_empty()), &text_embeddings], 1)?,
            None => text_embeddings,
        };
        let dev = text_embeddings.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), candle_core::DType::F32, dev)?;
        self.run_backbone_and_increment(state, &text_embeddings, &empty_latents)?;
        Ok(())
    }

    /// The CFG null branch's text: nothing with `cfg_null_text_empty`, as
    /// training's packed prefix gives for a dropped transcript, else the text
    /// conditioner's learnt padding row.
    pub fn prompt_text_null(&self, state: &mut TTSState) -> Result<()> {
        if let Some(sink) = &self.text_sink {
            let token = sink.token(false);
            let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), candle_core::DType::F32, token.device())?;
            return self.run_backbone_and_increment(state, token, &empty_latents);
        }
        if self.cfg_null_text_empty {
            return Ok(());
        }
        let empty_text = match self.flow_lm.conditioner.learnt_padding() {
            None => {
                return Err(candle_core::Error::Msg(
                    "Model does not support null text prompt".to_string(),
                ));
            }
            Some(p) => p,
        };
        let dev = empty_text.device();
        let empty_latents = Tensor::zeros((1, 0, self.flow_lm.ldim), candle_core::DType::F32, dev)?;
        self.run_backbone_and_increment(state, empty_text, &empty_latents)?;
        Ok(())
    }

    /// Prompt with no voice — a no-op on checkpoints without a speaker sink
    /// (the oracle's prompt_no_voice; audio prompts land with the mimi module).
    pub fn prompt_no_voice(&self, _state: &mut TTSState) -> Result<()> {
        Ok(())
    }

    fn run_backbone_and_increment(
        &self,
        state: &mut TTSState,
        text_embeddings: &Tensor,
        backbone_input_latents: &Tensor,
    ) -> Result<()> {
        let input = self.flow_lm.input_linear.forward(backbone_input_latents)?;
        let input = Tensor::cat(&[text_embeddings, &input], 1)?;
        let _out = self
            .flow_lm
            .transformer
            .forward(&input, &mut state.flow_lm_state.transformer_state)?;
        Ok(())
    }

    /// The empty text prefix a step with no new tokens is conditioned on.
    fn empty_text(&self) -> Result<Tensor> {
        Tensor::zeros(
            (1, 0, self.flow_lm.conditioner.output_dim),
            candle_core::DType::F32,
            self.flow_lm.device(),
        )
    }

    /// One autoregressive step, returning the latent and the *raw* eos logit.
    pub fn generate_step_parts(
        &self,
        state: &mut TTSState,
        input: StepInput<'_>,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, Tensor)> {
        self.flow_lm.sample_next_latent_parts(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            self.lsd_decode_steps,
            rng,
        )
    }

    /// Run one autoregressive generation step.
    /// Returns (next_latent [B, 1, ldim], is_eos).
    pub fn generate_step(
        &self,
        state: &mut TTSState,
        input: StepInput<'_>,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, bool)> {
        let (latent, eos_logit) = self.generate_step_parts(state, input, rng)?;
        Ok((latent, self.eos_from_logit(&to_vec1(&eos_logit)?)))
    }

    /// Threshold an eos logit the caller has brought back to the host.
    pub fn eos_from_logit(&self, eos_val: &[f32]) -> bool {
        FlowLM::eos_from_logit(eos_val, self.eos_threshold)
    }

    /// As [`Self::generate_step`], with classifier-free guidance. Reads the
    /// eos logit back every step, so unlike [`Self::generate_step_parts`] it
    /// is not browser-safe.
    pub fn generate_step_cfg(
        &self,
        state: &mut TTSState,
        null_state: &mut TTSState,
        cfg_coef: f32,
        input: StepInput<'_>,
        rng: &mut impl Rng,
    ) -> Result<(Tensor, bool)> {
        self.flow_lm.sample_next_latent_cfg(
            input,
            &self.empty_text()?,
            &mut state.flow_lm_state,
            &mut null_state.flow_lm_state,
            cfg_coef,
            self.lsd_decode_steps,
            rng,
            self.eos_threshold,
        )
    }

    /// De-normalize latents for the Mimi decoder — `latent * emb_std +
    /// emb_mean` (oracle tts_model.rs `decode_latent`; the flow LM works in
    /// normalized latent space).
    pub fn denorm_latents(&self, latents: &Tensor) -> Result<Tensor> {
        latents
            .broadcast_mul(&self.flow_lm.emb_std)?
            .broadcast_add(&self.flow_lm.emb_mean)
    }
}
