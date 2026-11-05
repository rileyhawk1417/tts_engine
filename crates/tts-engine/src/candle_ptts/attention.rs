//! Faithful port of the ptts-0.6.0 oracle's streaming transformer
//! (`transformer.rs`) — both attention kinds: `FlowLm` (grouped-query,
//! preallocated cache) for the backbone and `Mimi` (append+trim ring KV) for
//! the Mimi codec transformers. The oracle's `StreamingTransformer`,
//! `StreamingTransformerLayer`, `MimiStreamingMultiheadAttention`, `KvCache`,
//! and `ProjectedTransformer` are mirrored structurally and semantically.
//!
//! Exactness contract (q-law parity): same op order, same shapes, same
//! constants. GQA k/v heads are expanded BEFORE caching, RoPE is applied
//! before the cache write, tables come from absolute positions, rope is built
//! once per transformer forward at the first layer's cache offset, and the
//! causal mask is elided when `seq_len == 1` (identically zero).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::ops::softmax_last_dim as softmax_last_dim_op;
use candle_nn::{LayerNorm, Linear, Module};

use super::cfg::Gelu;
use super::nns::{RopeTables, VB, layer_norm_load, linear, rope_apply};

// ---- FlowLm attention state ----

/// State for one FlowLm layer's attention, mirroring `StreamingMHAState`.
#[derive(Clone)]
pub struct MhaState {
    /// [batch, seq_budget, num_heads, head_dim] prealloc, zeros.
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub current_end: usize,
}

impl MhaState {
    pub fn new(
        batch: usize,
        seq_budget: usize,
        num_heads: usize,
        head_dim: usize,
        dev: &Device,
    ) -> Result<Self> {
        let shape = (batch, seq_budget, num_heads, head_dim);
        Ok(Self {
            k_cache: Tensor::zeros(shape, DType::F32, dev)?,
            v_cache: Tensor::zeros(shape, DType::F32, dev)?,
            current_end: 0,
        })
    }

    /// `complete_kv`: append k/v at current_end, then return the **whole
    /// filled prefix** `[0, current_end + t)` — every cached key is visible
    /// to this step's queries (the oracle returns `cache[: end]`, not just
    /// the new block; the causal mask decides what the attention may see).
    pub fn complete_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, h, d) = k.dims4()?;
        let start = self.current_end;
        let ranges = [0..b, start..start + t, 0..h, 0..d];
        self.k_cache = self.k_cache.slice_assign(&ranges, k)?;
        self.v_cache = self.v_cache.slice_assign(&ranges, v)?;
        self.current_end = start + t;
        let keys = self.k_cache.narrow(1, 0, self.current_end)?.contiguous()?;
        let values = self.v_cache.narrow(1, 0, self.current_end)?.contiguous()?;
        Ok((keys, values))
    }
}

// ---- Mimi KV cache (oracle transformer.rs `KvCache`) ----

/// Simple append-based KV cache with optional context window trimming.
/// k/v layout is [b, h, seq, d] (oracle's dim-2 append/trim).
#[derive(Clone)]
pub struct MimiKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    pub context: usize,
    pub absolute_offset: usize,
}

impl MimiKvCache {
    pub fn new(context: usize) -> Self {
        Self {
            k: None,
            v: None,
            context,
            absolute_offset: 0,
        }
    }

    pub fn current_seq_len(&self) -> Result<usize> {
        match &self.k {
            Some(k) => k.dim(2),
            None => Ok(0),
        }
    }

    /// Append new k, v (shape [b, h, t, d]) and return full (k, v). Does NOT
    /// trim here — the oracle appends the full window and trims after use.
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => (
                Tensor::cat(&[prev_k, new_k], 2)?,
                Tensor::cat(&[prev_v, new_v], 2)?,
            ),
            _ => (new_k.clone(), new_v.clone()),
        };
        self.absolute_offset += new_k.dim(2)?;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// Keep `context - 1` positions: with the next query, a window of
    /// `context` (pocket-tts's `delta < context`).
    pub fn trim(&mut self) -> Result<()> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(()),
        };
        let seq_len = k.dim(2)?;
        let keep = self.context.saturating_sub(1);
        if seq_len > keep {
            let trim = seq_len - keep;
            let k = k.narrow(2, trim, keep)?.contiguous()?;
            let v = v.narrow(2, trim, keep)?.contiguous()?;
            self.k = Some(k);
            self.v = Some(v);
        }
        Ok(())
    }
}

// ---- Attention kinds ----

/// Streaming multi-head attention with grouped-query keys (FlowLm kind).
pub struct StreamingMha {
    in_proj: Linear,  // [attn_dim + 2*kv_dim, embed_dim], no bias
    out_proj: Linear, // [embed_dim, attn_dim], no bias
    pub embed_dim: usize,
    pub num_heads: usize,
    pub attn_dim: usize,
    pub num_kv_heads: usize,
}

impl StreamingMha {
    pub fn load(
        vb: &VB,
        embed_dim: usize,
        num_heads: usize,
        attn_dim: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let kv_dim = attn_dim / num_heads * num_kv_heads;
        let in_proj = linear(&vb.pp("in_proj"), embed_dim, attn_dim + 2 * kv_dim, false)?;
        let out_proj = linear(&vb.pp("out_proj"), attn_dim, embed_dim, false)?;
        Ok(Self {
            in_proj,
            out_proj,
            embed_dim,
            num_heads,
            attn_dim,
            num_kv_heads,
        })
    }

    pub fn init_state(&self, batch: usize, seq_budget: usize, dev: &Device) -> Result<MhaState> {
        let head_dim = self.attn_dim / self.num_heads;
        MhaState::new(batch, seq_budget, self.num_heads, head_dim, dev)
    }

    /// query: [b, t, embed_dim] — the oracle's `StreamingMultiheadAttention::forward`.
    pub fn forward(
        &self,
        query: &Tensor,
        state: &mut MhaState,
        tables: &RopeTables,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let (h, kvh) = (self.num_heads, self.num_kv_heads);
        let d = self.attn_dim / h;
        let projected = self.in_proj.forward(query)?;
        let (ad, kd) = (self.attn_dim, kvh * d);
        let q = projected
            .narrow(2, 0, ad)?
            .contiguous()?
            .reshape((b, t, h, d))?;
        // Each k/v head serves h / kvh consecutive query heads (repeat_interleave).
        // The cache holds them expanded, so attention sees as many k/v heads as query heads.
        let heads = |x: Tensor| -> Result<Tensor> {
            if kvh == h {
                return x.reshape((b, t, h, d));
            }
            let x = x.reshape((b, t, kvh, 1, d))?;
            let x = x.broadcast_as((b, t, kvh, h / kvh, d))?.contiguous()?;
            x.reshape((b, t, h, d))
        };
        let k = heads(projected.narrow(2, ad, kd)?.contiguous()?)?;
        let v = heads(projected.narrow(2, ad + kd, kd)?.contiguous()?)?;

        // RoPE on [b, t, h, d], tables anchored at the cache's current_end
        // (the transformer builds them once per forward).
        let (q, k) = rope_apply(&q, &k, tables)?;
        let (k, v) = state.complete_kv(&k, &v)?;

        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let q2 = q.transpose(1, 2)?.contiguous()?;
        let k2 = k.transpose(1, 2)?.contiguous()?;
        let v2 = v.transpose(1, 2)?.contiguous()?;
        let mut attn = q2.matmul(&k2.transpose(2, 3)?)?; // [b, h, t, s]
        attn = attn.affine(inv_sqrt_d as f64, 0.0)?;
        if let Some(m) = mask {
            attn = attn.broadcast_add(&m.unsqueeze(0)?.unsqueeze(0)?)?;
        }
        attn = softmax_last_dim_op(&attn)?;
        let x = attn.matmul(&v2)?; // [b, h, t, d]
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.attn_dim))?;
        self.out_proj.forward(&x)
    }
}

/// Mimi's streaming multihead attention (oracle `transformer.rs:289`): packed
/// in-projection, RoPE pre-cache, append+trim KV, bias-free projections.
pub struct MimiAttention {
    in_proj: Linear,
    out_proj: Linear,
    embed_dim: usize,
    num_heads: usize,
    context: usize,
}

impl MimiAttention {
    pub fn load(vb: &VB, embed_dim: usize, num_heads: usize, context: usize) -> Result<Self> {
        let out_dim = 3 * embed_dim;
        let in_proj = linear(&vb.pp("in_proj"), embed_dim, out_dim, false)?;
        let out_proj = linear(&vb.pp("out_proj"), embed_dim, embed_dim, false)?;
        Ok(Self {
            in_proj,
            out_proj,
            embed_dim,
            num_heads,
            context,
        })
    }

    pub fn init_state(&self) -> MimiKvCache {
        MimiKvCache::new(self.context)
    }

    /// query: [b, t, embed_dim]; oracle `MimiStreamingMultiheadAttention::forward`.
    pub fn forward(
        &self,
        query: &Tensor,
        tables: &RopeTables,
        state: &mut MimiKvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let d = self.embed_dim / self.num_heads;
        let h = self.num_heads;

        let projected = self.in_proj.forward(query)?;
        let packed = projected.reshape((b, t, 3, h, d))?;
        let q = packed
            .narrow(2, 0, 1)?
            .squeeze(2)?
            .contiguous()?
            .reshape((b, t, h, d))?;
        let k = packed
            .narrow(2, 1, 1)?
            .squeeze(2)?
            .contiguous()?
            .reshape((b, t, h, d))?;
        let v = packed
            .narrow(2, 2, 1)?
            .squeeze(2)?
            .contiguous()?
            .reshape((b, t, h, d))?;

        // RoPE on [b, t, h, d]
        let (q, k) = rope_apply(&q, &k, tables)?;

        // To [b, h, t, d]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // KV cache with context trimming
        let (k, v) = state.append(&k, &v)?;

        // Attention: q [b, h, t, d] x k [b, h, s, d]
        let scale = 1.0f32 / (d as f32).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?; // [b, h, t, s]
        let attn = attn.affine(scale as f64, 0.0)?;
        // The Mimi mask is (1, 1, t, s): broadcast_add reaches it directly.
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = softmax_last_dim_op(&attn)?;
        let x = attn.matmul(&v)?; // [b, h, t, d]

        state.trim()?;

        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.embed_dim))?;
        self.out_proj.forward(&x)
    }
}

enum AttentionKind {
    Mimi(MimiAttention),
    FlowLm(StreamingMha),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mimi,
    /// Queries and the attention output are `attn_dim` wide, and keys and
    /// values have `num_kv_heads` heads (grouped-query attention).
    FlowLm {
        attn_dim: usize,
        num_kv_heads: usize,
    },
}

#[derive(Clone)]
pub enum LayerState {
    Mimi(MimiKvCache),
    FlowLm(MhaState),
}

impl LayerState {
    /// Where the next keys land: the rope offset (oracle reads the first
    /// layer's state; per-layer values agree since layers append in lockstep).
    pub fn offset(&self) -> usize {
        match self {
            LayerState::Mimi(kv) => kv.absolute_offset,
            LayerState::FlowLm(mha) => mha.current_end,
        }
    }
}

pub struct LayerScale {
    scale: Tensor, // [channels]
}

impl LayerScale {
    pub fn load(vb: &VB, channels: usize) -> Result<Self> {
        Ok(Self {
            scale: vb.get(channels, "scale")?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

pub struct Layer {
    self_attn: AttentionKind,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear, // no bias
    linear2: Linear, // no bias
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
    gelu: Gelu,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        context: Option<usize>,
        layer_scale: Option<f64>,
        kind: Kind,
    ) -> Result<Self> {
        let self_attn = match kind {
            Kind::Mimi => AttentionKind::Mimi(MimiAttention::load(
                &vb.pp("self_attn"),
                d_model,
                num_heads,
                context.unwrap_or(250),
            )?),
            Kind::FlowLm {
                attn_dim,
                num_kv_heads,
            } => AttentionKind::FlowLm(StreamingMha::load(
                &vb.pp("self_attn"),
                d_model,
                num_heads,
                attn_dim,
                num_kv_heads,
            )?),
        };

        let norm1 = layer_norm_load(&vb.pp("norm1"), d_model, 1e-5)?;
        let norm2 = layer_norm_load(&vb.pp("norm2"), d_model, 1e-5)?;
        let linear1 = linear(&vb.pp("linear1"), d_model, dim_feedforward, false)?;
        let linear2 = linear(&vb.pp("linear2"), dim_feedforward, d_model, false)?;

        let layer_scale_1 = layer_scale
            .map(|_| LayerScale::load(&vb.pp("layer_scale_1"), d_model))
            .transpose()?;
        let layer_scale_2 = layer_scale
            .map(|_| LayerScale::load(&vb.pp("layer_scale_2"), d_model))
            .transpose()?;

        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
            layer_scale_1,
            layer_scale_2,
            gelu: Gelu::default(),
        })
    }

    pub fn init_state(&self, batch: usize, seq_budget: usize, dev: &Device) -> Result<LayerState> {
        match &self.self_attn {
            AttentionKind::Mimi(attn) => Ok(LayerState::Mimi(attn.init_state())),
            AttentionKind::FlowLm(attn) => {
                Ok(LayerState::FlowLm(attn.init_state(batch, seq_budget, dev)?))
            }
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        tables: &RopeTables,
        state: &mut LayerState,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention block: x + layer_scale_1(attn(norm1(x)))
        let norm1 = self.norm1.forward(x)?;
        let mut attn_out = match (&self.self_attn, state) {
            (AttentionKind::Mimi(attn), LayerState::Mimi(cache)) => {
                attn.forward(&norm1, tables, cache, mask)?
            }
            (AttentionKind::FlowLm(attn), LayerState::FlowLm(mha)) => {
                attn.forward(&norm1, mha, tables, mask)?
            }
            _ => {
                return Err(candle_core::Error::Msg(
                    "attention kind and state type mismatch".to_string(),
                ));
            }
        };
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let x = x.add(&attn_out)?;

        // FF block: x + layer_scale_2(ff(norm2(x)))
        let norm2 = self.norm2.forward(&x)?;
        let mut ff_out = self.linear1.forward(&norm2)?;
        ff_out = match self.gelu {
            Gelu::Tanh => gelu_tanh(&ff_out)?,
            Gelu::Erf => ff_out.gelu_erf()?,
        };
        ff_out = self.linear2.forward(&ff_out)?;
        if let Some(ls) = &self.layer_scale_2 {
            ff_out = ls.forward(&ff_out)?;
        }
        x.add(&ff_out)
    }
}

/// pocket-tts's tanh GELU, `x sigmoid(2u)` with `u = sqrt(2/pi)(x + 0.044715x^3)`.
/// Only reachable for pocket checkpoints (phonon uses the exact erf GELU).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    let u2 = x.sqr()?.affine(0.071_354_82, 1.595_769)?.broadcast_mul(x)?;
    let s = candle_nn::ops::sigmoid(&u2)?;
    s.broadcast_mul(x)
}

/// The whole streaming transformer (`StreamingTransformer` port), both kinds.
pub struct StreamingTransformer {
    pub layers: Vec<Layer>,
    max_period: f32,
    head_dim: usize,
}

#[derive(Clone)]
pub struct StreamingTransformerState {
    pub layer_states: Vec<LayerState>,
}

impl StreamingTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        dim_feedforward: usize,
        context: Option<usize>,
        max_period: f32,
        kind: Kind,
    ) -> Result<Self> {
        let (attn_dim, num_kv_heads) = match kind {
            Kind::Mimi => (d_model, num_heads),
            Kind::FlowLm {
                attn_dim,
                num_kv_heads,
            } => (attn_dim, num_kv_heads),
        };
        if num_heads == 0 || !attn_dim.is_multiple_of(num_heads) {
            return Err(candle_core::Error::Msg(format!(
                "the config's {num_heads} attention heads do not divide its width {attn_dim}"
            )));
        }
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(candle_core::Error::Msg(format!(
                "the config's {num_kv_heads} key/value heads do not divide its {num_heads} attention heads"
            )));
        }
        let head_dim = attn_dim / num_heads;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Layer::load(
                &vb.pp("layers").pp(i),
                d_model,
                num_heads,
                dim_feedforward,
                context,
                layer_scale,
                kind,
            )?);
        }
        Ok(Self {
            layers,
            head_dim,
            max_period,
        })
    }

    pub fn set_gelu(&mut self, gelu: Gelu) {
        self.layers.iter_mut().for_each(|l| l.gelu = gelu);
    }

    pub fn init_state(
        &self,
        batch: usize,
        seq_budget: usize,
        dev: &Device,
    ) -> Result<StreamingTransformerState> {
        let layer_states = self
            .layers
            .iter()
            .map(|l| l.init_state(batch, seq_budget, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(StreamingTransformerState { layer_states })
    }

    pub fn forward(&self, x: &Tensor, state: &mut StreamingTransformerState) -> Result<Tensor> {
        let mut x = x.clone();
        let (_, seq_len, _) = x.dims3()?;
        // At single-token decode every cached key is visible, so the causal
        // mask is identically zero: the FlowLm branch masks nothing at all,
        // and the Mimi branch trims its cache to `context - 1` before this
        // point, which makes its window condition vacuous too. Building that
        // mask means filling a Vec on the host and adding it per layer, all
        // to add zero.
        let mask = if seq_len == 1 {
            None
        } else {
            match &state.layer_states[0] {
                LayerState::Mimi(kv_cache) => {
                    let kv_seq_len = kv_cache.current_seq_len()?;
                    let context = kv_cache.context;
                    // Causal mask (1, 1, seq_len, kv_seq_len + seq_len) with -inf
                    // where `attn_idx > seq_idx || seq_idx >= attn_idx + context`.
                    let mask_data = (0..seq_len)
                        .flat_map(|seq_idx| {
                            let seq_idx = seq_idx + kv_seq_len;
                            (0..kv_seq_len + seq_len).map(move |attn_idx| {
                                if attn_idx <= seq_idx && seq_idx < attn_idx + context {
                                    0.0f32
                                } else {
                                    f32::NEG_INFINITY
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    Some(Tensor::from_vec(
                        mask_data,
                        (1, 1, seq_len, kv_seq_len + seq_len),
                        x.device(),
                    )?)
                }
                LayerState::FlowLm(kv_cache) => {
                    // FlowLm keeps its per-forward causal mask: keys are the
                    // cache's current_end plus the new block.
                    let current_end = kv_cache.current_end;
                    let num_keys = current_end + seq_len;
                    let data = (0..seq_len)
                        .flat_map(|q| {
                            (0..num_keys).map(move |k| {
                                if k <= q + current_end {
                                    0.0f32
                                } else {
                                    f32::NEG_INFINITY
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    Some(Tensor::from_vec(data, (seq_len, num_keys), x.device())?)
                }
            }
        };
        // Rope tables once, at the first layer's cache offset — oracle
        // `StreamingTransformer::forward` (layers agree since they append in
        // lockstep). Mimi attention's per-layer forward no longer builds its
        // own; FlowLm previously built per layer with the same values.
        let offset = state.layer_states[0].offset();
        let tables = RopeTables::new(self.head_dim, offset, seq_len, self.max_period, x.device())?;
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, &tables, layer_state, mask.as_ref())?;
        }
        Ok(x)
    }
}

/// `ProjectedTransformer` port: wraps a `StreamingTransformer` with optional
/// input/output projections in conv layout [B, C, T].
pub struct ProjectedTransformer {
    pub transformer: StreamingTransformer,
    input_proj: Option<Linear>,
    output_projs: Vec<Option<Linear>>,
}

impl ProjectedTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        input_dimension: usize,
        output_dimensions: &[usize],
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        layer_scale: Option<f64>,
        context: usize,
        max_period: f32,
        dim_feedforward: usize,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::load(
            &vb.pp("transformer"),
            d_model,
            num_heads,
            num_layers,
            layer_scale,
            dim_feedforward,
            Some(context),
            max_period,
            Kind::Mimi,
        )?;

        let input_proj = if d_model != input_dimension {
            Some(linear(
                &vb.pp("input_proj"),
                input_dimension,
                d_model,
                false,
            )?)
        } else {
            None
        };

        let mut output_projs = Vec::new();
        for (i, &out_dim) in output_dimensions.iter().enumerate() {
            if d_model == out_dim {
                output_projs.push(None);
            } else {
                output_projs.push(Some(linear(
                    &vb.pp("output_proj").pp(i),
                    d_model,
                    out_dim,
                    false,
                )?));
            }
        }

        Ok(Self {
            transformer,
            input_proj,
            output_projs,
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        dev: &Device,
    ) -> Result<StreamingTransformerState> {
        self.transformer
            .init_state(batch_size, sequence_length, dev)
    }

    /// Forward pass. Input x is [B, C, T] (conv layout).
    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut StreamingTransformerState,
    ) -> Result<Vec<Tensor>> {
        // [B, C, T] -> [B, T, C]
        let x = x.transpose(1, 2)?.contiguous()?;

        let x = match &self.input_proj {
            Some(proj) => proj.forward(&x)?,
            None => x,
        };

        let z = self.transformer.forward(&x, state)?;

        let mut ys = Vec::with_capacity(self.output_projs.len());
        for proj in &self.output_projs {
            let y = match proj {
                Some(p) => p.forward(&z)?,
                None => z.clone(),
            };
            ys.push(y.transpose(1, 2)?.contiguous()?);
        }
        Ok(ys)
    }
}
