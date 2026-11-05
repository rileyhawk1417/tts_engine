//! Faithful port of the ptts-0.6.0 oracle's streaming transformer
//! (`transformer.rs`), FlowLm path only — the Mimi decode path lands with the
//! mimi module in a later session.
//!
//! Exactness contract (q-law parity): same op order, same shapes, same
//! constants. GQA k/v heads are expanded BEFORE caching, RoPE is applied
//! before the cache write, tables come from absolute positions, and the
//! causal mask is built from each layer's `current_end`.

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::ops::softmax_last_dim as softmax_last_dim_op;
use candle_nn::{LayerNorm, Linear, Module};

use super::nns::{layer_norm_load, linear, rope_apply, RopeTables, VB};

/// State for one layer's attention, mirroring `StreamingMHAState`.
#[derive(Clone)]
pub struct MhaState {
    /// [batch, seq_budget, num_heads, head_dim] prealloc, zeros.
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub current_end: usize,
}

impl MhaState {
    pub fn new(batch: usize, seq_budget: usize, num_heads: usize, head_dim: usize, dev: &Device) -> Result<Self> {
        let shape = (batch, seq_budget, num_heads, head_dim);
        Ok(Self {
            k_cache: Tensor::zeros(shape, DType::F32, dev)?,
            v_cache: Tensor::zeros(shape, DType::F32, dev)?,
            current_end: 0,
        })
    }

    /// `complete_kv`: append k/v at current_end, return the filled prefix.
    pub fn complete_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, h, d) = k.dims4()?;
        let start = self.current_end;
        let ranges = [0..b, start..start + t, 0..h, 0..d];
        self.k_cache = self.k_cache.slice_assign(&ranges, k)?;
        self.v_cache = self.v_cache.slice_assign(&ranges, v)?;
        self.current_end = start + t;
        let keys = self.k_cache.narrow(1, start, self.current_end - start)?.contiguous()?;
        let values = self.v_cache.narrow(1, start, self.current_end - start)?.contiguous()?;
        Ok((keys, values))
    }
}

/// Streaming multi-head attention with grouped-query keys (FlowLm kind).
pub struct StreamingMha {
    in_proj: Linear, // [attn_dim + 2*kv_dim, embed_dim], no bias
    out_proj: Linear, // [embed_dim, attn_dim], no bias
    pub embed_dim: usize,
    pub num_heads: usize,
    pub attn_dim: usize,
    pub num_kv_heads: usize,
}

impl StreamingMha {
    pub fn load(vb: &VB, embed_dim: usize, num_heads: usize, attn_dim: usize, num_kv_heads: usize) -> Result<Self> {
        let kv_dim = attn_dim / num_heads * num_kv_heads;
        let in_proj = linear(&vb.pp("in_proj"), embed_dim, attn_dim + 2 * kv_dim, false)?;
        let out_proj = linear(&vb.pp("out_proj"), attn_dim, embed_dim, false)?;
        Ok(Self { in_proj, out_proj, embed_dim, num_heads, attn_dim, num_kv_heads })
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
        rope_max_period: f32,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = query.dims3()?;
        let (h, kvh) = (self.num_heads, self.num_kv_heads);
        let d = self.attn_dim / h;
        let projected = self.in_proj.forward(query)?;
        let (ad, kd) = (self.attn_dim, kvh * d);
        let q = projected.narrow(2, 0, ad)?.contiguous()?.reshape((b, t, h, d))?;
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

        // RoPE on [b, t, h, d], with tables anchored at the cache's current_end.
        let offset = state.current_end;
        let tables = RopeTables::new(d, offset, t, rope_max_period, query.device())?;
        let (q, k) = rope_apply(&q, &k, &tables)?;
        let (k, v) = state.complete_kv(&k, &v)?;

        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let x = if t == 1 {
            // Decode path: [b, h, 1, d] against [b, h, s, d]. Mask is None at
            // t == 1 (the transformer builds one only for t > 1).
            let q2 = q.transpose(1, 2)?.contiguous()?;
            let k2 = k.transpose(1, 2)?.contiguous()?;
            let v2 = v.transpose(1, 2)?.contiguous()?;
            let mut scores = q2.matmul(&k2.transpose(2, 3)?)?; // [b, h, 1, s]
            scores = scores.affine(inv_sqrt_d as f64, 0.0)?;
            if let Some(m) = mask {
                scores = scores.broadcast_add(&m.unsqueeze(0)?.unsqueeze(0)?)?;
            }
            scores = softmax_last_dim_op(&scores)?;
            let x = scores.matmul(&v2)?; // [b, h, 1, d]
            x.transpose(1, 2)?.contiguous()?.reshape((b, 1, self.attn_dim))
        } else {
            let q2 = q.transpose(1, 2)?.contiguous()?; // [b, h, t, d]
            let k2 = k.transpose(1, 2)?.contiguous()?;
            let v2 = v.transpose(1, 2)?.contiguous()?;
            let mut attn = q2.matmul(&k2.transpose(2, 3)?)?; // [b, h, t, s]
            attn = attn.affine(inv_sqrt_d as f64, 0.0)?;
            if let Some(m) = mask {
                attn = attn.broadcast_add(&m.unsqueeze(0)?.unsqueeze(0)?)?;
            }
            attn = softmax_last_dim_op(&attn)?;
            let x = attn.matmul(&v2)?; // [b, h, t, d]
            let x = x.transpose(1, 2)?.contiguous()?.reshape((b, t, self.attn_dim))?;
            Ok(x)
        }?;
        self.out_proj.forward(&x)
    }
}

/// `LayerScale` port (mimi's decoder transformer uses it; the flow backbone
/// passes None).
pub struct LayerScale {
    scale: Tensor, // [channels]
}

impl LayerScale {
    pub fn load(vb: &VB, channels: usize) -> Result<Self> {
        Ok(Self { scale: vb.get(channels, "scale")? })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

/// One streaming transformer layer. Only FlowLm attention is wired this
/// session; Mimi's kind comes with the mimi module.
pub struct Layer {
    self_attn: StreamingMha,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear, // no bias
    linear2: Linear, // no bias
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
    gelu_tanh: bool,
}

impl Layer {
    pub fn load(
        vb: &VB,
        d_model: usize,
        num_heads: usize,
        dim_feedforward: usize,
        layer_scale: Option<f64>,
        kind: (usize, usize), // (attn_dim, num_kv_heads)
        gelu_tanh: bool,
    ) -> Result<Self> {
        let self_attn = StreamingMha::load(&vb.pp("self_attn"), d_model, num_heads, kind.0, kind.1)?;
        let norm1 = layer_norm_load(&vb.pp("norm1"), d_model, 1e-5)?;
        let norm2 = layer_norm_load(&vb.pp("norm2"), d_model, 1e-5)?;
        let linear1 = linear(&vb.pp("linear1"), d_model, dim_feedforward, false)?;
        let linear2 = linear(&vb.pp("linear2"), dim_feedforward, d_model, false)?;
        let layer_scale_1 = layer_scale.map(|_| LayerScale::load(&vb.pp("layer_scale_1"), d_model)).transpose()?;
        let layer_scale_2 = layer_scale.map(|_| LayerScale::load(&vb.pp("layer_scale_2"), d_model)).transpose()?;
        Ok(Self { self_attn, norm1, norm2, linear1, linear2, layer_scale_1, layer_scale_2, gelu_tanh })
    }

    pub fn init_state(&self, batch: usize, seq_budget: usize, dev: &Device) -> Result<MhaState> {
        self.self_attn.init_state(batch, seq_budget, dev)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        state: &mut MhaState,
        rope_max_period: f32,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention block: x + attn(norm1(x))  [layer_scale optional]
        let norm1 = self.norm1.forward(x)?;
        let mut attn_out = self.self_attn.forward(&norm1, state, rope_max_period, mask)?;
        if let Some(ls) = &self.layer_scale_1 {
            attn_out = ls.forward(&attn_out)?;
        }
        let x = x.add(&attn_out)?;

        // FF block: x + ff(norm2(x))
        let norm2 = self.norm2.forward(&x)?;
        let mut ff_out = self.linear1.forward(&norm2)?;
        ff_out = if self.gelu_tanh { gelu_tanh(&ff_out)? } else { ff_out.gelu_erf()? };
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

/// The whole streaming transformer, FlowLm kind (`StreamingTransformer` port).
pub struct StreamingTransformer {
    pub layers: Vec<Layer>,
    max_period: f32,
}

/// Holds one `MhaState` per layer.
#[derive(Clone)]
pub struct StreamingTransformerState {
    pub layer_states: Vec<MhaState>,
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
        max_period: f32,
        attn_dim: usize,
        num_kv_heads: usize,
        gelu_tanh: bool,
    ) -> Result<Self> {
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
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Layer::load(
                &vb.pp("layers").pp(i),
                d_model,
                num_heads,
                dim_feedforward,
                layer_scale,
                (attn_dim, num_kv_heads),
                gelu_tanh,
            )?);
        }
        Ok(Self { layers, max_period })
    }

    pub fn init_state(&self, batch: usize, seq_budget: usize, dev: &Device) -> Result<StreamingTransformerState> {
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
        // At single-token decode every cached key is visible: no mask. For t > 1
        // the FlowLm branch masks future keys against current_end.
        let num_keys = state.layer_states[0].current_end + seq_len;
        let mask = if seq_len == 1 {
            None
        } else {
            let current_end = state.layer_states[0].current_end;
            let mut data = Vec::with_capacity(seq_len * num_keys);
            for q in 0..seq_len {
                for k in 0..num_keys {
                    if k <= q + current_end {
                        data.push(0.0f32);
                    } else {
                        data.push(f32::NEG_INFINITY);
                    }
                }
            }
            Some(Tensor::from_vec(data, (seq_len, num_keys), x.device())?)
        };
        // Each layer builds its rope tables at its own cache's current_end — the
        // oracle's exact behavior (RotaryEmbedding::new per layer call). Layers
        // append in lockstep, so the offsets agree.
        for (layer, layer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            x = layer.forward(&x, layer_state, self.max_period, mask.as_ref())?;
        }
        Ok(x)
    }
}
