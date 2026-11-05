//! Primitive layers shared by the candle port of the ptts flow-LM runtime.
//!
//! Session-2 (T008) slice: linear layers, layer norms, and the RoPE primitive
//! — all faithful ports of the ptts-0.6.0 oracle, f32-only, CPU device.
//!
//! Labeled deviation (documented, not silent): candle 0.11 tensors own their
//! data, so checkpoint weights are materialized once at load (~98 MiB f32 for
//! phonon-31m-en) instead of being read through mmap views zero-copy like xn.

use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear};

pub type VB = candle_nn::VarBuilder<'static>;

/// `xn::nn::Linear::load` (bias-free) / `Linear::load_b` (with bias).
pub fn linear(vb: &VB, in_dim: usize, out_dim: usize, bias: bool) -> Result<Linear> {
    let w = vb.get((out_dim, in_dim), "weight")?;
    let b = if bias { Some(vb.get(out_dim, "bias")?) } else { None };
    Ok(Linear::new(w, b))
}

/// Functional layer norm with explicit weight/bias (candle-nn's op kernel).
pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::layer_norm(x, w, b, eps as f32)
}

/// `xn::nn::LayerNorm::load` with a learned scale AND bias.
pub fn layer_norm_load(vb: &VB, dim: usize, eps: f64) -> Result<LayerNorm> {
    let w = vb.get(dim, "weight")?;
    let b = vb.get(dim, "bias")?;
    Ok(LayerNorm::new(w, b, eps))
}

/// `xn::nn::LayerNorm::new(ones, zeros, eps)` — constant norm for FinalLayer.
pub fn layer_norm_const(model_channels: usize, eps: f64, dev: &Device) -> Result<LayerNorm> {
    let ones = Tensor::ones(model_channels, candle_core::DType::F32, dev)?;
    let zeros = Tensor::zeros(model_channels, candle_core::DType::F32, dev)?;
    Ok(LayerNorm::new(ones, zeros, eps))
}

// ---- RoPE (faithful port of the oracle's rope.rs) ----

/// cos/sin tables built from ABSOLUTE positions (`pos + offset`), exactly as
/// the oracle's `RotaryEmbedding::new` does.
pub struct RopeTables {
    cos: Tensor, // [seq_len, half_dim]
    sin: Tensor, // [seq_len, half_dim]
}

impl RopeTables {
    pub fn new(
        head_dim: usize,
        offset: usize,
        seq_len: usize,
        max_period: f32,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv_freq.push(1.0f32 / max_period.powf(i as f32 / half_dim as f32));
        }

        let mut cos_data = Vec::with_capacity(seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(seq_len * half_dim);
        for pos in 0..seq_len {
            let pos = pos + offset;
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos_data, (seq_len, half_dim), device)?;
        let sin = Tensor::from_vec(sin_data, (seq_len, half_dim), device)?;
        Ok(Self { cos, sin })
    }
}

/// `xn::Tensor::rope_i` semantics: interleaveD-pair rotation on the head axis,
/// applied to [B, T, H, D] inputs (transposing to [B, H, T, D] first, like the
/// oracle's `RotaryEmbedding::forward`).
pub fn rope_apply(q: &Tensor, k: &Tensor, tables: &RopeTables) -> Result<(Tensor, Tensor)> {
    let q = rope_one(q, tables)?;
    let k = rope_one(k, tables)?;
    Ok((q, k))
}

fn rope_one(x: &Tensor, tables: &RopeTables) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    let half = d / 2;
    let x = x.transpose(1, 2)?.contiguous()?; // [b, h, t, d]
    let pairs = x.reshape((b, h, t, half, 2))?;
    let a = pairs.narrow(4, 0, 1)?.squeeze(4)?; // [b, h, t, half] — even indices
    let c = pairs.narrow(4, 1, 1)?.squeeze(4)?; // [b, h, t, half] — odd indices
    // tables rows 0..t hold absolute positions offset..offset+t.
    let cos = tables.cos.narrow(0, 0, t)?.reshape((1, 1, t, half))?;
    let sin = tables.sin.narrow(0, 0, t)?.reshape((1, 1, t, half))?;
    // dst[2i]   = src[2i] * cos - src[2i+1] * sin
    // dst[2i+1] = src[2i] * sin + src[2i+1] * cos
    let o_a = a.broadcast_mul(&cos)?.broadcast_sub(&c.broadcast_mul(&sin)?)?;
    let o_c = a.broadcast_mul(&sin)?.broadcast_add(&c.broadcast_mul(&cos)?)?;
    let out = Tensor::cat(&[o_a.unsqueeze(4)?, o_c.unsqueeze(4)?], 4)?; // [b, h, t, half, 2]
    let out = out.reshape((b, h, t, d))?;
    let out = out.transpose(1, 2)?.contiguous()?; // back to [b, t, h, d]
    Ok(out)
}
