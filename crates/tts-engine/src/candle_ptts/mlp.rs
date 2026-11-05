//! The flow head — faithful port of the ptts-0.6.0 oracle's `mlp.rs`
//! (TimestepEmbedder / ResBlock / FinalLayer / SimpleMLPAdaLN) plus
//! `lsd_decode` from `flow_lm.rs`.
//!
//! Port-fidelity pins (from the oracle):
//! - `modulate(x, shift, scale) = x·(scale+1) + shift`
//! - TimestepEmbedder: freqs embed (cos‖sin) → linear1 → silu → linear2 →
//!   TimeRmsNorm; phonon sets `rms` = the UNCENTRED audiocraft formula
//!   `x·α/√(mean(x²)+eps)`.
//! - ResBlock: `in_ln` LayerNorm eps **1e-6** (attention norms stay 1e-5).
//! - FinalLayer: constant ones/zeros LayerNorm eps 1e-6, then modulate, then a
//!   biased linear.

use candle_core::{Tensor, D};
use candle_nn::{LayerNorm, Linear, Module};

use super::cfg::TimeRmsNorm;
use super::nns::{layer_norm_const, layer_norm_load, linear, VB};

/// `modulate` — exact oracle order.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    let one_plus_scale = scale.affine(1.0, 1.0)?;
    x.broadcast_mul(&one_plus_scale)?.broadcast_add(shift)
}

pub struct TimestepEmbedder {
    linear1: Linear, // [hidden, fes], bias
    linear2: Linear, // [hidden, hidden], bias
    alpha: Tensor,   // [hidden]
    freqs: Tensor,   // [fes / 2]
    norm: TimeRmsNorm,
}

impl TimestepEmbedder {
    pub fn load(vb: &VB, hidden_size: usize, frequency_embedding_size: usize, norm: TimeRmsNorm) -> candle_core::Result<Self> {
        let mlp = vb.pp("mlp");
        let linear1 = linear(&mlp.pp("0"), frequency_embedding_size, hidden_size, true)?;
        let linear2 = linear(&mlp.pp("2"), hidden_size, hidden_size, true)?;
        let alpha = mlp.get(hidden_size, "3.alpha")?;
        let freqs = vb.get(frequency_embedding_size / 2, "freqs")?;
        Ok(Self { linear1, linear2, alpha, freqs, norm })
    }

    /// t: [batch, 1] → [batch, hidden]
    pub fn forward(&self, t: &Tensor) -> candle_core::Result<Tensor> {
        let args = t.broadcast_mul(&self.freqs)?;
        let cos = args.cos()?;
        let sin = args.sin()?;
        let embedding = Tensor::cat(&[&cos, &sin], D::Minus1)?;
        let mut x = self.linear1.forward(&embedding)?;
        x = x.silu()?;
        x = self.linear2.forward(&x)?;
        self.normalize(&x)
    }

    fn normalize(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let eps = 1e-5f32;
        let last = x.dims().len() - 1;
        let n = x.dims()[last] as f32;
        match self.norm {
            TimeRmsNorm::Rms => {
                // Divide in f32 first so the constant is bit-identical to the
                // oracle's `T::from_f32(1.0 / n)` (avoids double rounding).
                let m = (1.0f32 / n) as f64;
                let ms = x.sqr()?.sum_keepdim(last)?.affine(m, 0.0)?;
                x.broadcast_mul(&ms.affine(1.0, eps as f64)?.recip()?)
                    .and_then(|v| v.broadcast_mul(&self.alpha))
            }
            TimeRmsNorm::RmsVar => {
                let mean = x.sum_keepdim(last)?.affine((1.0f32 / n) as f64, 0.0)?;
                let var = x.broadcast_sub(&mean)?.sqr()?.sum_keepdim(last)?;
                let inv = var
                    .affine((1.0f32 / (n - 1.0f32)) as f64, eps as f64)?
                    .recip()?;
                x.broadcast_mul(&inv)?.broadcast_mul(&self.alpha)
            }
        }
    }
}

pub struct ResBlock {
    in_ln: LayerNorm,        // eps 1e-6
    mlp_linear1: Linear,     // bias
    mlp_linear2: Linear,     // bias
    ada_ln_silu_linear: Linear, // [3C, C], bias
}

impl ResBlock {
    pub fn load(vb: &VB, channels: usize) -> candle_core::Result<Self> {
        let in_ln = layer_norm_load(&vb.pp("in_ln"), channels, 1e-6)?;
        let mlp = vb.pp("mlp");
        let mlp_linear1 = linear(&mlp.pp("0"), channels, channels, true)?;
        let mlp_linear2 = linear(&mlp.pp("2"), channels, channels, true)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = linear(&ada.pp("1"), channels, 3 * channels, true)?;
        Ok(Self { in_ln, mlp_linear1, mlp_linear2, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, y: &Tensor) -> candle_core::Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(&y.silu()?)?;
        let channels = *x.dims().last().unwrap();

        let shift_mlp = ada.narrow(D::Minus1, 0, channels)?.contiguous()?;
        let scale_mlp = ada.narrow(D::Minus1, channels, channels)?.contiguous()?;
        let gate_mlp = ada
            .narrow(D::Minus1, 2 * channels, channels)?
            .contiguous()?;

        // h = modulate(ln(x), shift, scale)
        let h = self.in_ln.forward(x)?;
        let h = modulate(&h, &shift_mlp, &scale_mlp)?;

        // MLP
        let h = self.mlp_linear1.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp_linear2.forward(&h)?;
        x.add(&gate_mlp.broadcast_mul(&h)?)
    }
}

pub struct FinalLayer {
    norm_final: LayerNorm, // ones/zeros, eps 1e-6
    linear: Linear,        // [out, model], bias
    ada_ln_silu_linear: Linear, // [2C, C], bias
}

impl FinalLayer {
    pub fn load(vb: &VB, model_channels: usize, out_channels: usize) -> candle_core::Result<Self> {
        let norm_final = layer_norm_const(model_channels, 1e-6, vb.device())?;
        let out_linear = linear(&vb.pp("linear"), model_channels, out_channels, true)?;
        let ada = vb.pp("adaLN_modulation");
        let ada_ln_silu_linear = linear(&ada.pp("1"), model_channels, 2 * model_channels, true)?;
        Ok(Self { norm_final, linear: out_linear, ada_ln_silu_linear })
    }

    pub fn forward(&self, x: &Tensor, c: &Tensor) -> candle_core::Result<Tensor> {
        let ada = self.ada_ln_silu_linear.forward(&c.silu()?)?;
        let model_channels = *x.dims().last().unwrap();
        let shift = ada.narrow(D::Minus1, 0, model_channels)?.contiguous()?;
        let scale = ada
            .narrow(D::Minus1, model_channels, model_channels)?
            .contiguous()?;
        let x = self.norm_final.forward(x)?;
        let x = modulate(&x, &shift, &scale)?;
        self.linear.forward(&x)
    }
}

pub struct SimpleMLPAdaLN {
    time_embeds: Vec<TimestepEmbedder>,
    cond_embed: Linear,      // [model, cond], bias
    input_proj: Linear,      // [model, in], bias
    res_blocks: Vec<ResBlock>,
    final_layer: FinalLayer,
    pub num_time_conds: usize,
}

impl SimpleMLPAdaLN {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        in_channels: usize,
        model_channels: usize,
        out_channels: usize,
        cond_channels: usize,
        num_res_blocks: usize,
        num_time_conds: usize,
        time_norm: TimeRmsNorm,
    ) -> candle_core::Result<Self> {
        let mut time_embeds = Vec::new();
        for i in 0..num_time_conds {
            time_embeds.push(TimestepEmbedder::load(
                &vb.pp("time_embed").pp(i),
                model_channels,
                256, // oracle hardcodes frequency_embedding_size 256
                time_norm,
            )?);
        }
        let cond_embed = linear(&vb.pp("cond_embed"), cond_channels, model_channels, true)?;
        let input_proj = linear(&vb.pp("input_proj"), in_channels, model_channels, true)?;
        let mut res_blocks = Vec::new();
        for i in 0..num_res_blocks {
            res_blocks.push(ResBlock::load(&vb.pp("res_blocks").pp(i), model_channels)?);
        }
        let final_layer = FinalLayer::load(&vb.pp("final_layer"), model_channels, out_channels)?;
        Ok(Self { time_embeds, cond_embed, input_proj, res_blocks, final_layer, num_time_conds })
    }

    /// Forward pass.
    /// c: conditioning from AR transformer; ts: one time tensor per time embedding;
    /// x: input tensor [N, C].
    pub fn forward(&self, c: &Tensor, ts: &[&Tensor], x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = self.input_proj.forward(x)?;
        let c = self.cond_embed.forward(c)?;
        // A head without time conditions (drifting, pointwise) is conditioned on `c` alone.
        let y = match self.time_embeds.split_first() {
            None => c,
            Some((first, rest)) => {
                let mut t_combined = first.forward(ts[0])?;
                for (embed, t_input) in rest.iter().zip(ts[1..].iter()) {
                    t_combined = t_combined.add(&embed.forward(t_input)?)?;
                }
                let scale = 1.0f32 / self.num_time_conds as f32;
                t_combined.affine(scale as f64, 0.0)?.add(&c)?
            }
        };
        for block in &self.res_blocks {
            x = block.forward(&x, &y)?;
        }
        self.final_layer.forward(&x, &y)
    }
}

/// Least-squares-diffusion decode — the oracle's `lsd_decode`. For phonon
/// (num_steps == 1) this is `noise + flow_net(c, [0, 1], noise)`.
pub fn lsd_decode(
    flow_net: &SimpleMLPAdaLN,
    transformer_out: &Tensor,
    x_0: &Tensor,
    num_steps: usize,
) -> candle_core::Result<Tensor> {
    let mut current = x_0.clone();
    let dev = x_0.device();

    for i in 0..num_steps {
        let s_val = i as f32 / num_steps as f32;
        let t_val = (i + 1) as f32 / num_steps as f32;

        // s/t tensors matching x_0 shape but with last dim = 1.
        let mut shape: Vec<usize> = x_0.dims().to_vec();
        *shape.last_mut().unwrap() = 1;
        let s = Tensor::full(s_val as f64, shape.clone(), dev)?;
        let t = Tensor::full(t_val as f64, shape, dev)?;

        let flow_dir = flow_net.forward(transformer_out, &[&s, &t], &current)?;
        let step_scale = 1.0f32 / num_steps as f32;
        current = current.add(&flow_dir.affine(step_scale as f64, 0.0)?)?;
    }
    Ok(current)
}
