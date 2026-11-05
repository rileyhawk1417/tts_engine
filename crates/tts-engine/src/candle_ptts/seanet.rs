//! SEANet resnet block + decoder — faithful port of oracle `seanet.rs` (383
//! lines). The encoder and its tests are omitted: the flow path never builds
//! one (`mimi.encoder` is listed in `is_unused_by_tts_model`), and voice
//! latents arrive pre-encoded as `speaker_wavs` [1, 512, T] tensors. Conv
//! tensor names keep the Python layout (`block.{2*i+1}`, `model.{i}`).

#![allow(dead_code)]

use candle_core::{Result, Tensor};

use super::conv::{
    PadMode, StreamingConv1d, StreamingConv1dState, StreamingConvTr1dState,
    StreamingConvTranspose1d,
};
use super::nns::VB;

// ---- SEANetResnetBlock ----

pub struct SEANetResnetBlock {
    convs: Vec<StreamingConv1d>,
}

#[derive(Debug, Clone)]
pub struct SEANetResnetBlockState {
    conv_states: Vec<StreamingConv1dState>,
}

impl SEANetResnetBlock {
    pub fn load(
        vb: &VB,
        dim: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        pad_mode: PadMode,
        compress: usize,
    ) -> Result<Self> {
        if compress == 0 || compress > dim {
            return Err(candle_core::Error::Msg(format!(
                "the config's SEANet compress is {compress}; it must be from 1 to {dim}"
            )));
        }
        let hidden = dim / compress;
        let mut convs = Vec::new();
        for (i, (&ks, &dil)) in kernel_sizes.iter().zip(dilations.iter()).enumerate() {
            let in_c = if i == 0 { dim } else { hidden };
            let out_c = if i == kernel_sizes.len() - 1 {
                dim
            } else {
                hidden
            };
            // block.{2*i+1} in Python (ELU at even indices, Conv at odd)
            let conv = StreamingConv1d::load(
                &vb.pp("block").pp(2 * i + 1),
                in_c,
                out_c,
                ks,
                1,
                dil,
                pad_mode,
                1,
                true,
            )?;
            convs.push(conv);
        }
        Ok(Self { convs })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        dev: &candle_core::Device,
    ) -> Result<SEANetResnetBlockState> {
        let conv_states = self
            .convs
            .iter()
            .map(|c| c.init_state(batch_size, dev))
            .collect::<Result<Vec<_>>>()?;
        Ok(SEANetResnetBlockState { conv_states })
    }

    pub fn forward(&self, x: &Tensor, state: &mut SEANetResnetBlockState) -> Result<Tensor> {
        let mut v = x.clone();
        for (conv, cs) in self.convs.iter().zip(state.conv_states.iter_mut()) {
            v = v.elu(1.0)?;
            v = conv.forward(&v, cs)?;
        }
        x.add(&v)
    }
}

// ---- SEANetDecoder ----

struct DecoderLayer {
    upsample: StreamingConvTranspose1d,
    residuals: Vec<SEANetResnetBlock>,
}

pub struct SEANetDecoder {
    init_conv: StreamingConv1d,
    layers: Vec<DecoderLayer>,
    final_conv: StreamingConv1d,
}

#[derive(Debug, Clone)]
pub struct SEANetDecoderState {
    init_conv_state: StreamingConv1dState,
    layer_states: Vec<(StreamingConvTr1dState, Vec<SEANetResnetBlockState>)>,
    final_conv_state: StreamingConv1dState,
}

impl SEANetDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        channels: usize,
        dimension: usize,
        n_filters: usize,
        n_residual_layers: usize,
        ratios: &[usize],
        kernel_size: usize,
        last_kernel_size: usize,
        residual_kernel_size: usize,
        dilation_base: usize,
        pad_mode: PadMode,
        compress: usize,
    ) -> Result<Self> {
        let mut mult = 1 << ratios.len();

        let init_conv = StreamingConv1d::load(
            &vb.pp("model").pp(0),
            dimension,
            mult * n_filters,
            kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        let mut layers = Vec::new();
        let mut layer_idx = 1usize;

        for &ratio in ratios {
            // ELU at layer_idx, upsample at layer_idx+1
            let upsample = StreamingConvTranspose1d::load(
                &vb.pp("model").pp(layer_idx + 1),
                mult * n_filters,
                mult * n_filters / 2,
                ratio * 2,
                ratio,
                1,
                true,
            )?;
            layer_idx += 2;

            let mut residuals = Vec::new();
            for j in 0..n_residual_layers {
                let dilation = dilation_base.pow(j as u32);
                let block = SEANetResnetBlock::load(
                    &vb.pp("model").pp(layer_idx),
                    mult * n_filters / 2,
                    &[residual_kernel_size, 1],
                    &[dilation, 1],
                    pad_mode,
                    compress,
                )?;
                residuals.push(block);
                layer_idx += 1;
            }

            layers.push(DecoderLayer {
                upsample,
                residuals,
            });
            mult /= 2;
        }

        // ELU at layer_idx, final conv at layer_idx+1
        let final_conv = StreamingConv1d::load(
            &vb.pp("model").pp(layer_idx + 1),
            n_filters,
            channels,
            last_kernel_size,
            1,
            1,
            pad_mode,
            1,
            true,
        )?;

        Ok(Self {
            init_conv,
            layers,
            final_conv,
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        dev: &candle_core::Device,
    ) -> Result<SEANetDecoderState> {
        let init_conv_state = self.init_conv.init_state(batch_size, dev)?;
        let layer_states = self
            .layers
            .iter()
            .map(|l| {
                let up = l.upsample.init_state(batch_size, dev)?;
                let res = l
                    .residuals
                    .iter()
                    .map(|r| r.init_state(batch_size, dev))
                    .collect::<Result<Vec<_>>>()?;
                Ok((up, res))
            })
            .collect::<Result<Vec<_>>>()?;
        let final_conv_state = self.final_conv.init_state(batch_size, dev)?;
        Ok(SEANetDecoderState {
            init_conv_state,
            layer_states,
            final_conv_state,
        })
    }

    pub fn forward(&self, z: &Tensor, state: &mut SEANetDecoderState) -> Result<Tensor> {
        let mut z = self.init_conv.forward(z, &mut state.init_conv_state)?;
        for (layer, (up_state, res_states)) in self.layers.iter().zip(state.layer_states.iter_mut())
        {
            z = z.elu(1.0)?;
            z = layer.upsample.forward(&z, up_state)?;
            for (res, rs) in layer.residuals.iter().zip(res_states.iter_mut()) {
                z = res.forward(&z, rs)?;
            }
        }
        z = z.elu(1.0)?;
        self.final_conv.forward(&z, &mut state.final_conv_state)
    }
}
