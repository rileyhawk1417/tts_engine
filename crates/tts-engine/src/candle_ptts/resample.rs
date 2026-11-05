//! Channel-wise upsample — faithful port of oracle `resample.rs` (76 lines),
//! `ConvTrUpsample1d` only. The encoder's `ConvDownsample1d` is omitted: the
//! flow path never encodes (`mimi.encoder` is unused; voice latents arrive
//! pre-encoded). Depthwise (groups = dimension) convtranspose, k = 2·stride.

#![allow(dead_code)]

use candle_core::{Device, Result, Tensor};

use super::conv::{StreamingConvTr1dState, StreamingConvTranspose1d};
use super::nns::VB;

pub struct ConvTrUpsample1d {
    convtr: StreamingConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn load(vb: &VB, stride: usize, dimension: usize) -> Result<Self> {
        let convtr = StreamingConvTranspose1d::load(
            &vb.pp("convtr"),
            dimension,
            dimension,
            2 * stride,
            stride,
            dimension, // groups = dimension (depthwise)
            false,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize, dev: &Device) -> Result<StreamingConvTr1dState> {
        self.convtr.init_state(batch_size, dev)
    }

    pub fn forward(&self, x: &Tensor, state: &mut StreamingConvTr1dState) -> Result<Tensor> {
        self.convtr.forward(x, state)
    }
}
