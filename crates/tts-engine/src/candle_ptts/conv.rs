//! Streaming 1D convolutions — faithful port of oracle `conv.rs` (245 lines) to
//! candle 0.11. Every numeric path mirrors the oracle; only the backend differs.
//!
//! Candle's `conv1d`/`conv_transpose1d` carry no bias parameter, so the bias is
//! broadcast-added after the kernel op (oracle folds it in via `xn`; same math).
//! `slice_assign` on candle takes an array of `Range`s (one per dim).

#![allow(clippy::wildcard_imports)]
#![allow(dead_code)]

use candle_core::{DType, Device, Result, Tensor};

use super::nns::VB;

/// Pad input so that a convolution covers the full input. Dim 2 is time.
pub fn pad_for_conv1d(x: &Tensor, kernel_size: usize, stride: usize) -> Result<Tensor> {
    let length = x.dim(2)?;
    let n_frames = (length as f64 - kernel_size as f64) / stride as f64 + 1.0;
    let ideal_length = (n_frames.ceil() as usize - 1) * stride + kernel_size;
    let extra = ideal_length.saturating_sub(length);
    if extra > 0 {
        x.pad_with_zeros(2, 0, extra)
    } else {
        Ok(x.clone())
    }
}

/// Conv1d wrapper with weight and optional bias (oracle `conv.rs` `Conv1d`).
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    pub stride: usize,
    pub dilation: usize,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub groups: usize,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels / groups, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            dilation,
            kernel_size,
            in_channels,
            groups,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.weight, 0, self.stride, self.dilation, self.groups)?;
        match &self.bias {
            // [out] -> [1, out, 1]
            Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?),
            None => Ok(y),
        }
    }
}

/// ConvTranspose1d wrapper (oracle `conv.rs` `ConvTranspose1d`).
pub struct ConvTranspose1d {
    weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub kernel_size: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl ConvTranspose1d {
    pub fn load(
        vb: &VB,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.get((in_channels, out_channels / groups, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            kernel_size,
            out_channels,
            groups,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, self.groups)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?),
            None => Ok(y),
        }
    }
}

/// Streaming state for [`StreamingConv1d`].
#[derive(Debug, Clone)]
pub struct StreamingConv1dState {
    pub previous: Tensor,
    pub first: bool,
}

#[derive(Clone, Copy)]
pub enum PadMode {
    Constant,
    Replicate,
}

/// Streaming Conv1d with causal padding (oracle `conv.rs` `StreamingConv1d`).
pub struct StreamingConv1d {
    pub conv: Conv1d,
    pad_mode: PadMode,
}

impl StreamingConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VB,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = Conv1d::load(
            &vb.pp("conv"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
        )?;
        Ok(Self { conv, pad_mode })
    }

    fn effective_kernel_size(&self) -> usize {
        (self.conv.kernel_size - 1) * self.conv.dilation + 1
    }

    pub fn init_state(&self, batch_size: usize, dev: &Device) -> Result<StreamingConv1dState> {
        let kernel = self.effective_kernel_size();
        let prev_len = kernel - self.conv.stride;
        let previous = Tensor::zeros(
            (batch_size, self.conv.in_channels, prev_len),
            DType::F32,
            dev,
        )?;
        Ok(StreamingConv1dState {
            previous,
            first: true,
        })
    }

    pub fn forward(&self, x: &Tensor, state: &mut StreamingConv1dState) -> Result<Tensor> {
        let tp = state.previous.dim(2)?;

        // On first call with replicate padding, fill previous with first sample
        if tp > 0 && matches!(self.pad_mode, PadMode::Replicate) && state.first {
            let init = x.narrow(2, 0, 1)?.contiguous()?;
            state.previous = if tp == 1 {
                init
            } else {
                let refs: Vec<&Tensor> = (0..tp).map(|_| &init).collect();
                Tensor::cat(&refs, 2)?
            };
        }

        // Prepend previous state
        let x = if tp > 0 {
            Tensor::cat(&[&state.previous, x], 2)?
        } else {
            x.clone()
        };

        // Run convolution
        let y = self.conv.forward(&x)?;

        // Update state
        if tp > 0 {
            let xlen = x.dim(2)?;
            state.previous = x.narrow(2, xlen - tp, tp)?.contiguous()?;
            if matches!(self.pad_mode, PadMode::Replicate) {
                state.first = false;
            }
        }

        Ok(y)
    }
}

/// Streaming state for [`StreamingConvTranspose1d`].
#[derive(Debug, Clone)]
pub struct StreamingConvTr1dState {
    pub partial: Tensor,
}

/// Streaming ConvTranspose1d (oracle `conv.rs` `StreamingConvTranspose1d`).
///
/// The overlap-add uses bias-adjusted tails: candle's kernel op has no bias, so
/// the returned `y` here is already biased, and the oracle's subtract-bias-from
/// -partial step applies to a tail of the biased output — subtracting then
/// re-adding on the next call cancels, matching the oracle exactly.
pub struct StreamingConvTranspose1d {
    pub convtr: ConvTranspose1d,
}

impl StreamingConvTranspose1d {
    pub fn load(
        vb: &VB,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let convtr = ConvTranspose1d::load(
            &vb.pp("convtr"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize, dev: &Device) -> Result<StreamingConvTr1dState> {
        let pt = self.convtr.kernel_size - self.convtr.stride;
        let partial = Tensor::zeros((batch_size, self.convtr.out_channels, pt), DType::F32, dev)?;
        Ok(StreamingConvTr1dState { partial })
    }

    pub fn forward(&self, x: &Tensor, state: &mut StreamingConvTr1dState) -> Result<Tensor> {
        let mut y = self.convtr.forward(x)?;
        let pt = state.partial.dim(2)?;

        if pt > 0 {
            // Add overlap from previous
            let y_start = y.narrow(2, 0, pt)?.contiguous()?;
            let y_start = y_start.broadcast_add(&state.partial)?;
            y = y.slice_assign(&[0..y.dim(0)?, 0..y.dim(1)?, 0..pt], &y_start)?;

            // Save new partial (subtract bias if present)
            let y_len = y.dim(2)?;
            let mut for_partial = y.narrow(2, y_len - pt, pt)?.contiguous()?;
            if let Some(bias) = &self.convtr.bias {
                let bias = bias.reshape((1, bias.elem_count(), 1))?;
                for_partial = for_partial.broadcast_sub(&bias)?;
            }
            state.partial = for_partial;

            // Return without the partial tail
            y = y.narrow(2, 0, y_len - pt)?.contiguous()?;
        }

        Ok(y)
    }
}
