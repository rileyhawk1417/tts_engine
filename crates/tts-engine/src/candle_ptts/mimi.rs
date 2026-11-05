//! Mimi codec decoder — faithful port of oracle `mimi.rs` (247 lines) +
//! `dummy_quantizer.rs` (24). The encoder (`MimiEncoder`, its transformer and
//! downsample) is omitted: the flow path never encodes (`mimi.encoder` and
//! `mimi.downsample` are listed in `is_unused_by_tts_model`), voice latents
//! arrive pre-encoded as `speaker_wavs` tensors.
//!
//! Stream math (phonon-31m-en): latent frame 1/12.5 s → depthwise convtr
//! upsample ×16 (200 fps, hop 120) → decoder transformer (context 250,
//! d 512, 2 layers, layer_scale 0.01, erf gelu) → SEANet decoder ratios
//! 6·5·4 = 120 → 24 kHz PCM (frame = 1920 samples).

#![allow(dead_code)]

use candle_core::{Result, Tensor};

use super::attention::ProjectedTransformer;
use super::cfg::MimiConfig;
use super::conv::PadMode;
use super::nns::VB;
use super::resample::ConvTrUpsample1d;
use super::seanet::SEANetDecoder;

/// Simplified quantizer that only provides output projection for TTS
/// (oracle `dummy_quantizer.rs`).
pub struct DummyQuantizer {
    output_proj_weight: Tensor,
    output_proj_bias: Option<Tensor>,
}

impl DummyQuantizer {
    pub fn load(vb: &VB, dimension: usize, output_dimension: usize) -> Result<Self> {
        let vb = vb.pp("output_proj");
        let output_proj_weight = vb.get((output_dimension, dimension, 1), "weight")?;
        let output_proj_bias = if vb.contains_tensor("bias") {
            Some(vb.get(output_dimension, "bias")?)
        } else {
            None
        };
        Ok(Self {
            output_proj_weight,
            output_proj_bias,
        })
    }

    /// Forward pass: Conv1d with kernel_size=1, with the bias when the
    /// checkpoint has one. Input [B, dimension, T] -> Output [B, out_dim, T].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.output_proj_weight, 0, 1, 0, 1)?;
        match &self.output_proj_bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.elem_count(), 1))?),
            None => Ok(y),
        }
    }
}

#[derive(Clone)]
pub struct MimiDecoderState {
    decoder_state: super::seanet::SEANetDecoderState,
    decoder_transformer_state: super::attention::StreamingTransformerState,
    upsample_state: Option<super::conv::StreamingConvTr1dState>,
}

pub struct MimiDecoder {
    decoder: SEANetDecoder,
    decoder_transformer: ProjectedTransformer,
    upsample: Option<ConvTrUpsample1d>,
    pub quantizer: DummyQuantizer,
    pub sample_rate: usize,
    pub frame_rate: f64,
}

impl MimiDecoder {
    /// `MimiDecoder::load` (oracle mimi.rs:155). The gelu on the transformer
    /// layers is set by the caller via [`MimiDecoder::set_gelu`] (the oracle's
    /// `TTSModel::load` passes `cfg.mimi_with_gelu(&cfg.mimi)`).
    pub fn load(vb: &VB, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = PadMode::Constant;
        let decoder_dimension = cfg.decoder_dimension.unwrap_or(cfg.dimension);

        let decoder = SEANetDecoder::load(
            &vb.pp("decoder"),
            cfg.channels,
            decoder_dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let quantizer = DummyQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dimension,
            cfg.quantizer_output_dimension,
        )?;
        let output_dimensions = vec![decoder_dimension];
        let mut decoder_transformer = ProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;
        decoder_transformer
            .transformer
            .set_gelu(cfg.gelu.unwrap_or_default());
        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let upsample = if (encoder_frame_rate - cfg.frame_rate).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
            let us = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, cfg.dimension)?;
            Some(us)
        } else {
            None
        };
        Ok(Self {
            decoder,
            decoder_transformer,
            upsample,
            quantizer,
            sample_rate: cfg.sample_rate,
            frame_rate: cfg.frame_rate,
        })
    }

    pub fn set_gelu(&mut self, gelu: super::cfg::Gelu) {
        self.decoder_transformer.transformer.set_gelu(gelu);
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
        dev: &candle_core::Device,
    ) -> Result<MimiDecoderState> {
        let upsample_state = match &self.upsample {
            Some(us) => Some(us.init_state(batch_size, dev)?),
            None => None,
        };
        Ok(MimiDecoderState {
            decoder_state: self.decoder.init_state(batch_size, dev)?,
            decoder_transformer_state: self.decoder_transformer.init_state(
                batch_size,
                sequence_length,
                dev,
            )?,
            upsample_state,
        })
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    pub fn decode_from_latent_step(
        &self,
        latent: &Tensor,
        state: &mut MimiDecoderState,
    ) -> Result<Tensor> {
        // Upsample to encoder frame rate
        let emb = match (&self.upsample, &mut state.upsample_state) {
            (Some(us), Some(us_state)) => us.forward(latent, us_state)?,
            _ => latent.clone(),
        };

        let outs = self
            .decoder_transformer
            .forward(&emb, &mut state.decoder_transformer_state)?;
        self.decoder.forward(&outs[0], &mut state.decoder_state)
    }

    /// `quantizer.forward` (1×1 conv), as `decode_latent` applies it.
    pub fn quantize(&self, x: &Tensor) -> Result<Tensor> {
        self.quantizer.forward(x)
    }
}
