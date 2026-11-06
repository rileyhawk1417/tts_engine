//! The synthesis driver for the candle port — a faithful port of the
//! ptts-0.6.0 oracle's CPU one-shot path (`synth.rs`: `SynthBuilder::build` →
//! `SynthOf::stream_with` → one-shot `SessionOf::stream_chunks` →
//! `run_inline`, with `Defaults::default` values), specialized to:
//!
//! - CPU + f32 weights (the candle lane's contract; oracle `Quant::F32`), and
//! - `Defaults.cfg_coef = None` (CFG off — the oracle's `SynthBuilder`
//!   default, which the flow-ptts lane also ran with), so the CFG state and
//!   `generate_step_cfg` stay dormant,
//! - an empty post-process chain (oracle default): the tail-flush sends
//!   nothing and each pull is the raw decoded frame.
//!
//! Structure is the oracle's one-shot semantics: `stream_with` plans the
//! chunks, builds one session sized to this text, and pulls frames lazily —
//! each pull is one flow-LM step plus one fused Mimi decode, emitting 1920
//! PCM samples per item, exactly like the ptts lane's `SpeechStream`.
//!
//! DEVIATION (labeled, T008 session-3): the oracle primes a session at the
//! voice-prompt length and then `grow`s the state's KV caches to `seq_budget`
//! (session reuse over several generations). The one-shot driver here builds
//! the state directly at `seq_budget`: the caches are zero-padded past the
//! filled prefix and attention reads only filled keys, so the math is
//! identical — only the temporary allocation pattern differs, and one-shot
//! has no primed cache to reuse.

use candle_core::Tensor;
use ptts::preprocess::Normalize;

use super::mimi::MimiDecoderState;
use super::model::{StepInput, TTSModel, TTSState};
use super::plan::{self, Chunk, EosPolicy};
use super::rng::{NormalRng, Rng};

/// Oracle `Defaults` (`SynthBuilder::new`): temperature 0.3, seed
/// 4242424242424242, CFG off, 50 tokens per chunk. The checkpoint's `temp`
/// config field does not feed these — the oracle keeps it in the unused extra
/// fields, and `Defaults::default` is the builder default.
pub const DEFAULT_TEMPERATURE: f32 = 0.3;
pub const DEFAULT_SEED: u64 = 4242424242424242;

/// A registered voice: name plus its prompt payload (`Voice` in the oracle's
/// `SynthOf`). `Emb` is the phonon-style `[1, T, d_model]` flow-LM
/// conditioning embedding; `Kv` is the pocket-tts pre-baked per-layer cache —
/// the voice lives in the attention state, not as a prefix embedding.
#[derive(Clone)]
pub struct Voice {
    pub name: String,
    pub payload: crate::candle_ptts::loader::LoadedVoice,
}

impl Voice {
    /// Prompt length in frames (the embedding's T or the cache end offset).
    pub fn frames(&self) -> Result<usize, String> {
        self.payload.frames().map_err(|e| e.to_string())
    }
}

/// Oracle `SpeechOptions`, narrowed to what the lane seam passes through.
#[derive(Debug, Clone, Default)]
pub struct SynthOptions<'a> {
    pub voice: Option<&'a str>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
}

/// Oracle `resolve` (synth.rs): fall back to `Defaults`, validate temperature.
fn resolve_options(opts: &SynthOptions<'_>) -> Result<(Option<String>, f32, u64), String> {
    let temperature = opts.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    // NaN spelled out rather than `!(t >= 0)`, which clippy rejects on
    // partially ordered types. Zero is fine: it is greedy sampling, and
    // `Normal::new` takes a zero std dev (oracle note verbatim).
    if temperature.is_nan() || temperature < 0.0 {
        return Err(format!("temperature must be >= 0, got {temperature}"));
    }
    Ok((
        opts.voice.map(str::to_string),
        temperature,
        opts.seed.unwrap_or(DEFAULT_SEED),
    ))
}

/// Oracle `voice_for` + `default_voice`: pick `name`, else the `default`
/// voice, else the first registered one; `None` when nothing is registered.
fn voice_for<'v>(voices: &'v [Voice], name: Option<&str>) -> Result<Option<&'v Voice>, String> {
    let default = if name.is_none() {
        voices
            .iter()
            .find(|v| v.name == "default")
            .map(|v| v.name.clone())
            .or_else(|| voices.first().map(|v| v.name.clone()))
    } else {
        None
    };
    let Some(name) = name.map(str::to_string).or(default) else {
        return Ok(None);
    };
    match voices.iter().find(|v| v.name == name) {
        Some(voice) => Ok(Some(voice)),
        None => {
            let mut known = voices.iter().map(|v| v.name.as_str()).collect::<Vec<_>>();
            known.sort();
            Err(format!(
                "unknown voice {name:?}; known voices: [{}]",
                known.join(", ")
            ))
        }
    }
}

/// Oracle `plan_chunks` (synth.rs, free fn): normalize, split into chunks, cut
/// any chunk longer than `fit_to` (refusing worse than that is an error).
fn plan_chunks(
    model: &TTSModel,
    frame_rate: f64,
    text: &str,
    max_tokens_per_chunk: usize,
    normalize: Normalize,
    fit_to: Option<usize>,
) -> Result<Vec<Chunk>, String> {
    let Some(tokenizer) = model.flow_lm.conditioner.tokenizer.as_ref() else {
        return Err(
            "this model was loaded without a tokenizer; pass one to load, or use the \
             lower-level TTSModel API with pre-tokenized input"
                .to_string(),
        );
    };
    let chunks = plan::chunks(
        tokenizer.as_ref(),
        text,
        normalize,
        max_tokens_per_chunk,
        frame_rate,
    )?;
    match fit_to {
        Some(max) => plan::fit_or_error(chunks, max, tokenizer.as_ref(), frame_rate)
            .map_err(|e| e.to_string()),
        None => Ok(chunks),
    }
}

/// The driving state behind a one-shot generation: an oracle `SessionOf` with
/// the two-thread pipeline inlined (the CPU path the oracle runs for f32/q8
/// quant weights — flow LM and Mimi decode on the caller's thread, no queues).
pub struct SynthDriver {
    model: std::sync::Arc<TTSModel>,
    /// The voice-conditioned (or no-voice) base state; cloned per chunk and
    /// never mutated after priming (oracle `base_state`).
    base_state: TTSState,
    /// The current chunk's running state (text prompt applied, frames appended).
    run_state: Option<TTSState>,
    rng: Box<dyn Rng + Send>,
    mimi_state: MimiDecoderState,
    chunks: std::collections::VecDeque<Chunk>,
    /// Current chunk, primed with its text prompt on first pull after swap.
    current: Option<Chunk>,
    /// Previous step's latent — `None` at every chunk start (step input Bos).
    prev: Option<Tensor>,
    eos: EosPolicy,
    frames_done: usize,
}

impl SynthDriver {
    /// Oracle `stream_with` → `session_at` → `stream_chunks` inlined: plan,
    /// size the budget, prime the state, and hand back the lazy driver.
    pub fn new(
        model: &std::sync::Arc<TTSModel>,
        voices: &[Voice],
        text: &str,
        opts: &SynthOptions<'_>,
        normalize: Normalize,
    ) -> Result<Self, String> {
        let (voice_name, temperature, seed) = resolve_options(opts)?;
        let frame_rate = model.mimi.frame_rate;
        let chunks = plan_chunks(
            model,
            frame_rate,
            text,
            plan::MAX_TOKENS_PER_CHUNK,
            normalize,
            Some(plan::MAX_FIT_TOKENS),
        )?;
        let voice = voice_for(voices, voice_name.as_deref())?.cloned();
        // One-shot budget: the voice prompt's real length, never less than
        // PROMPT_SEQ_HEADROOM — a reserve alone would leave a long prompt no
        // room (oracle synth.rs stream_with comment, verbatim).
        let prompt = match &voice {
            Some(voice) => voice.frames()?,
            None => 0,
        };
        let text_budget = chunks
            .iter()
            .map(|c| c.tokens.len() + c.frame_budget)
            .max()
            .unwrap_or(0);
        let seq_budget = text_budget + prompt.max(plan::PROMPT_SEQ_HEADROOM);
        if prompt > 0 && prompt >= seq_budget {
            // Oracle Error::VoicePromptTooLong — unreachable for derived
            // budgets, kept as the session's own guard.
            return Err(format!(
                "voice prompt of {prompt} frames does not fit the {seq_budget}-frame \
                 sequence budget"
            ));
        }

        // Oracle primed_state: base state conditioned on the voice (or the
        // no-voice prompt). Conditioner defaults are already summed into the
        // model's condition providers at load, so the voice's `conditions`
        // slot stays `None` here, exactly as the oracle leaves it for a
        // checkpoint-voice with none.
        let mut base = model
            .init_flow_lm_state(1, seq_budget)
            .map_err(|e| e.to_string())?;
        prompt_base(model, &mut base, &voice)?;
        let rng = Box::new(NormalRng::new(temperature, seed)?);
        Ok(Self {
            model: model.clone(),
            base_state: base,
            run_state: None,
            rng,
            mimi_state: model
                .init_mimi_state(1)
                .map_err(|e| format!("mimi state: {e}"))?,
            chunks: chunks.into(),
            current: None,
            prev: None,
            eos: EosPolicy::new(0),
            frames_done: 0,
        })
    }

    /// Oracle `run_backbone` per-chunk setup, inlined into the pull loop:
    /// fresh state from the base, its text prompt run through the backbone.
    fn prime_current_chunk(&mut self) -> Result<(), String> {
        let Some(chunk) = self.chunks.pop_front() else {
            return Err("no chunk left to prime".to_string());
        };
        self.eos = EosPolicy::new(chunk.frames_after_eos);
        self.frames_done = 0;
        self.prev = None;
        let mut state = self.base_state.clone();
        self.model
            .prompt_text(&mut state, &chunk.tokens)
            .map_err(|e| format!("chunk text prompt: {e}"))?;
        self.run_state = Some(state);
        self.current = Some(chunk);
        Ok(())
    }
}

impl std::iter::Iterator for SynthDriver {
    /// One flow-LM step + fused Mimi decode = one frame of audio (1920 PCM
    /// samples at phonon's rates). Mirrors oracle `run_inline`'s per-frame
    /// emission: the EOS frame is emitted before the chunk stops.
    type Item = Result<Vec<f32>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current.is_none() {
                if self.chunks.is_empty() {
                    return None;
                }
                if let Err(e) = self.prime_current_chunk() {
                    return Some(Err(e));
                }
                continue;
            }
            // Step input: previous latent, or Bos at the sequence start.
            let input = match self.prev.as_ref() {
                Some(prev) => StepInput::Latent(prev),
                None => StepInput::Bos { batch: 1 },
            };
            let Some(chunk) = self.current.clone() else {
                unreachable!("current checked non-empty above");
            };
            let run_state = self.run_state.as_mut().expect("primed with the chunk");
            match self.model.generate_step(run_state, input, &mut self.rng) {
                Ok((next, is_eos)) => {
                    let audio = match self.model.decode_latent(&next, &mut self.mimi_state) {
                        Ok(audio) => audio,
                        Err(e) => return Some(Err(format!("mimi decode: {e}"))),
                    };
                    self.frames_done += 1;
                    // EOS is honored only once `min_frames_before_eos` frames
                    // exist (pocket fork constant 6; phonon = 0 so the frame
                    // counter has passed it after the first step — unchanged
                    // behavior for the phonon family).
                    let stop = self.frames_done > self.model.min_frames_before_eos
                        && self.eos.should_stop(is_eos);
                    if stop || self.frames_done >= chunk.frame_budget {
                        // Chunk done — `current` stays taken; the next pull
                        // primes the next chunk. prev resets with it.
                        self.current = None;
                        self.prev = None;
                    } else {
                        self.prev = Some(next);
                    }
                    return Some(to_pcm(&audio));
                }
                Err(e) => return Some(Err(format!("flow step: {e}"))),
            }
        }
    }
}

/// Oracle `primed_state`: apply the voice conditioning (or the no-voice
/// prompt) to a fresh state. Pocket-tts voices are pre-baked attention caches:
/// the base state imports them instead of running a voice-prompt backbone pass.
fn prompt_base(model: &TTSModel, base: &mut TTSState, voice: &Option<Voice>) -> Result<(), String> {
    match voice {
        Some(voice) => match &voice.payload {
            crate::candle_ptts::loader::LoadedVoice::Emb(emb) => model
                .prompt_audio(base, emb)
                .map_err(|e| format!("voice prompt: {e}")),
            crate::candle_ptts::loader::LoadedVoice::Kv(layers) => model
                .import_voice_kv(base, layers)
                .map_err(|e| format!("voice cache import: {e}")),
        },
        None => model
            .prompt_no_voice(base)
            .map_err(|e| format!("no-voice prompt: {e}")),
    }
}

/// Oracle `run_inline`: the decoder output `[1, 1, samples]` becomes host f32.
fn to_pcm(audio: &Tensor) -> Result<Vec<f32>, String> {
    let dims = audio
        .dims3()
        .map_err(|e| format!("decode output dims: {e}"))?;
    match dims {
        (1, 1, _) => audio
            .squeeze(0)
            .and_then(|a| a.squeeze(0))
            .and_then(|a| a.to_vec1())
            .map_err(|e| format!("decode output to host f32: {e}")),
        _ => Err(format!(
            "unexpected decode output shape {dims:?}, expected [1, 1, samples]"
        )),
    }
}
