//! Generation policy: chunking, budgets, and EOS tail — faithful port of the
//! oracle's `plan.rs` (616 lines) plus `tts_model.rs`'s
//! `prepare_text_prompt` / `split_into_best_sentences` / `full_stop` helpers
//! (test code omitted).
//!
//! The one labeled delegation: text normalization is the oracle's own
//! `ptts::preprocess::Normalize` (2 028 lines of pure text logic), used as-is
//! so both lanes normalize identically by construction. Everything else here
//! is ported.

use ptts::preprocess::Normalize;

use super::tok::Tokenizer;

/// Extra KV-cache entries reserved on top of the text tokens and the generated
/// frames, covering the voice-prompt frames (~125 at 12.5Hz for a 10s prompt)
/// plus slack.
pub const PROMPT_SEQ_HEADROOM: usize = 512;

/// The most text tokens a chunk may have after [`fit`] cuts an oversized sentence, whatever
/// room a fixed budget leaves. Normal sentence grouping targets 50 tokens by default; that
/// target does not split a sentence on its own.
pub const MAX_FIT_TOKENS: usize = 200;

/// The most text tokens one chunk can have by default (`tts_model.rs`).
pub const MAX_TOKENS_PER_CHUNK: usize = 50;

/// Marks that always end a sentence (`tts_model.rs`).
pub const SENTENCE_ENDS: &[&str] = &[".", "!", "...", "?", "。", "！", "？", "।", "؟"];

/// Frames of audio to generate for a text prompt of `num_tokens` tokens.
pub fn frame_budget(num_tokens: usize, frame_rate: f64) -> usize {
    ((num_tokens as f64 / 3.0 + 2.0) * frame_rate).ceil() as usize
}

/// KV-cache length to allocate for a single text chunk: its text tokens, the
/// frames it may generate, and [`PROMPT_SEQ_HEADROOM`] for the voice prompt.
pub fn seq_budget(num_tokens: usize, frame_budget: usize) -> usize {
    num_tokens + PROMPT_SEQ_HEADROOM + frame_budget
}

/// Tracks the tail of a generation: once the model reports EOS, a few more
/// frames are still emitted so the codec can close out the utterance cleanly.
///
/// `frames_after_eos` comes from [`prepare_text_prompt`] — 3 for very short
/// prompts, 1 otherwise.
#[derive(Clone, Copy, Debug)]
pub struct EosPolicy {
    frames_after_eos: usize,
    countdown: Option<usize>,
}

impl EosPolicy {
    pub fn new(frames_after_eos: usize) -> Self {
        Self {
            frames_after_eos,
            countdown: None,
        }
    }

    /// Records the EOS flag of the frame that was just generated and returns
    /// whether generation should stop now.
    ///
    /// Call this once per frame, *after* the frame has been handed to the
    /// decoder — the EOS frame itself is part of the output.
    pub fn should_stop(&mut self, is_eos: bool) -> bool {
        if is_eos && self.countdown.is_none() {
            self.countdown = Some(self.frames_after_eos);
        }
        match self.countdown.as_mut() {
            None => false,
            Some(0) => true,
            Some(countdown) => {
                *countdown -= 1;
                false
            }
        }
    }

    /// True once the model has signalled EOS, whether or not the tail has run out.
    pub fn saw_eos(&self) -> bool {
        self.countdown.is_some()
    }
}

fn full_stop(c: char) -> char {
    match c as u32 {
        0x0900..=0x097F => '।',
        0x3040..=0x30FF
        | 0x3400..=0x9FFF
        | 0xF900..=0xFAFF
        | 0xFF66..=0xFF9F
        | 0x20000..=0x2FA1F => '。',
        _ => '.',
    }
}

/// Port of `tts_model::prepare_text_prompt`: trim, collapse whitespace,
/// capitalize, append a full stop, and pick the EOS tail length.
pub fn prepare_text_prompt(text: &str) -> (String, usize) {
    let text = text.trim().to_string();
    if text.is_empty() {
        return (text, 3);
    }
    let text = text.replace(['\n', '\r'], " ");
    let mut text: String = text.split_whitespace().collect::<Vec<_>>().join(" ");

    let number_of_words = text.split_whitespace().count();
    let frames_after_eos = if number_of_words <= 4 { 3 } else { 1 };
    let mut chars = text.chars();
    if let Some(first) = chars.next() {
        text = first.to_uppercase().to_string() + chars.as_str();
    }
    if let Some(last) = text.chars().last().filter(|c| c.is_alphanumeric()) {
        let stop = full_stop(last);
        // Text spaced one character at a time, as Japanese is, keeps its `。` spaced too.
        if stop == '。' && text.chars().rev().nth(1) == Some(' ') {
            text.push(' ');
        }
        text.push(stop);
    }
    (text, frames_after_eos)
}

/// Port of `tts_model::split_into_best_sentences` (Python-named): sentence
/// boundaries after EOS-mark tokens, greedy grouping under `max_tokens`.
fn split_into_best_sentences(
    tokenizer: &dyn Tokenizer,
    text: &str,
    max_tokens: Option<usize>,
) -> Result<Vec<String>, String> {
    let max_tokens = max_tokens.unwrap_or(MAX_TOKENS_PER_CHUNK);
    let (prepared, _) = prepare_text_prompt(text);
    let prepared = prepared.trim().to_string();
    let tokens = tokenizer.encode(&prepared)?;

    // A token ends a sentence when its piece is one of the marks, which the tokenizers keep as
    // pieces of their own, and which a tokenizer without a piece for a mark never produces.
    let mut pieces = std::collections::HashMap::new();
    let mut is_eos = |token: u32| -> Result<bool, String> {
        Ok(match pieces.get(&token) {
            Some(&eos) => eos,
            None => {
                let eos = SENTENCE_ENDS.contains(&tokenizer.decode(&[token])?.trim());
                pieces.insert(token, eos);
                eos
            }
        })
    };

    // Find sentence boundary indices: positions where a non-EOS token follows one or more EOS tokens.
    let mut sentence_boundaries = vec![0usize];
    let mut prev_was_eos = false;

    for (idx, &token) in tokens.iter().enumerate() {
        if is_eos(token)? {
            prev_was_eos = true;
        } else {
            if prev_was_eos {
                sentence_boundaries.push(idx);
            }
            prev_was_eos = false;
        }
    }
    sentence_boundaries.push(tokens.len());

    // Build (token_count, sentence_text) pairs by decoding each token sub-range.
    let mut sentences = Vec::new();
    for window in sentence_boundaries.windows(2) {
        let (start, end) = (window[0], window[1]);
        let text = tokenizer.decode(&tokens[start..end])?;
        sentences.push((end - start, text));
    }

    // Greedily group sentences into chunks that stay under max_tokens.
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_token_count = 0;

    for (nb_tokens, sentence) in sentences {
        if current_chunk.is_empty() {
            current_chunk = sentence;
            current_token_count = nb_tokens;
            continue;
        }

        if current_token_count + nb_tokens > max_tokens {
            chunks.push(current_chunk.trim().to_string());
            current_chunk = sentence;
            current_token_count = nb_tokens;
        } else {
            current_chunk.push(' ');
            current_chunk.push_str(&sentence);
            current_token_count += nb_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    Ok(chunks)
}

/// One piece of the text, ready to prompt the model with.
///
/// Built by [`chunks`] or [`Chunk::new`], so the budgets always match the tokens.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// The piece of normalized text, before [`prepare_text_prompt`].
    pub text: String,
    pub tokens: Vec<u32>,
    /// The tail [`EosPolicy`] runs after the model signals the end.
    pub frames_after_eos: usize,
    /// The most frames this chunk may generate, from [`frame_budget`].
    pub frame_budget: usize,
}

impl Chunk {
    /// Prepare and tokenize one piece of already normalized text.
    pub fn new(text: String, tokenizer: &dyn Tokenizer, frame_rate: f64) -> Result<Self, String> {
        let (prepared, frames_after_eos) = prepare_text_prompt(&text);
        let tokens = tokenizer.encode(&prepared)?;
        let frame_budget = frame_budget(tokens.len(), frame_rate);
        Ok(Self {
            text,
            tokens,
            frames_after_eos,
            frame_budget,
        })
    }

    /// The KV-cache length this chunk needs, from [`seq_budget`].
    pub fn seq_budget(&self) -> usize {
        seq_budget(self.tokens.len(), self.frame_budget)
    }
}

/// Split `text` into chunks the way every frontend does.
///
/// The whole input is normalized first, since that rewrites the characters the sentence
/// splitter looks for. Then whole sentences are grouped into chunks of up to `max_tokens`
/// tokens ([`split_into_best_sentences`]); a single sentence longer than that stays one chunk.
/// Each chunk is then prepared and tokenized on its own.
pub fn chunks(
    tokenizer: &dyn Tokenizer,
    text: &str,
    normalize: Normalize,
    max_tokens: usize,
    frame_rate: f64,
) -> Result<Vec<Chunk>, String> {
    let text = normalize.apply(text);
    let chunks = split_into_best_sentences(tokenizer, &text, Some(max_tokens))?
        .into_iter()
        .map(|text| Chunk::new(text, tokenizer, frame_rate))
        .collect::<Result<Vec<_>, String>>()?;
    if chunks.is_empty() {
        return Err("nothing to synthesize: the text is empty".to_string());
    }
    Ok(chunks)
}

/// Cut every chunk of more than `max` tokens until each piece fits, keeping their order.
pub fn fit(
    chunks: Vec<Chunk>,
    max: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>, String> {
    fit_with_word_limit(chunks, max, max, tokenizer, frame_rate)
}

/// Cut chunks toward `max` tokens while keeping a single word intact when it fits
/// `word_limit`. Use this when the model accepts a longer word than the preferred
/// chunk size. The result may contain single-word chunks over `max`, but never over
/// `word_limit` unless even one character is too large.
pub fn fit_with_word_limit(
    chunks: Vec<Chunk>,
    max: usize,
    word_limit: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>, String> {
    let word_limit = word_limit.max(max);
    let mut fitted = Vec::with_capacity(chunks.len());
    let mut todo: Vec<Chunk> = chunks.into_iter().rev().collect();
    while let Some(chunk) = todo.pop() {
        if chunk.tokens.len() <= max {
            fitted.push(chunk);
            continue;
        }
        let words: Vec<&str> = chunk.text.split_whitespace().collect();
        let halves = if words.len() >= 2 {
            let (a, b) = words.split_at(cut(&words, max.div_ceil(4), tokenizer)?);
            [a.join(" "), b.join(" ")]
        } else {
            if chunk.tokens.len() <= word_limit {
                fitted.push(chunk);
                continue;
            }
            let word = chunk.text.trim();
            match word.char_indices().nth(word.chars().count() / 2) {
                Some((mid, _)) if mid > 0 => [word[..mid].to_string(), word[mid..].to_string()],
                _ => {
                    fitted.push(chunk);
                    continue;
                }
            }
        };
        for half in halves.into_iter().rev() {
            todo.push(Chunk::new(half, tokenizer, frame_rate)?);
        }
    }
    Ok(fitted)
}

/// Cut chunks to `max` tokens and reject a piece that cannot be split further.
pub fn fit_or_error(
    chunks: Vec<Chunk>,
    max: usize,
    tokenizer: &dyn Tokenizer,
    frame_rate: f64,
) -> Result<Vec<Chunk>, String> {
    let chunks = fit(chunks, max, tokenizer, frame_rate)?;
    if let Some(chunk) = chunks.iter().find(|chunk| chunk.tokens.len() > max) {
        return Err(format!(
            "a text chunk is {} tokens and cannot fit the {max}-token limit; shorten or split it",
            chunk.tokens.len()
        ));
    }
    Ok(chunks)
}

/// Where to cut `words`, as the number of words before the cut: after the sentence end nearest
/// the middle in tokens that leaves `min_side` tokens on each side, else after such a clause mark,
/// else after the word nearest the middle. Takes at least two words, and always leaves at least
/// one on each side.
fn cut(words: &[&str], min_side: usize, tokenizer: &dyn Tokenizer) -> Result<usize, String> {
    // `ends[i]` is the number of tokens in `words[..=i]`.
    let mut ends = Vec::with_capacity(words.len());
    let mut total = 0;
    for word in words {
        total += tokenizer.encode(word)?.len();
        ends.push(total);
    }
    let inner = || ends[..words.len() - 1].iter().copied().enumerate();
    let imbalance = |&(_, end): &(usize, usize)| end.abs_diff(total - end);
    // A mark can sit inside closing quotes or brackets: `"yes,"`, `“yes,”`, `oui,»`.
    let closers = ['"', '\'', ')', ']', '”', '’', '»', '」', '』', '）'];
    let after = |marks: &[char]| {
        inner()
            .filter(|&(i, end)| {
                words[i].trim_end_matches(closers).ends_with(marks)
                    && end >= min_side
                    && total - end >= min_side
            })
            .min_by_key(imbalance)
    };
    let (i, _) = after(&['.', '!', '?', '。', '！', '？', '।', '؟'])
        .or_else(|| after(&[',', ';', ':', '、', '，', '；', '：', '،', '؛']))
        .or_else(|| inner().min_by_key(imbalance))
        .ok_or_else(|| "nothing to cut: fewer than two words".to_string())?;
    Ok(i + 1)
}

/// The most text tokens one chunk can have when `room` KV slots are left after the voice prompt:
/// the largest `n` with `n + frame_budget(n, frame_rate) <= room`.
pub fn max_tokens_for(room: usize, frame_rate: f64) -> usize {
    (0..=room)
        .rev()
        .find(|&n| n + frame_budget(n, frame_rate) <= room)
        .unwrap_or(0)
}
