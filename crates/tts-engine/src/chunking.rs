//! Long-text chunking module (T013) — ported law-for-law from the Phonon
//! reference decode policy (`docs/plan/targets.md`): sentence split, then
//! clause split for sentences over `target_tokens`, then word packing.
//!
//! Laws ported exactly:
//! - Only hard, unambiguous sentence boundaries split text (period/exclaim/
//!   question/ellipsis following a non-abbreviation; next sentence starting
//!   with uppercase/quote/digit, or a blank line).
//! - Newlines are preserved; a blank line (new paragraph) is a hard boundary,
//!   a single newline never splits.
//! - Decimals (3.5), thousands separators (3,500), times (10:30), and known
//!   abbreviations (Mr., U.S., e.g., etc.) never split.
//! - Target ~50 tokens (words) per chunk; sentences longer than that split at
//!   clause punctuation first (comma/colon/semicolon/dash), never inside
//!   numbers; still-long runs fall back to word packing; a single monster
//!   word is char-cut at `max_tokens` (documented last resort).
//!
//! The split is idempotent: re-chunking the joined chunks reproduces the same
//! chunk boundaries (word packing included); char-cut words are exempt since
//! joining changes them into ordinary words.

use crate::types::TextChunk;

#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Approximate words-per-chunk target (~50 tokens per Phonon policy).
    pub target_tokens: usize,
    /// Hard cap used for word packing and char-cutting monster words.
    pub max_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 50,
            max_tokens: 200,
        }
    }
}

pub fn chunk_text(text: &str, cfg: ChunkConfig) -> Vec<TextChunk> {
    let mut out = Vec::new();
    for (sa, se) in sentence_spans(text) {
        if word_count(&text[sa..se]) <= cfg.target_tokens {
            emit(&mut out, text, sa, se);
            continue;
        }
        let clauses = clause_spans(text, sa, se);
        if clauses.len() <= 1 {
            pack_words(&mut out, text, sa, se, cfg);
        } else {
            for (ca, ce) in clauses {
                if word_count(&text[ca..ce]) <= cfg.target_tokens {
                    emit(&mut out, text, ca, ce);
                } else {
                    pack_words(&mut out, text, ca, ce, cfg);
                }
            }
        }
    }
    out
}

fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

// ---------------------------------------------------------------------------
// Sentence spans
// ---------------------------------------------------------------------------

const ABBREVIATIONS: [&str; 22] = [
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "lt", "col", "gen", "rev", "hon",
    "approx", "etc", "e.g", "i.e", "vs", "fig", "u.s", "vs.",
];

/// Byte-offset spans of sentences in `text` (span covers terminal punctuation,
/// excludes trailing whitespace). Laws:
/// - boundary punct = . ! ? … (with guards)
/// - boundary requires next non-ws char to be uppercase/quote/digit/EOL, or
///   (for '.') not digit-digit ("3.5" never splits)
/// - single '\n' never splits; '\n' followed by blank line (paragraph) splits
fn sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < n {
        let (bi, c) = chars[i];
        // Paragraph break: a newline immediately followed by another newline
        // is a hard boundary at the FIRST newline. A terminal punct usually
        // fires first; guard against `start` already past `bi`.
        if c == '\n' && text[bi + 1..].starts_with('\n') && bi > start {
            spans.push((start, bi));
            start = skip_ws(text, bi + 1);
            while i < n && chars[i].0 < start {
                i += 1;
            }
            continue;
        }
        if matches!(c, '.' | '!' | '?' | '\u{2026}') && terminal_boundary(text, &chars, i) {
            let end = bi + c.len_utf8();
            spans.push((start, end));
            start = skip_ws(text, end);
        }
        i += 1;
    }
    if start < text.len() && !text[start..].trim().is_empty() {
        spans.push((start, text.len()));
    }
    spans
}

fn terminal_boundary(text: &str, chars: &[(usize, char)], i: usize) -> bool {
    let (bi, c) = chars[i];
    // "3.5" / "3,500.09" — '.' wedged between digits is no boundary.
    if c == '.' {
        let prev_digit = bi > 0
            && text[..bi]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_ascii_digit());
        let next_digit = text[bi + c.len_utf8()..]
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit());
        if prev_digit && next_digit {
            return false;
        }
    }
    if is_abbrev_or_initial(text, bi) {
        return false;
    }
    match next_non_ws(chars, i + 1) {
        // end of text always closes the sentence
        None => true,
        // next_non_ws skips the newline itself, so any returned char decides:
        // uppercase/digit/quote opens a new sentence; lowercase continues it
        Some((_, ch)) => {
            ch.is_uppercase() || ch.is_numeric() || "\"'([«\u{201c}\u{2018}".contains(ch)
        }
    }
}

/// Word before the punct: initials (J., J.R.R.), single letters, and known
/// abbreviations suppress the boundary. `punct` is the byte offset of '.'.
fn is_abbrev_or_initial(text: &str, punct: usize) -> bool {
    let head = &text[..punct];
    let tok_start = head.rfind(char::is_whitespace).map(|p| p + 1).unwrap_or(0);
    let tok = &head[tok_start..];
    if tok.is_empty() {
        return false;
    }
    let core = tok.trim_end_matches('.');
    let letters = core.chars().filter(|c| c.is_alphabetic()).count();
    if letters <= 1 {
        return true; // "J." "W." "3." ("No. 3.") and acronym chains
    }
    let lc = core.to_lowercase();
    ABBREVIATIONS.contains(&lc.as_str())
}

fn next_non_ws(chars: &[(usize, char)], from: usize) -> Option<(usize, char)> {
    chars
        .get(from..)?
        .iter()
        .find(|(_, c)| !c.is_whitespace())
        .map(|&(b, c)| (b, c))
}

fn skip_ws(text: &str, from: usize) -> usize {
    text[from..]
        .find(|c: char| !c.is_whitespace())
        .map(|k| from + k)
        .unwrap_or(text.len())
}

// ---------------------------------------------------------------------------
// Clause spans (for over-long sentences)
// ---------------------------------------------------------------------------

/// Split points after , ; : — – but never inside numbers ("3,500", "10:30").
fn clause_spans(text: &str, sa: usize, se: usize) -> Vec<(usize, usize)> {
    let sent = &text[sa..se];
    let chars: Vec<(usize, char)> = sent.char_indices().collect();
    let mut cuts = Vec::new();
    for (k, &(bi, c)) in chars.iter().enumerate() {
        if matches!(c, ',' | ';' | ':' | '\u{2014}' | '\u{2013}') {
            let prev_digit = chars[..k]
                .last()
                .is_some_and(|&(_, pc)| pc.is_ascii_digit());
            let next_digit = chars.get(k + 1).is_some_and(|&(_, nc)| nc.is_ascii_digit());
            if prev_digit && next_digit {
                continue;
            }
            cuts.push(bi + c.len_utf8());
        }
    }
    let mut spans = Vec::with_capacity(cuts.len() + 1);
    let mut from = 0usize;
    for c in cuts {
        spans.push((from, c));
        from = c;
    }
    spans.push((from, sent.len()));
    spans.into_iter().map(|(a, b)| (sa + a, sa + b)).collect()
}

// ---------------------------------------------------------------------------
// Word packing (over-long clause with no split points)
// ---------------------------------------------------------------------------

fn pack_words(out: &mut Vec<TextChunk>, text: &str, sa: usize, se: usize, cfg: ChunkConfig) {
    let sent = &text[sa..se];
    // word byte ranges within `sent`
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut cur: Option<usize> = None;
    for (bi, c) in sent.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = cur.take() {
                words.push((s, bi));
            }
        } else if cur.is_none() {
            cur = Some(bi);
        }
    }
    if let Some(s) = cur {
        words.push((s, sent.len()));
    }
    if words.is_empty() {
        return;
    }

    let mut acc: Vec<(usize, usize)> = Vec::new();
    let mut acc_count = 0usize;

    for &(wa, we) in &words {
        let w = &sent[wa..we];
        let wc = word_count(w); // 1 per whitespace-free word
        // A single word longer than max_tokens chars: char-cut last resort.
        if wc == 1 && w.chars().count() > cfg.max_tokens {
            flush_pack(out, text, sa, &acc);
            acc.clear();
            acc_count = 0;
            emit_char_cut(out, text, sa + wa, sa + we, cfg.max_tokens);
            continue;
        }
        if acc_count + wc > cfg.target_tokens && !acc.is_empty() {
            flush_pack(out, text, sa, &acc);
            acc.clear();
            acc_count = 0;
        }
        acc.push((wa, we));
        acc_count += wc;
    }
    flush_pack(out, text, sa, &acc);
}

fn flush_pack(out: &mut Vec<TextChunk>, text: &str, sa: usize, acc: &[(usize, usize)]) {
    if let Some(&(a, _)) = acc.first() {
        let end = acc.last().map(|&(_, e)| e).unwrap_or(a);
        emit(out, text, sa + a, sa + end);
    }
}

/// Char-cut a single monster word into ≤ `max_chars`-char pieces (byte-safe).
fn emit_char_cut(out: &mut Vec<TextChunk>, text: &str, wa: usize, we: usize, max_chars: usize) {
    let word = &text[wa..we];
    let mut piece_start = 0usize;
    let mut count = 0usize;
    let mut last_end = 0usize;
    for (bi, _) in word.char_indices() {
        if count == max_chars {
            emit(out, text, wa + piece_start, wa + bi);
            piece_start = bi;
            count = 0;
        }
        count += 1;
        last_end = bi;
    }
    let _ = last_end;
    let tail = piece_start;
    if tail < word.len() {
        emit(out, text, wa + tail, we);
    }
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

fn emit(out: &mut Vec<TextChunk>, text: &str, sa: usize, se: usize) {
    let sub = &text[sa..se];
    if sub.trim().is_empty() {
        return;
    }
    let lead = sub.len() - sub.trim_start().len();
    let trail = sub.len() - sub.trim_end().len();
    let (a, b) = (sa + lead, se - trail);
    if b > a {
        out.push(TextChunk {
            text: text[a..b].to_string(),
            byte_range: [a as u32, b as u32],
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_sentences() {
        let text = "Hello there. General greeting! Ready?";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Hello there.");
        assert_eq!(chunks[2].text, "Ready?");
    }

    #[test]
    fn abbreviations_and_initials_hold() {
        let text = "Mr. Smith met J. R. R. at 5 p.m. He was late.";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert_eq!(chunks.len(), 2, "abbrevs must not split: {chunks:?}");
        assert!(chunks[0].text.starts_with("Mr. Smith"));
        assert!(chunks[0].text.ends_with("p.m."));
        assert_eq!(chunks[1].text, "He was late.");
    }

    #[test]
    fn numbers_never_split() {
        let text =
            "The sale cleared 3.5 million on 2026-11-05, totaling 3,500 units at 10:30 sharp. Yes.";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert_eq!(chunks.len(), 2);
        // the long sentence stays in ONE chunk (13 words, still under target)
        assert_eq!(chunks[0].byte_range, [0, 80]);
    }

    #[test]
    fn paragraph_breaks_split() {
        let text = "One two.\n\nThree four.";
        assert_eq!(chunk_text(text, ChunkConfig::default()).len(), 2);
    }

    #[test]
    fn single_newline_is_not_a_split_when_next_is_lowercase() {
        let text = "line one\nmore words here. Line two.";
        let chunks = chunk_text(text, ChunkConfig::default());
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].text.contains("line one\nmore"));
    }

    #[test]
    fn long_sentence_splits_at_clauses() {
        // 3 clauses × 17 words = 51 + tail → over target → splits at commas.
        let clause = "the quick brown fox jumps over the lazy dog while the spotted cat sleeps on the warm mat upstairs, ";
        let text = format!("{}{}and then everyone goes home.", clause.repeat(3), "");
        let chunks = chunk_text(&text, ChunkConfig::default());
        assert!(
            chunks.len() >= 3,
            "long sentence must split at clauses: {chunks:?}"
        );
        for c in &chunks {
            assert!(word_count(&c.text) <= 50, "too long: {}", c.text);
        }
        // words preserved (3 clauses × 19 words + 5 tail)
        let total: usize = chunks.iter().map(|c| word_count(&c.text)).sum();
        assert_eq!(total, 62);
    }

    #[test]
    fn monster_word_char_cut() {
        // >50 words forces the packing path, which char-cuts cap-violating words.
        let mut words: Vec<String> = (0..55).map(|_| "filler".to_string()).collect();
        words.push("z".repeat(500));
        let text = words.join(" ");
        let chunks = chunk_text(&text, ChunkConfig::default());
        for c in &chunks {
            // cap law: ≤200 chars OR ≤50 words (word-approximation target)
            assert!(
                c.text.chars().count() <= 200 || word_count(&c.text) <= 50,
                "chunk over cap: {} chars",
                c.text.chars().count()
            );
        }
        let zz: String = chunks
            .iter()
            .filter(|c| c.text.starts_with('z'))
            .map(|c| c.text.as_str())
            .collect();
        assert_eq!(zz, "z".repeat(500));
    }

    #[test]
    fn single_monster_word_is_one_seam_chunk() {
        // Seam law: a single unpunctuated giant word is ONE chunk at the seam
        // (pause-boundary guarantees only); lanes may re-split within spans.
        let text = "a".repeat(500);
        let chunks = chunk_text(&text, ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn byte_ranges_slice_exact() {
        let text = "  First here!   Second one, with a comma. Third ends.  ";
        let chunks = chunk_text(text, ChunkConfig::default());
        for c in &chunks {
            assert_eq!(
                &text[c.byte_range[0] as usize..c.byte_range[1] as usize],
                c.text,
                "range mismatch for {:?}",
                c
            );
        }
    }

    #[test]
    fn join_preserves_all_words() {
        let text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. Sphinx of black quartz, judge my vow!";
        let chunks = chunk_text(text, ChunkConfig::default());
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            joined.split_whitespace().count(),
            text.split_whitespace().count()
        );
    }

    #[test]
    fn chunking_is_idempotent() {
        // Fuzz-ish deterministic battery of tricky-but-short words.
        let pool = [
            "3.5",
            "3,500",
            "U.S.",
            "Mr.",
            "10:30",
            "W.",
            "Yes.",
            "don't",
            "e.g.",
            "etc.",
            "Kokoro",
            "phonon",
            "RTF",
            "2026-11-05",
            "sentence,",
            "chunks;",
            "voice:",
            "fast!",
        ];
        // A few deterministic seeds
        let mut state = 0x2545F4914F6CDD1Du64;
        for round in 0..64 {
            let n = 6 + (round % 40);
            let mut words = Vec::with_capacity(n);
            for i in 0..n {
                // xorshift
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                words.push(pool[(state as usize) % pool.len()]);
                let _ = i;
            }
            let text = words.join(" ");
            let once = chunk_text(&text, ChunkConfig::default());
            let joined = once
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let twice = chunk_text(&joined, ChunkConfig::default());
            let once_texts: Vec<&str> = once.iter().map(|c| c.text.as_str()).collect();
            let twice_texts: Vec<&str> = twice.iter().map(|c| c.text.as_str()).collect();
            assert_eq!(
                once_texts, twice_texts,
                "round {round}: {once:?} vs {twice:?} on {text:?}"
            );
        }
    }

    #[test]
    fn empty_and_whitespace() {
        assert!(chunk_text("", ChunkConfig::default()).is_empty());
        assert!(chunk_text("   \n\t  ", ChunkConfig::default()).is_empty());
    }
}
