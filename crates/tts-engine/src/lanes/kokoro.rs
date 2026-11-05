//! The Kokoro-82M v1.0 lane (`graph-ort` feature) — one ONNX graph over `ort`.
//!
//! Model family: `onnx-community/Kokoro-82M-v1.0-ONNX` (Apache-2.0, upstream
//! `hexgrad/Kokoro-82M`). The graph is StyleTTS2-shaped: text encoder + PLBert
//! style encoder + duration predictor + iSTFTNet vocoder fused into ONE graph,
//! outputting `waveform` f32 at 24 kHz.
//!
//! Contract (validated against the int8 export with a live round-trip probe):
//! - inputs: `input_ids` int64 `[1, seq]` (dynamic; older exports name it
//!   `tokens`), `style` f32 `[1, 256]`, `speed` f32 `[1]`
//! - output: `waveform` f32 `[1, N]`, 24 kHz mono
//! - tokens = `[0, *vocab_ids, 0]`, ids from the 114-symbol training vocab
//!   (kokoro-onnx `config.json`, matching `hexgrad/Kokoro-82M` KDATA VOCAB)
//! - style vector = voice pack row `min(len(tokens), 510) - 1` — the official
//!   KPipeline law (`pack[len(ps)-1]` in `kokoro/pipeline.py`); voice packs are
//!   raw f32 `[510, 256]` (the onnx-community `voices/*.bin` files)
//!
//! G2P: espeak-ng subprocess (the misaki espeak-fallback materialization),
//! with phonemizer-style punctuation restoration and exact word↔phoneme-word
//! alignment inside windows. A count-mismatched window (numbers, "u.s.",
//! contractions) falls back to per-word espeak calls — exact, but costs
//! ~12 ms spawn per word, so windows are batched 48 words per call.
//!
//! Stage-slot mapping (`docs/seam.md` §4, fixed slots):
//! - `g2p`: espeak calls, punctuation restore, vocab filter, phoneme batching
//! - `acoustic`: every ORT graph run. The iSTFTNet vocoder runs INSIDE the
//!   graph; without a duration tap it cannot be timed separately, so the
//!   graph wall is reported here, and the vocoder slot carries post-graph PCM
//!   assembly (concat + pause insertion). Honest split for a fused graph.
//! - `vocoder`: batch concat + pause insertion (see above)
//! - `io`: f32 → i16 quantize
//!
//! No trim: the onnx-community v1.0 exports have no `duration` output, so the
//! kokoro-onnx energy trim does not run; batch-end model silence stays in the
//! PCM.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::lanes::physical_core_estimate;
use crate::types::{
    ChunkOut, Determinism, LaneError, LoadInfo, ModelRef, RunMeta, StageTimings, SynthRequest,
    SynthStream, TtsLane, VoiceRef,
};

// ORT api/env helpers live in the lanes module root: ONE env per process
// (ort 2.0-rc law). The piper lane (T011) shares them.
use super::{ort, ort_env};

pub const SAMPLE_RATE: u32 = 24_000;
const MAX_PHONEME_LENGTH: usize = 510;
const VOICE_ROWS: usize = 510;
const VOICE_COLS: usize = 256;
const DEFAULT_VOICE: &str = "af_heart";
const ESPEAK_WINDOW: usize = 48;
const SENTENCE_PAUSE_S: f32 = 0.25;
const CLAUSE_PAUSE_S: f32 = 0.10;

/// The 114-symbol training vocab, from kokoro-onnx `config.json`
/// (`DEFAULT_VOCAB`), sorted by id. Unknown characters are dropped, not
/// rejected — the kokoro-onnx tokenize law.
static KOKORO_VOCAB: [(i64, char); 114] = [
    (1, ';'),
    (2, ':'),
    (3, ','),
    (4, '.'),
    (5, '!'), //
    (6, '?'),
    (9, '—'),
    (10, '…'),
    (11, '"'),
    (12, '('), //
    (13, ')'),
    (14, '“'),
    (15, '”'),
    (16, ' '),
    (17, '̃'), //
    (18, 'ʣ'),
    (19, 'ʥ'),
    (20, 'ʦ'),
    (21, 'ʨ'),
    (22, 'ᵝ'), //
    (23, 'ꭧ'),
    (24, 'A'),
    (25, 'I'),
    (31, 'O'),
    (33, 'Q'), //
    (35, 'S'),
    (36, 'T'),
    (39, 'W'),
    (41, 'Y'),
    (42, 'ᵊ'), //
    (43, 'a'),
    (44, 'b'),
    (45, 'c'),
    (46, 'd'),
    (47, 'e'), //
    (48, 'f'),
    (50, 'h'),
    (51, 'i'),
    (52, 'j'),
    (53, 'k'), //
    (54, 'l'),
    (55, 'm'),
    (56, 'n'),
    (57, 'o'),
    (58, 'p'), //
    (59, 'q'),
    (60, 'r'),
    (61, 's'),
    (62, 't'),
    (63, 'u'), //
    (64, 'v'),
    (65, 'w'),
    (66, 'x'),
    (67, 'y'),
    (68, 'z'), //
    (69, 'ɑ'),
    (70, 'ɐ'),
    (71, 'ɒ'),
    (72, 'æ'),
    (75, 'β'), //
    (76, 'ɔ'),
    (77, 'ɕ'),
    (78, 'ç'),
    (80, 'ɖ'),
    (81, 'ð'), //
    (82, 'ʤ'),
    (83, 'ə'),
    (85, 'ɚ'),
    (86, 'ɛ'),
    (87, 'ɜ'), //
    (90, 'ɟ'),
    (92, 'ɡ'),
    (99, 'ɥ'),
    (101, 'ɨ'),
    (102, 'ɪ'), //
    (103, 'ʝ'),
    (110, 'ɯ'),
    (111, 'ɰ'),
    (112, 'ŋ'),
    (113, 'ɳ'), //
    (114, 'ɲ'),
    (115, 'ɴ'),
    (116, 'ø'),
    (118, 'ɸ'),
    (119, 'θ'), //
    (120, 'œ'),
    (123, 'ɹ'),
    (125, 'ɾ'),
    (126, 'ɻ'),
    (128, 'ʁ'), //
    (129, 'ɽ'),
    (130, 'ʂ'),
    (131, 'ʃ'),
    (132, 'ʈ'),
    (133, 'ʧ'), //
    (135, 'ʊ'),
    (136, 'ʋ'),
    (138, 'ʌ'),
    (139, 'ɣ'),
    (140, 'ɤ'), //
    (142, 'χ'),
    (143, 'ʎ'),
    (147, 'ʒ'),
    (148, 'ʔ'),
    (156, 'ˈ'), //
    (157, 'ˌ'),
    (158, 'ː'),
    (162, 'ʰ'),
    (164, 'ʲ'),
    (169, '↓'), //
    (171, '→'),
    (172, '↗'),
    (173, '↘'),
    (177, 'ᵻ'),
];

fn char_to_id(c: char) -> Option<i64> {
    KOKORO_VOCAB
        .iter()
        .find(|&(_, ch)| *ch == c)
        .map(|&(id, _)| id)
}

/// Characters that split a word token (phonemizer-style punctuation set, plus
/// every non-alphanumeric symbol). Apostrophes are NOT separators —
/// contractions must stay whole for espeak.
fn is_separator(c: char) -> bool {
    !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}')
}

/// One word token with the punctuation that surrounds it, exactly as
/// phonemizer's `preserve_punctuation` restores marks around phonemes.
type Core = (String, String, String);

/// Split text into word tokens `(lead, core, trail)`. Whitespace never enters
/// tokens; punct runs between core and whitespace attach to the neighbouring
/// word (after a core → its `trail`; before the next core → that core's
/// `lead`). All-punct runs become `(lead, "", trail)` tokens so punctuation
/// survives even without a word around it.
fn split_cores(text: &str) -> Vec<Core> {
    let mut tokens: Vec<Core> = Vec::new();
    let mut lead = String::new();
    let mut core = String::new();
    let mut punct = String::new();

    for ch in text.chars() {
        if !is_separator(ch) {
            if !core.is_empty() && !punct.is_empty() {
                // a separator split two words: the punct closes the finished
                // word ("u.s." -> "u.")
                tokens.push((
                    std::mem::take(&mut lead),
                    std::mem::take(&mut core),
                    std::mem::take(&mut punct),
                ));
            }
            core.push(ch);
        } else if ch.is_whitespace() {
            if !core.is_empty() {
                tokens.push((
                    std::mem::take(&mut lead),
                    std::mem::take(&mut core),
                    std::mem::take(&mut punct),
                ));
            } else if !lead.is_empty() {
                // punct stranded before whitespace: punct-only token
                tokens.push((String::new(), String::new(), std::mem::take(&mut lead)));
            }
        } else if !core.is_empty() {
            punct.push(ch);
        } else {
            lead.push(ch);
        }
    }
    if !core.is_empty() {
        tokens.push((std::mem::take(&mut lead), core, punct));
    } else if !lead.is_empty() {
        tokens.push((String::new(), String::new(), lead));
    }
    tokens
}

fn is_sentence_mark(c: char) -> bool {
    c == '.' || c == '!' || c == '?' || c == '\u{2026}'
}

fn is_clause_mark(c: char) -> bool {
    c == ',' || c == ';' || c == ':'
}

/// Seconds of silence owed after a batch ending on this mark
/// (kokoro-onnx `pause_after`: sentence 0.25 s, clause 0.1 s).
fn pause_after_mark(batch: &str) -> f32 {
    match batch.trim_end().chars().next_back() {
        Some(c) if is_sentence_mark(c) => SENTENCE_PAUSE_S,
        Some(c) if is_clause_mark(c) => CLAUSE_PAUSE_S,
        _ => 0.0,
    }
}

/// Split AFTER mark characters (the mark stays with the left piece),
/// whitespace between pieces is consumed, pieces are trimmed and de-duped.
/// Mirrors `(?<=[marks])\s+` from kokoro-onnx `chunker.py`. Returns `None`
/// when the mark never appears (no split).
fn split_after_marks(ps: &str, pred: impl Fn(char) -> bool) -> Option<Vec<String>> {
    let chars: Vec<char> = ps.chars().collect();
    let mut bounds: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if pred(chars[i]) && i + 1 < chars.len() && chars[i + 1].is_whitespace() {
            bounds.push(i + 1);
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    if bounds.is_empty() {
        return None;
    }
    let mut pieces: Vec<String> = Vec::new();
    let mut prev = 0usize;
    for b in bounds {
        let piece: String = chars[prev..b].iter().collect();
        let piece = piece.trim().to_string();
        if !piece.is_empty() {
            pieces.push(piece);
        }
        prev = b;
        while prev < chars.len() && chars[prev].is_whitespace() {
            prev += 1;
        }
    }
    let last: String = chars[prev..].iter().collect();
    let last = last.trim().to_string();
    if !last.is_empty() {
        pieces.push(last);
    }
    if pieces.is_empty() {
        None
    } else {
        Some(pieces)
    }
}

/// kokoro-onnx `_atoms`: yield pieces no longer than `max_length`, cutting at
/// sentence marks, then clause marks, then whitespace; single unbroken runs
/// are sliced mid-string as the last resort (kokoro drops the tail; we keep
/// it — the seam chunks should never reach this path).
fn atoms(ps: &str, max_length: usize, out: &mut Vec<String>) {
    if ps.chars().count() <= max_length {
        if !ps.is_empty() {
            out.push(ps.to_string());
        }
        return;
    }
    // waterfall: sentence marks, then clause marks, then whitespace; every
    // sub-piece restarts from the top (a long mark-less sentence still gets
    // clause/ws splitting). No split -> char-slice fallback.
    type Splitter = fn(&str) -> Option<Vec<String>>;
    let levels: [Splitter; 3] = [
        |ps| split_after_marks(ps, is_sentence_mark),
        |ps| split_after_marks(ps, is_clause_mark),
        |ps| {
            let pieces: Vec<String> = ps.split_whitespace().map(str::to_string).collect();
            if pieces.len() > 1 { Some(pieces) } else { None }
        },
    ];
    for split in levels {
        if let Some(pieces) = split(ps)
            && pieces.len() > 1
        {
            for piece in pieces {
                atoms(&piece, max_length, out);
            }
            return;
        }
    }
    let chars: Vec<char> = ps.chars().collect();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + max_length).min(chars.len());
        out.push(chars[start..end].iter().collect());
        start = end;
    }
}

/// kokoro-onnx `_pack`: group consecutive atoms into batches within `limit`
/// (index ranges), counting one space join between atoms.
fn pack(lengths: &[usize], limit: usize) -> Vec<(usize, usize)> {
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let (mut start, mut size) = (0usize, 0usize);
    for (i, &len) in lengths.iter().enumerate() {
        let candidate = if i == start { len } else { size + 1 + len };
        if candidate > limit && i > start {
            batches.push((start, i));
            start = i;
            size = len;
        } else {
            size = candidate;
        }
    }
    if !lengths.is_empty() {
        batches.push((start, lengths.len()));
    }
    batches
}

/// kokoro-onnx `split_phonemes`: balanced batches — the smallest limit that
/// keeps the batch count of the plain 510-limit packing.
fn split_phonemes(ps: &str) -> Vec<String> {
    let mut at = Vec::new();
    atoms(ps.trim(), MAX_PHONEME_LENGTH, &mut at);
    if at.is_empty() {
        return Vec::new();
    }
    let lengths: Vec<usize> = at.iter().map(|s| s.chars().count()).collect();
    let fewest = pack(&lengths, MAX_PHONEME_LENGTH).len();
    let (mut lo, mut hi) = (*lengths.iter().max().unwrap(), MAX_PHONEME_LENGTH);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pack(&lengths, mid).len() <= fewest {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    pack(&lengths, lo)
        .into_iter()
        .map(|(s, e)| at[s..e].join(" "))
        .collect()
}

fn espeak_voice_for(
    lang: crate::types::Lang,
    options: &crate::types::LaneOptions,
) -> Result<String, LaneError> {
    if let Some(v) = options.get("g2p_voice") {
        return Ok(v.to_string());
    }
    match lang {
        crate::types::Lang::En => Ok("en-us".to_string()),
        crate::types::Lang::Fr => Ok("fr-fr".to_string()),
        crate::types::Lang::Es => Ok("es".to_string()),
        crate::types::Lang::Pt => Ok("pt-br".to_string()),
        // kokoro v1.0 shipped: a,b (en), es, fr, hi, it, pt-br, ja, zh — no de.
        crate::types::Lang::De => Err(LaneError::Synth(
            "lang 'de' is not covered by kokoro-82M v1.0 (a,b,es,fr,hi,it,pt-br,ja,zh only)".into(),
        )),
    }
}

/// One espeak-ng call. Output newlines collapse into spaces — the kokoro-onnx
/// join law (`" ".join(phonemes.split())`).
fn espeak_call(words: &[&str], voice: &str) -> Result<String, LaneError> {
    if words.is_empty() {
        return Ok(String::new());
    }
    let bin = std::env::var("TTS_ESPEAK").unwrap_or_else(|_| "espeak-ng".to_string());
    let mut child = Command::new(&bin)
        .args(["-q", "--ipa=3", "--tie=^", "-v", voice, "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            LaneError::Synth(format!(
                "espeak-ng spawn failed ({bin}): {e} — install espeak-ng or set TTS_ESPEAK"
            ))
        })?;
    child
        .stdin
        .as_mut()
        .expect("piped stdin")
        // espeak-ng clips the final phoneme without a trailing newline
        .write_all(format!("{}\n", words.join("\n")).as_bytes())
        .map_err(|e| LaneError::Synth(format!("espeak-ng stdin write: {e}")))?;
    drop(child.stdin.take()); // close stdin so espeak flushes and exits
    let out = child
        .wait_with_output()
        .map_err(|e| LaneError::Synth(format!("espeak-ng wait failed: {e}")))?;
    if !out.status.success() {
        return Err(LaneError::Synth(format!(
            "espeak-ng exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" "))
}

/// G2P for one text chunk → phoneme string (punctuation restored), with
/// per-window word↔phoneme-word alignment and exact per-word fallback.
/// The espeak stdin-lines-merge behavior makes batched per-word mapping
/// impossible; the phoneme-word count decides between aligned fast path and
/// per-word fallback (~12 ms per spawn).
fn phonemize_chunk(text: &str, voice: &str) -> Result<String, LaneError> {
    let cores = split_cores(text);
    let mut out = String::new();
    let mut idx = 0;
    while idx < cores.len() {
        let end = (idx + ESPEAK_WINDOW).min(cores.len());
        let window: &[(String, String, String)] = &cores[idx..end];
        let live: Vec<&str> = window
            .iter()
            .filter(|(_, c, _)| !c.is_empty())
            .map(|(_, c, _)| c.as_str())
            .collect();
        let raw = espeak_call(&live, voice)?;
        let phwords: Vec<&str> = raw.split_whitespace().collect();
        let aligned = live.is_empty() || phwords.len() == live.len();
        let mut pit = phwords.iter();
        for (lead, core, trail) in window {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(lead);
            if !core.is_empty() {
                if aligned {
                    out.push_str(pit.next().expect("count-aligned window"));
                } else {
                    out.push_str(&espeak_call(&[core.as_str()], voice)?);
                }
            }
            out.push_str(trail);
        }
        idx = end;
    }
    Ok(out.split_whitespace().collect::<Vec<&str>>().join(" "))
}

/// Filter the phoneme stream through the vocab (drop unknown chars — the
/// kokoro-onnx tokenize law) and return ids.
fn tokenize_phonemes(ps: &str) -> Vec<i64> {
    ps.chars().filter_map(char_to_id).collect()
}

/// Style-pack row index for a token list: `min(len, 510) - 1`, the official
/// KPipeline law (`pack[len(ps)-1]`). The +2 boundary tokens (`[0, *ps, 0]`)
/// do NOT count — the law reads the phoneme length, tokens here carry only
/// the phoneme ids pre-pad.
fn style_row_index(token_len: usize) -> usize {
    token_len.min(MAX_PHONEME_LENGTH).saturating_sub(1)
}

struct KokoroRuntime {
    session: Mutex<Session>,
    tokens_input: String,
}

pub struct KokoroLane {
    runtime: Option<Arc<KokoroRuntime>>,
    model_dir: Option<PathBuf>,
    voice: Option<Vec<f32>>,
    voice_label: String,
    model_label: String,
    model_sha: Option<String>,
    load_ms: u64,
}

impl KokoroLane {
    pub fn new() -> Self {
        Self {
            runtime: None,
            model_dir: None,
            voice: None,
            voice_label: String::new(),
            model_label: "unloaded".to_string(),
            model_sha: None,
            load_ms: 0,
        }
    }

    fn load_voice_file(path: &Path) -> Result<Vec<f32>, LaneError> {
        let bytes = std::fs::read(path)
            .map_err(|e| LaneError::Load(format!("voice file {}: {e}", path.display())))?;
        if bytes.len() != VOICE_ROWS * VOICE_COLS * 4 {
            return Err(LaneError::Load(format!(
                "voice file {}: expected {VOICE_ROWS}x{VOICE_COLS} raw f32 ({} bytes), got {} bytes — \
                 this lane reads the onnx-community voices/*.bin files only",
                path.display(),
                VOICE_ROWS * VOICE_COLS * 4,
                bytes.len()
            )));
        }
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn ensure_voice(&mut self, model_dir: &Path, voice: &VoiceRef) -> Result<(), LaneError> {
        let (path, label) = match voice {
            VoiceRef::Id(id) => (
                model_dir.join("voices").join(format!("{id}.bin")),
                id.clone(),
            ),
            VoiceRef::Path(p) => (
                p.clone(),
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string()),
            ),
            VoiceRef::Default => (
                model_dir
                    .join("voices")
                    .join(format!("{DEFAULT_VOICE}.bin")),
                DEFAULT_VOICE.to_string(),
            ),
        };
        if self.voice_label == label && self.voice.is_some() {
            return Ok(());
        }
        self.voice = Some(Self::load_voice_file(&path)?);
        self.voice_label = label;
        Ok(())
    }
}

impl Default for KokoroLane {
    fn default() -> Self {
        Self::new()
    }
}

impl TtsLane for KokoroLane {
    fn lane_name(&self) -> &'static str {
        "kokoro"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        let p = match model {
            ModelRef::Path(p) => p,
            // seam law: real lanes reject Pseudo; hub download is not in scope
            ModelRef::Hub { id, .. } => {
                return Err(LaneError::Load(format!(
                    "kokoro lane requires a disk path, got hub model '{id}' (download the \
                     onnx-community/Kokoro-82M-v1.0-ONNX artifacts and pass the .onnx path)"
                )));
            }
            ModelRef::Pseudo(id) => {
                return Err(LaneError::Load(format!(
                    "kokoro lane requires a disk ONNX graph, got pseudo model '{id}'"
                )));
            }
        };
        if !p.is_file() {
            return Err(LaneError::Load(format!(
                "model file not found: {} (pass e.g. models/kokoro-82m-v1.0-onnx/model_quantized.onnx)",
                p.display()
            )));
        }
        let env = ort_env()?;
        let intra: usize = std::env::var("TTS_ORT_INTRA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(physical_core_estimate);
        let session = Session::builder(env)
            .map_err(|e| LaneError::Load(format!("ort session builder: {e}")))?
            // GraphOptimizationLevel::All == ORT_ENABLE_ALL. rc.15's Level3
            // maps to ORT_ENABLE_LAYOUT, which the 1.22.1 runtime rejects
            // (te nemo.rs session-dials note).
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| LaneError::Load(format!("ort opt level: {e}")))?
            .with_intra_threads(intra)
            .map_err(|e| LaneError::Load(format!("ort intra threads: {e}")))?
            .with_inter_threads(1)
            .map_err(|e| LaneError::Load(format!("ort inter threads: {e}")))?
            .commit_from_file(p)
            .map_err(|e| LaneError::Load(format!("ort commit {}: {e}", p.display())))?;
        let names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let tokens_input = if names.iter().any(|n| n == "input_ids") {
            "input_ids".to_string()
        } else if names.iter().any(|n| n == "tokens") {
            "tokens".to_string()
        } else {
            return Err(LaneError::Load(format!(
                "graph {} has no 'input_ids'/'tokens' input (inputs: {names:?}) — \
                 this lane targets the onnx-community/Kokoro-82M-v1.0-ONNX export family",
                p.display()
            )));
        };
        for req_input in ["style", "speed"] {
            if !names.iter().any(|n| n == req_input) {
                return Err(LaneError::Load(format!(
                    "graph {} missing '{req_input}' input (inputs: {names:?})",
                    p.display()
                )));
            }
        }
        self.runtime = Some(Arc::new(KokoroRuntime {
            session: Mutex::new(session),
            tokens_input,
        }));
        self.model_dir = p.parent().map(|d| d.to_path_buf());
        self.model_label = p
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string());
        self.model_sha = crate::lanes::model_sha16(p);
        self.load_ms = t0.elapsed().as_millis() as u64;
        Ok(LoadInfo {
            load_ms: self.load_ms,
        })
    }

    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError> {
        if req.text.trim().is_empty() {
            return Err(LaneError::Synth("empty text".into()));
        }
        let runtime = self
            .runtime
            .clone()
            .ok_or_else(|| LaneError::Synth("lane not loaded".into()))?;
        let model_dir = self
            .model_dir
            .clone()
            .ok_or_else(|| LaneError::Synth("lane not loaded".into()))?;
        let tokens_input = runtime.tokens_input.clone();
        let espeak_voice = espeak_voice_for(req.lang, &req.lane_options)?;
        let speed: f32 = req
            .lane_options
            .get("speed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        if !(0.5..=2.0).contains(&speed) {
            return Err(LaneError::Synth(format!(
                "lane option 'speed' must be within 0.5..=2.0, got {speed}"
            )));
        }
        let voice_ref = req.voice.clone().unwrap_or(VoiceRef::Default);
        self.ensure_voice(&model_dir, &voice_ref)?;
        let voice = self.voice.clone().expect("just loaded");
        let voice_label = self.voice_label.clone();
        let voice_for_stream = voice;

        let chunks_in = if req.chunks.is_empty() {
            vec![crate::types::TextChunk {
                text: req.text.clone(),
                byte_range: [0, req.text.len() as u32],
            }]
        } else {
            req.chunks.clone()
        };

        let meta = RunMeta {
            lane: "kokoro",
            model: self.model_label.clone(),
            model_sha: self.model_sha.clone(),
            voice: voice_label,
            sample_rate: SAMPLE_RATE,
            lang: req.lang,
            determinism: Determinism::Deterministic,
            seed: req.seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "ort-dylib (onnxruntime 1.22.x)".to_string(),
            load_ms: self.load_ms,
            chunk_policy: "seam-provided".to_string(),
            restart_voice: false,
        };

        let iter = chunks_in.into_iter().enumerate().map(move |(idx, chunk)| {
            // ---- g2p: espeak, punctuation restore, vocab filter, batching ----
            let t_g = Instant::now();
            let phonemes = phonemize_chunk(&chunk.text, &espeak_voice)?;
            let batches = split_phonemes(&phonemes);
            let batch_tokens: Vec<Vec<i64>> =
                batches.iter().map(|b| tokenize_phonemes(b)).collect();
            if std::env::var("TTS_DEBUG_G2P").is_ok() {
                for (bi, b) in batches.iter().enumerate() {
                    eprintln!(
                        "[g2p] chunk {idx} batch {bi}: {:?} -> len(ps)={} len(ids)={}",
                        b,
                        b.chars().count(),
                        batch_tokens[bi].len()
                    );
                }
            }
            let g2p = t_g.elapsed();

            // ---- acoustic: the fused graph (encoder+PLBert+duration+iSTFT) ----
            let t_a = Instant::now();
            let mut batch_audio: Vec<(Vec<f32>, f32)> = Vec::new();
            let mut any_tokens = false;
            for tokens in &batch_tokens {
                if tokens.is_empty() {
                    batch_audio.push((Vec::new(), 0.0));
                    continue;
                }
                any_tokens = true;
                let mut input_ids = Vec::with_capacity(tokens.len() + 2);
                input_ids.push(0i64);
                input_ids.extend_from_slice(tokens);
                input_ids.push(0i64);
                let row = style_row_index(tokens.len());
                let style: Vec<f32> =
                    voice_for_stream[row * VOICE_COLS..(row + 1) * VOICE_COLS].to_vec();

                let mut sess = runtime
                    .session
                    .lock()
                    .map_err(|_| LaneError::Synth("kokoro session poisoned".into()))?;
                let token_t = ort(
                    Tensor::from_array(([1usize, input_ids.len()], input_ids)),
                    "input_ids tensor",
                )?;
                let style_t = ort(
                    Tensor::from_array(([1usize, VOICE_COLS], style)),
                    "style tensor",
                )?;
                let speed_t = ort(Tensor::from_array(([1usize], vec![speed])), "speed tensor")?;
                let outs = ort(
                    sess.run(ort::inputs![
                        tokens_input.as_str() => token_t,
                        "style" => style_t,
                        "speed" => speed_t,
                    ]),
                    "kokoro graph run",
                )?;
                let wave = outs.get("waveform").ok_or_else(|| {
                    LaneError::Synth("kokoro graph has no 'waveform' output".into())
                })?;
                let (_, data) = ort(wave.try_extract_tensor::<f32>(), "waveform extract")?;
                batch_audio.push((data.to_vec(), pause_after_mark(&batches[batch_audio.len()])));
                drop(outs);
                drop(sess); // release the session lock before PCM work
            }
            let acoustic = if any_tokens {
                t_a.elapsed()
            } else {
                std::time::Duration::ZERO
            };

            // ---- vocoder slot: concat + pause insertion ----
            let t_v = Instant::now();
            let mut pcm_f: Vec<f32> = Vec::new();
            for (samples, pause) in &batch_audio {
                pcm_f.extend_from_slice(samples);
                if !samples.is_empty() && *pause > 0.0 {
                    let n = (*pause * SAMPLE_RATE as f32).round() as usize;
                    pcm_f.resize(pcm_f.len() + n, 0.0);
                }
            }
            let vocoder = t_v.elapsed();

            // ---- io: f32 -> i16 ----
            let t_i = Instant::now();
            let pcm: Vec<i16> = if any_tokens {
                pcm_f
                    .iter()
                    .map(|&s| {
                        let v = (s.clamp(-1.0, 1.0) * 32767.0).round();
                        v as i16
                    })
                    .collect()
            } else {
                // law: empty token stream -> 10 ms of silence, one honest chunk
                vec![0i16; SAMPLE_RATE as usize / 100]
            };
            let io = t_i.elapsed();

            Ok(ChunkOut {
                index: idx as u32,
                stages: StageTimings {
                    g2p,
                    acoustic,
                    vocoder,
                    io,
                },
                pcm,
            })
        });

        Ok(SynthStream {
            meta,
            chunks: Box::new(iter),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn espeak_available() -> bool {
        Command::new("espeak-ng")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn vocab_lookup_samples() {
        assert_eq!(char_to_id('.'), Some(4));
        assert_eq!(char_to_id(' '), Some(16));
        assert_eq!(char_to_id('\u{2cc}'), Some(157));
        assert_eq!(char_to_id('^'), None); // espeak tie marker, dropped
        assert_eq!(char_to_id('x'), Some(66));
        assert_eq!(char_to_id('\u{2026}'), Some(10)); // ellipsis
    }

    #[test]
    fn split_cores_punctuation() {
        let t = split_cores("Hello, world!");
        assert_eq!(
            t,
            vec![
                ("".into(), "Hello".into(), ",".into()),
                ("".into(), "world".into(), "!".into()),
            ]
        );
        let q = split_cores("\u{201c}Hello world.");
        assert_eq!(
            q,
            vec![
                ("\u{201c}".into(), "Hello".into(), "".into()),
                ("".into(), "world".into(), ".".into()),
            ]
        );
    }

    #[test]
    fn split_cores_abbrev_and_numbers() {
        // "u.s. army" → core tokens "u","s","army" (periods restored between)
        let t = split_cores("u.s. army");
        let cores: Vec<&str> = t.iter().map(|(_, c, _)| c.as_str()).collect();
        assert_eq!(cores, vec!["u", "s", "army"]);
        // contractions stay whole
        assert_eq!(split_cores("isn't").len(), 1);
        assert_eq!(split_cores("isn't")[0].1, "isn't");
        // decimal splits at the dot, dot is restored in-stream
        let d = split_cores("3.45");
        assert_eq!(d[0].1, "3");
        assert_eq!(d[0].2, ".");
        assert_eq!(d[1].1, "45");
    }

    #[test]
    fn split_cores_leading_and_trailing() {
        // punct after whitespace before word → punct-only token between words
        let t = split_cores("Hello — world");
        assert_eq!(t[0].1, "Hello");
        assert_eq!(t[1].1, "");
        assert_eq!(t[1].2, "—");
        assert_eq!(t[2].1, "world");
        // trailing punct-only run
        let e = split_cores("ok …");
        assert_eq!(e[0].1, "ok");
        assert_eq!(e[1].1, "");
        assert_eq!(e[1].2, "…");
    }

    #[test]
    fn split_phonemes_short_and_empty() {
        assert!(split_phonemes("").is_empty());
        assert!(split_phonemes("   ").is_empty());
        assert_eq!(split_phonemes("həlˈo^ʊ"), vec!["həlˈo^ʊ".to_string()]);
    }

    #[test]
    fn split_phonemes_slices_long_single_atom() {
        // one 800-char atom → sliced into [510, 290]; joined batches ≤ 510
        let ps = "a".repeat(800);
        let batches = split_phonemes(&ps);
        assert_eq!(batches.len(), 2);
        assert!(batches[0].chars().count() <= MAX_PHONEME_LENGTH);
        assert!(batches[1].chars().count() <= MAX_PHONEME_LENGTH);
        // rejoin must reproduce the original atom (space-insert at cut)
        let joined = batches.join("");
        let stripped: String = joined.chars().filter(|&c| c != ' ').collect();
        assert_eq!(stripped, ps);
    }

    #[test]
    fn split_phonemes_balances_atoms() {
        // four 200-char atoms: balanced batches of 401 each (2 batches)
        let ps = vec!["x".repeat(200); 4].join(" ");
        let batches = split_phonemes(&ps);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].chars().count(), 401);
        assert_eq!(batches[1].chars().count(), 401);
    }

    #[test]
    fn sentence_marks_split_batches() {
        let ps = format!("{}. {}", "a".repeat(300), "b".repeat(300));
        let batches = split_phonemes(&ps);
        // atoms: [300a ".", 300 b] → both fit one 602-char batch? 300+1+300=601+mark... '.' kept with left piece
        // "…a"." has 300 a-chars + "."; joining: 301 + 1 + 300 = 602 > 510 → two batches
        assert_eq!(batches.len(), 2);
        assert!(batches[0].ends_with('.'));
        assert_eq!(pause_after_mark(&batches[0]), SENTENCE_PAUSE_S);
        assert_eq!(pause_after_mark(&batches[1]), 0.0);
    }

    #[test]
    fn pause_marks() {
        assert_eq!(pause_after_mark("world."), 0.25);
        assert_eq!(pause_after_mark("it,"), 0.10);
        assert_eq!(pause_after_mark("x"), 0.0);
        assert_eq!(pause_after_mark("why?"), 0.25);
    }

    #[test]
    fn tokenize_drops_unknown() {
        let ids = tokenize_phonemes("həlˈo^ʊ");
        assert_eq!(
            ids,
            vec![50, 83, 54, 156, 57, 135] // h ə l ˈ o ʊ ('^' tie dropped)
        );
    }

    #[test]
    fn style_row_law() {
        assert_eq!(style_row_index(1), 0);
        assert_eq!(style_row_index(82), 81);
        assert_eq!(style_row_index(510), 509);
        assert_eq!(style_row_index(700), 509); // over-limit tokens clamp
    }

    #[test]
    fn espeak_g2p_end_to_end() {
        if !espeak_available() {
            eprintln!("espeak-ng not installed; skipping G2P e2e");
            return;
        }
        let ps = phonemize_chunk("Hello, world.", "en-us").expect("g2p");
        // punctuation restored around the phonemes (loose asserts: espeak
        // version drift changes exact IPA, the shape must hold)
        assert!(ps.contains(','), "comma restored: {ps}");
        assert!(ps.ends_with('.'), "period restored: {ps}");
        assert!(ps.contains('h'), "phonemes present: {ps}");
        // espeak emits ^ tie markers inside diphthongs; the vocab filter drops
        // them at tokenization (they may be present in the phoneme string)
        // count-aligned path: two cores → two phoneme-words → commas inline
        let ids = tokenize_phonemes(&ps);
        assert!(!ids.is_empty());
        // per-word fallback: number window misaligns on purpose
        let ps_num = phonemize_chunk("Route 47.", "en-us").expect("g2p numbers");
        assert!(ps_num.ends_with('.'));
        assert!(ps_num.chars().count() > 8, "number expanded: {ps_num}");
    }
}
