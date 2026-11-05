//! The Piper VITS lane (`piper` feature) — fused VITS ONNX voices over
//! `ort`, with caller-supplied seeded noise via graph surgery (T011
//! session-1).
//!
//! Model family: piper fused VITS voices. Drop-dir ground truth (T011
//! session-1): amy `en_US-amy-medium` (piper_version 1.0.0, 63 MB, 256
//! symbols) and ryan `en_US-ryan-high` (piper_version 0.2.0 legacy, 121 MB,
//! 130 symbols). Both graphs are FUSED VITS — text encoder + duration
//! predictor + flow + decoder (no vocoder boundary; audio straight out at
//! 22050 Hz, no mel intermediate).
//!
//! Contract (validated against the patched graphs in T011 session-1; amy 3×
//! byte-identical `021686cc…`, ryan 3× `f48f731e…`):
//! - inputs: `input` int64 `[1, N]`, `input_lengths` int64 `[1]`, `scales`
//!   f32 `[3]` = [noise_scale, length_scale, noise_w], `noise_0` f32
//!   `[1, 2, N]` (duration noise, caller-exact T_text), `noise_1` f32
//!   `[1, 192, BUDGET]` (spectral noise; the graph Slices to the actual
//!   T_mel — buffer extent is caller-chosen, 2048 frames ≈ 23.7 s here)
//! - output: `output` f32 `(1, 1, 1, N)`, 22050 Hz mono
//! - stock piper graphs draw noise internally via two unseeded
//!   `RandomNormalLike` ops (no determinism). This lane REQUIRES the
//!   patched graphs from `eval/patch_piper_rng.py` (`*-rng.onnx`), which
//!   replace those ops with `Slice(noise_k, ...)` so `--seed` stays a live
//!   structural dial (flow-candle ReplayRng architecture, strictly
//!   friendlier than both stock piper and the frozen-seed streaming
//!   lineage).
//!
//! G2P: `espeak-rs` 0.2 — in-process vendored espeak-ng 1.52.0.1 (GPLv3
//! linking, same class as the kokoro espeak question; documented in
//! `docs/notes/piper.md`). `text_to_phonemes(text, voice, None)` returns
//! ONE IPA string per sentence (clauses accumulated, punctuation
//! embedded); no separator (piper codepoint-map law). Data dir from
//! `PIPER_ESPEAKNG_DATA_DIRECTORY` (parent of `espeak-ng-data`; system
//! `/usr/share` matches the vendored 1.52.0.1 lib). G2P is NOT a perf
//! surface — 0.03% of the VITS wall (T011 session-1.5 measurement).
//!
//! Phoneme-id assembly, ported VERBATIM from piper source (both law
//! lineages, per-codepoint — both drop-dir voice maps are 100%
//! single-codepoint keys, verified; no diphthong keys, `ɪ`/`ʊ` etc. map
//! individually):
//! - piper_version ≥ 1.0.0 (piper1-gpl `src/piper/phoneme_ids.py`):
//!   `[BOS, PAD] ++ (map[ph] ++ [PAD])* ++ [EOS]`, missing phoneme →
//!   warn + skip
//! - legacy 0.2.0 (rhasspy/piper `python_run/piper/voice.py`):
//!   `[BOS] ++ (map[ph] ++ [PAD])* ++ [EOS]`
//! - `phoneme_type` null/missing → espeak (legacy `Config` default —
//!   upstream would crash on ryan's explicit JSON null; this lane is
//!   strictly more permissive, labeled deviation). `text` → NOT shipped in
//!   phase A (needs NFD codepoints = unicode-normalization); pinyin /
//!   hebrew / japanese / thai / lithuanian → Load error.
//!
//! io encode law, per format, verbatim from piper source:
//! - 1.0.0 (`piper1-gpl src/piper/voice.py`): peak-normalize (max|a| <
//!   1e-8 → zeros; normalize_audio default ON) → clip [-1, 1] → `× 32767.0`
//!   → clip ±32767.0 → trunc → i16
//! - legacy (`rhasspy/piper python_run/piper/util.py`): `a ×
//!   32767/max(0.01, max|a|)` → clip ±32767.0 → trunc → i16
//!
//! (Trunc, not round: numpy `.astype(int16)` is a C cast; the flow-candle
//! lane's `round` is the ptts oracle law, a different lane.)
//!
//! Scales: voice JSON `inference` `{noise_scale, length_scale, noise_w}`;
//! upstream defaults 0.667 / 1.0 / 0.8 when absent (both Config classes).
//! Overridable per run via lane options `noise-scale` / `length-scale` /
//! `noise-w` (the piper CLI flags).
//!
//! Stage-slot mapping (`docs/seam.md` §4, fixed slots):
//! - `g2p`: espeak-rs calls + per-codepoint id assembly (both law tables)
//! - `acoustic`: every ORT graph run (fused VITS — this is the whole
//!   synthesis graph)
//! - `vocoder`: ZERO — the vocoder is fused into the graph (honest split,
//!   same convention as flow-candle's fused decode)
//! - `io`: the encode law above (f32 → i16)
//!
//! Multi-speaker voices: `num_speakers > 1` feeds `sid` from
//! `speaker_id_map` (VoiceRef::Id as speaker name) else
//! `default_speaker_id` (0) — the upstream speaker laws. Drop-dir voices
//! are single-speaker.
//!
//! Determinism: `Seeded` under `--seed` (rand StdRng; eps ~ N(0,1)
//! std-normal — the patched graph Slices a `RandomNormalLike(std=1)`
//! replacement, scales wiring untouched; per-sentence draw order
//! noise_0 (2·N) then noise_1 (192·budget)); unseeded runs report
//! `NonDeterministic` like the flow lanes (`flow_ptts.rs` law).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use crate::types::{
    ChunkOut, Determinism, LaneError, LoadInfo, ModelRef, RunMeta, StageTimings, SynthRequest,
    SynthStream, TextChunk, VoiceRef,
};

/// Spectral-noise buffer extent (mel frames). The patched graph clips this
/// to the runtime T_mel via Slice; 2048 frames ≈ 23.7 s @ 22050 Hz / hop
/// 256 — beyond any T013 chunk. Pinned to the T011 session-1 acceptance
/// shapes.
const NOISE_1_BUDGET_FRAMES: usize = 2048;
/// Mel bins in both drop-dir voices (session-1 ground truth).
const MEL_BINS: usize = 192;
/// piper `_MAX_WAV_VALUE` (piper1-gpl `src/piper/voice.py` — 32767.0,
/// NOT 32768).
const MAX_WAV_VALUE: f32 = 32767.0;
/// Legacy peak-normalize floor (`audio_float_to_int16`: max(0.01, …)).
const LEGACY_NORM_FLOOR: f32 = 0.01;
/// 1.0.0 empty-audio threshold (`synthesize`: max_val < 1e-8 → zeros).
const EMPTY_AUDIO_THRESHOLD: f32 = 1e-8;

const REQUIRED_INPUTS: [&str; 5] = ["input", "input_lengths", "scales", "noise_0", "noise_1"];

// ---------------------------------------------------------------------------
// Voice JSON (piper config) parsing — defaults verbatim from upstream
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawVoiceCfg {
    #[serde(default)]
    piper_version: Option<String>,
    audio: Option<RawAudio>,
    #[serde(default)]
    inference: Option<RawInference>,
    espeak: Option<RawEspeak>,
    #[serde(default)]
    // Parsed-and-ignored (the tolerant-parse deviation): an explicit JSON
    // `null` is a hard ValueError in upstream legacy 0.2.0 and we absorb it;
    // law dispatch is by `piper_version`, so the value itself is dead here.
    #[allow(dead_code)]
    phoneme_type: Option<String>,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    #[serde(default)]
    num_speakers: Option<u64>,
    #[serde(default)]
    speaker_id_map: BTreeMap<String, i64>,
    #[serde(default)]
    default_speaker_id: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct RawAudio {
    sample_rate: u32,
}

#[derive(Debug, Deserialize)]
struct RawInference {
    #[serde(default)]
    noise_scale: Option<f32>,
    #[serde(default)]
    length_scale: Option<f32>,
    #[serde(default)]
    noise_w: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawEspeak {
    voice: String,
}

/// Pinned law triple: `(BOS ids, EOS ids, PAD ids)` from the voice json's
/// `phoneme_id_map` (`law_ids()`), threaded through chunk assembly.
pub(crate) type LawTriple = (Vec<i64>, Vec<i64>, Vec<i64>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IdLaw {
    /// piper_version ≥ 1.0.0 — `[BOS, PAD] ++ (map[ph] ++ [PAD])* ++ [EOS]`
    Piper1Gpl,
    /// legacy 0.2.0 — `[BOS] ++ (map[ph] ++ [PAD])* ++ [EOS]`
    Legacy,
}

#[derive(Debug, Clone)]
struct PiperVoiceCfg {
    sample_rate: u32,
    espeak_voice: String,
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
    num_speakers: u64,
    default_speaker_id: i64,
    speaker_id_map: BTreeMap<String, i64>,
    id_map: HashMap<String, Vec<i64>>,
    id_law: IdLaw,
}

impl PiperVoiceCfg {
    /// Parse + apply defaults. `phoneme_type` null (ryan's explicit JSON
    /// null) and missing both → espeak, the legacy python Config default.
    fn parse(raw: RawVoiceCfg) -> Result<Self, LaneError> {
        let sample_rate = raw
            .audio
            .ok_or_else(|| LaneError::Load("voice json missing 'audio'".into()))?
            .sample_rate;
        let espeak_voice = raw
            .espeak
            .map(|e| e.voice)
            .unwrap_or_else(|| "en-us".to_string());
        let inference = raw.inference.unwrap_or(RawInference {
            noise_scale: None,
            length_scale: None,
            noise_w: None,
        });
        // Upstream defaults (piper1-gpl config.py:
        // DEFAULT_NOISE_SCALE/LENGTH_SCALE/NOISE_W_SCALE; legacy defaults
        // 0.667/1.0/0.8 — identical values).
        const DEFAULT_NOISE_SCALE: f32 = 0.667;
        const DEFAULT_LENGTH_SCALE: f32 = 1.0;
        const DEFAULT_NOISE_W: f32 = 0.8;
        let id_law = match raw
            .piper_version
            .as_deref()
            .and_then(version_major)
            .unwrap_or(0)
        {
            m if m >= 1 => IdLaw::Piper1Gpl,
            _ => IdLaw::Legacy,
        };
        Ok(Self {
            sample_rate,
            espeak_voice,
            noise_scale: inference.noise_scale.unwrap_or(DEFAULT_NOISE_SCALE),
            length_scale: inference.length_scale.unwrap_or(DEFAULT_LENGTH_SCALE),
            noise_w: inference.noise_w.unwrap_or(DEFAULT_NOISE_W),
            num_speakers: raw.num_speakers.unwrap_or(1),
            default_speaker_id: raw.default_speaker_id.unwrap_or(0),
            speaker_id_map: raw.speaker_id_map,
            id_map: raw.phoneme_id_map,
            id_law,
        })
    }

    /// The three law symbols (`rhasspy const.py` / piper1-gpl `const.py`):
    /// PAD `"_"` (0), BOS `"^"` (1), EOS `"$"` (2).
    fn law_ids(&self) -> Result<LawTriple, LaneError> {
        let sym = |k: &str| -> Result<Vec<i64>, LaneError> {
            self.id_map.get(k).cloned().ok_or_else(|| {
                LaneError::Load(format!("voice json id map missing {k:?} (law symbol)"))
            })
        };
        Ok((sym("^")?, sym("$")?, sym("_")?))
    }
}

/// Major-version prefix of a piper_version string ("1.0.0" → 1, "0.2.0" →
/// 0). Anything unparseable → 0 (legacy).
fn version_major(v: &str) -> Option<u32> {
    let digits: String = v.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

// ---------------------------------------------------------------------------
// Lane
// ---------------------------------------------------------------------------

struct PiperRuntime {
    session: Mutex<ort::session::Session>,
}

pub struct PiperLane {
    runtime: Option<Arc<PiperRuntime>>,
    model_path: Option<PathBuf>,
    model_label: String,
    model_sha: Option<String>,
    load_ms: u64,
}

impl Default for PiperLane {
    fn default() -> Self {
        Self::new()
    }
}

impl PiperLane {
    pub fn new() -> Self {
        Self {
            runtime: None,
            model_path: None,
            model_label: String::new(),
            model_sha: None,
            load_ms: 0,
        }
    }
}

impl crate::types::TtsLane for PiperLane {
    fn lane_name(&self) -> &'static str {
        "piper"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        let p = match model {
            ModelRef::Path(p) => p,
            // seam law: real lanes reject Pseudo; hub download is not in scope
            ModelRef::Hub { id, .. } => {
                return Err(LaneError::Load(format!(
                    "piper lane requires a disk path, got hub model '{id}' \
                     (pass the patched *-rng.onnx path)"
                )));
            }
            ModelRef::Pseudo(id) => {
                return Err(LaneError::Load(format!(
                    "piper lane requires a disk ONNX graph, got pseudo model '{id}'"
                )));
            }
        };
        if !p.is_file() {
            return Err(LaneError::Load(format!(
                "model file not found: {} (pass the patched voice graph, e.g. \
                 /teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx)",
                p.display()
            )));
        }

        let env = super::ort_env()?;
        let intra: usize = std::env::var("TTS_ORT_INTRA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(crate::lanes::physical_core_estimate);
        let session = ort::session::Session::builder(env)
            .map_err(|e| LaneError::Load(format!("ort session builder: {e}")))?
            // Session dials are the kokoro lane's (te law: threads=cores;
            // graph opt level All — rc.15's Level3 maps to ENABLE_LAYOUT,
            // which the 1.22.1 runtime rejects).
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
        let missing: Vec<&str> = REQUIRED_INPUTS
            .iter()
            .copied()
            .filter(|r| !names.iter().any(|n| n == r))
            .collect();
        if !missing.is_empty() {
            return Err(LaneError::Load(format!(
                "graph {} is missing inputs {missing:?} — this looks like a STOCK piper \
                 graph, which draws noise internally via two unseeded RandomNormalLike ops \
                 (no determinism). Patch it with eval/patch_piper_rng.py (T011 session-1; \
                 RandomNormalLike → Slice(caller-supplied seeded noise)) and pass the \
                 *-rng.onnx (inputs: {names:?})",
                p.display()
            )));
        }
        let out_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if !out_names.iter().any(|n| n == "output") {
            return Err(LaneError::Load(format!(
                "graph {} has no 'output' (audio) output (outputs: {out_names:?})",
                p.display()
            )));
        }

        self.runtime = Some(Arc::new(PiperRuntime {
            session: Mutex::new(session),
        }));
        self.model_path = Some(p.to_path_buf());
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
        let model_path = self
            .model_path
            .clone()
            .ok_or_else(|| LaneError::Synth("lane not loaded".into()))?;

        // Voice config: upstream law — `config_path = f"{model_path}.json"`.
        // VoiceRef::Path may override (alternate config beside the graph);
        // VoiceRef::Id / Default use the model-side json. The voice label
        // follows the model twin (voice is 1:1 with the model config for
        // single-speaker piper voices, unlike kokoro's voice tables).
        let voice_ref = req.voice.clone().unwrap_or(VoiceRef::Default);
        let (config_path, voice_label) = match &voice_ref {
            VoiceRef::Path(p) => {
                let label = p
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string());
                (Some(p.clone()), label)
            }
            VoiceRef::Id(id) => (None, id.clone()),
            VoiceRef::Default => (None, "default".to_string()),
        };
        let config_path = match config_path {
            Some(p) if p.is_file() => p,
            Some(p) => {
                return Err(LaneError::Load(format!(
                    "voice config not found: {} (pass the model-side json or the \
                     model_path.json twin)",
                    p.display()
                )));
            }
            // Upstream law appends .json to the FULL filename
            // (`en_US-amy-medium.onnx.json`, both piper lineages). Patched
            // `*-rng.onnx` graphs share the original's voice json — the
            // patch tool (eval/patch_piper_rng.py) writes no config of its
            // own — so the -rng twin falls back to the unsuffixed name.
            None => {
                // -rng twin: only swap in the original's json when it exists.
                match original_voice_json(&model_path) {
                    Some(orig) if orig.is_file() => orig,
                    _ => add_json_ext(&model_path),
                }
            }
        };
        let voice_label = match voice_ref {
            VoiceRef::Id(_) | VoiceRef::Default => model_path
                .file_stem()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or(voice_label),
            _ => voice_label,
        };
        let raw_json = std::fs::read(&config_path).map_err(|e| {
            LaneError::Load(format!(
                "voice config {}: {e} (piper voice json must sit beside the graph as \
                 <model>.onnx.json, the upstream config_path law; patched *-rng graphs \
                 fall back to the original's twin)",
                config_path.display()
            ))
        })?;
        let raw: RawVoiceCfg = serde_json::from_slice(&raw_json).map_err(|e| {
            LaneError::Load(format!("voice config parse {}: {e}", config_path.display()))
        })?;
        let vcfg = Arc::new(PiperVoiceCfg::parse(raw)?);
        let (bos_law, eos_law, pad_law) = vcfg.law_ids()?;

        // Scales: config defaults, overridable per run (the piper CLI flags).
        let mut noise_scale = vcfg.noise_scale;
        let mut length_scale = vcfg.length_scale;
        let mut noise_w = vcfg.noise_w;
        let mut bad_dial: Option<(String, String)> = None;
        for (k, v) in &req.lane_options.extra {
            match k.as_str() {
                "noise-scale" => match v.parse::<f32>() {
                    Ok(f) => noise_scale = f,
                    Err(_) => bad_dial = Some((k.clone(), v.clone())),
                },
                "length-scale" => match v.parse::<f32>() {
                    Ok(f) => length_scale = f,
                    Err(_) => bad_dial = Some((k.clone(), v.clone())),
                },
                "noise-w" => match v.parse::<f32>() {
                    Ok(f) => noise_w = f,
                    Err(_) => bad_dial = Some((k.clone(), v.clone())),
                },
                _ => bad_dial = Some((k.clone(), v.clone())),
            }
        }
        if let Some((k, v)) = bad_dial {
            return Err(LaneError::Synth(format!(
                "lane option {k:?}={v:?} is not a known piper dial (noise-scale / \
                 length-scale / noise-w) or not a float"
            )));
        }

        // Multi-speaker sid (upstream law; drop-dir voices are single).
        let speaker_id: Option<i64> = if vcfg.num_speakers > 1 {
            Some(match &voice_ref {
                VoiceRef::Id(name) => vcfg
                    .speaker_id_map
                    .get(name)
                    .copied()
                    .unwrap_or(vcfg.default_speaker_id),
                _ => vcfg.default_speaker_id,
            })
        } else {
            None
        };

        // Chunks: seam-provided (T013), else one chunk over the whole text.
        let chunks_in: Vec<TextChunk> = if req.chunks.is_empty() {
            vec![TextChunk {
                text: req.text.clone(),
                byte_range: [0, req.text.len() as u32],
            }]
        } else {
            req.chunks.clone()
        };

        let rng: rand::rngs::StdRng = match req.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            // Unseeded: entropy seed → outputs uncorrelated across runs
            // (byte-identity gate only runs with --seed; the flow lanes'
            // NonDeterministic contract).
            None => rand::rngs::StdRng::seed_from_u64(rand::random()),
        };
        let normal = Normal::new(0.0f32, 1.0).expect("std normal params are valid");

        let meta = RunMeta {
            lane: "piper",
            model: self.model_label.clone(),
            model_sha: self.model_sha.clone(),
            voice: voice_label,
            sample_rate: vcfg.sample_rate,
            lang: req.lang,
            determinism: match req.seed {
                Some(_) => Determinism::Seeded,
                None => Determinism::NonDeterministic(
                    "unseeded piper run (entropy seed) — pass --seed for byte-identity".to_string(),
                ),
            },
            seed: req.seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "ort-dylib (onnxruntime 1.22.x)".to_string(),
            load_ms: self.load_ms,
            // Grain = one T013 text chunk (what `tts serve` and the relay
            // parity need; the mel-level streaming lineage is phase B and
            // gated on split-graph artifacts).
            chunk_policy: "text-chunks(T013)".to_string(),
            restart_voice: false,
        };

        // Per-stream synthesis state (the `SynthStream` iterator's captures).
        struct ChunkState {
            runtime: Arc<PiperRuntime>,
            chunk_iter: std::vec::IntoIter<TextChunk>,
            vcfg: Arc<PiperVoiceCfg>,
            bos: Vec<i64>,
            eos: Vec<i64>,
            pad: Vec<i64>,
            noise_scale: f32,
            length_scale: f32,
            noise_w: f32,
            speaker_id: Option<i64>,
            rng: rand::rngs::StdRng,
            normal: Normal<f32>,
            warned_missing: HashSet<char>,
            index: u32,
        }

        /// One chunk of synthesis. Returns `Ok(None)` at stream end; the iterator
        /// closure just `.transpose()`s this into its `Option<Result<_, _>>` shape.
        fn next_chunk(st: &mut ChunkState) -> Result<Option<ChunkOut>, LaneError> {
            let chunk = match st.chunk_iter.next() {
                Some(c) => c,
                None => return Ok(None),
            };
            // ---- g2p: espeak-rs phonemize + id assembly ----
            let t_g = Instant::now();
            let sentences = espeak::phonemize(&chunk.text, &st.vcfg.espeak_voice)?;
            let mut sent_ids: Vec<Vec<i64>> = Vec::with_capacity(sentences.len());
            for s in &sentences {
                sent_ids.push(assemble_ids(
                    &st.vcfg.id_map,
                    (&st.bos, &st.eos, &st.pad),
                    st.vcfg.id_law,
                    s,
                    &mut st.warned_missing,
                ));
            }
            let g2p = t_g.elapsed();

            // ---- acoustic: ORT graph run per sentence ----
            let t_a = Instant::now();
            let mut audio_f: Vec<f32> = Vec::new();
            // Upstream piper applies `sentence_silence` (default 0.0 s) between
            // sentences; 0.0 is our constant, so no padding runs.
            let mut sess = st
                .runtime
                .session
                .lock()
                .map_err(|_| LaneError::Synth("piper session poisoned".into()))?;
            for ids in &sent_ids {
                let n_text = ids.len();
                if n_text <= st.bos.len() + st.eos.len() + st.pad.len() {
                    // Degenerate: no phonemes (whitespace sentence).
                    continue;
                }
                // Seeded noise draw order (pinned): noise_0 (2·N) then noise_1
                // (192·budget), eps ~ N(0,1).
                let noise_0: Vec<f32> = st
                    .normal
                    .sample_iter(&mut st.rng)
                    .take(2 * n_text)
                    .collect();
                let noise_1: Vec<f32> = st
                    .normal
                    .sample_iter(&mut st.rng)
                    .take(MEL_BINS * NOISE_1_BUDGET_FRAMES)
                    .collect();
                let input_t = super::ort(
                    Tensor::from_array(([1usize, n_text], ids.clone())),
                    "input tensor",
                )?;
                let lengths_t = super::ort(
                    Tensor::from_array(([1usize], vec![n_text as i64])),
                    "input_lengths tensor",
                )?;
                let scales_t = super::ort(
                    Tensor::from_array((
                        [3usize],
                        vec![st.noise_scale, st.length_scale, st.noise_w],
                    )),
                    "scales tensor",
                )?;
                let noise0_t = super::ort(
                    Tensor::from_array(([1usize, 2usize, n_text], noise_0)),
                    "noise_0 tensor",
                )?;
                let noise1_t = super::ort(
                    Tensor::from_array(([1usize, MEL_BINS, NOISE_1_BUDGET_FRAMES], noise_1)),
                    "noise_1 tensor",
                )?;
                let sid_t = match st.speaker_id {
                    Some(sid) => Some(super::ort(
                        Tensor::from_array(([1usize], vec![sid])),
                        "sid tensor",
                    )?),
                    None => None,
                };
                let outs = match sid_t {
                    Some(sid) => super::ort(
                        sess.run(ort::inputs![
                            "input" => input_t,
                            "input_lengths" => lengths_t,
                            "scales" => scales_t,
                            "noise_0" => noise0_t,
                            "noise_1" => noise1_t,
                            "sid" => sid,
                        ]),
                        "piper graph run",
                    )?,
                    None => super::ort(
                        sess.run(ort::inputs![
                            "input" => input_t,
                            "input_lengths" => lengths_t,
                            "scales" => scales_t,
                            "noise_0" => noise0_t,
                            "noise_1" => noise1_t,
                        ]),
                        "piper graph run",
                    )?,
                };
                let wave = outs.get("output").ok_or_else(|| {
                    LaneError::Synth("piper graph has no 'output' (audio)".into())
                })?;
                let (_, data) = super::ort(wave.try_extract_tensor::<f32>(), "audio extract")?;
                audio_f.extend_from_slice(data);
                drop(outs);
            }
            drop(sess);
            let acoustic = t_a.elapsed();

            // ---- io: the encode law (per format) ----
            let t_i = Instant::now();
            let pcm: Vec<i16> = match st.vcfg.id_law {
                IdLaw::Piper1Gpl => encode_int16_piper1(&audio_f),
                IdLaw::Legacy => encode_int16_legacy(&audio_f),
            };
            let io = t_i.elapsed();
            // Law mirror (kokoro): empty token stream → 10 ms honest chunk.
            let pcm = if pcm.is_empty() {
                vec![0i16; st.vcfg.sample_rate as usize / 100]
            } else {
                pcm
            };

            let out = ChunkOut {
                index: st.index,
                stages: StageTimings {
                    g2p,
                    acoustic,
                    // Fused graph: no vocoder boundary exists (§4 table).
                    vocoder: std::time::Duration::ZERO,
                    io,
                },
                pcm,
            };
            st.index += 1;
            Ok(Some(out))
        }

        let mut state = ChunkState {
            runtime,
            chunk_iter: chunks_in.into_iter(),
            vcfg,
            bos: bos_law,
            eos: eos_law,
            pad: pad_law,
            noise_scale,
            length_scale,
            noise_w,
            speaker_id,
            rng,
            normal,
            warned_missing: HashSet::new(),
            index: 0,
        };

        let iter = std::iter::from_fn(move || next_chunk(&mut state).transpose());

        Ok(SynthStream {
            meta,
            chunks: Box::new(iter),
        })
    }
}

// ---------------------------------------------------------------------------
// Phoneme-id assembly + encode law (verbatim ports, see module header)
// ---------------------------------------------------------------------------

/// Appends `.json` to the full filename (both piper lineages: the config
/// twin of `x.onnx` is `x.onnx.json`, NOT `x.json`).
fn add_json_ext(model_path: &Path) -> PathBuf {
    model_path.with_file_name(format!(
        "{}.json",
        model_path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default()
    ))
}

/// For a patched `*-rng.onnx` graph, the path to the ORIGINAL graph's
/// config twin (`en_US-amy-medium-rng.onnx` → `en_US-amy-medium.onnx.json`).
/// `None` when the file isn't a `-rng` twin.
fn original_voice_json(model_path: &Path) -> Option<PathBuf> {
    let file = model_path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())?;
    file.strip_suffix("-rng.onnx")
        .map(|stem| model_path.with_file_name(format!("{stem}.onnx.json")))
}

/// One IPA sentence string → phoneme ids. Per-codepoint (both drop-dir
/// maps are 100% single-codepoint keys). Missing phoneme → warn once per
/// symbol + skip (upstream `_LOGGER.warning` law).
fn assemble_ids(
    map: &HashMap<String, Vec<i64>>,
    law: (&[i64], &[i64], &[i64]),
    law_table: IdLaw,
    phonemes: &str,
    warned: &mut HashSet<char>,
) -> Vec<i64> {
    let (bos, eos, pad) = law;
    let mut ids = Vec::with_capacity(phonemes.chars().count() * (pad.len() + 4));
    ids.extend_from_slice(bos);
    if matches!(law_table, IdLaw::Piper1Gpl) {
        // piper1-gpl law: PAD after BOS.
        ids.extend_from_slice(pad);
    }
    for ph in phonemes.chars() {
        match map.get(&ph.to_string()) {
            Some(pids) => {
                ids.extend_from_slice(pids);
                ids.extend_from_slice(pad);
            }
            None => {
                if warned.insert(ph) {
                    eprintln!("[piper] missing phoneme from id map: {ph}");
                }
            }
        }
    }
    ids.extend_from_slice(eos);
    ids
}

/// 1.0.0 encode law (piper1-gpl `src/piper/voice.py`): peak-normalize
/// (max|a| < 1e-8 → zeros) → clip [-1, 1] → `× 32767.0` → clip ±32767.0 →
/// trunc → i16.
fn encode_int16_piper1(audio: &[f32]) -> Vec<i16> {
    let max_abs = audio.iter().fold(0f32, |m, &s| s.abs().max(m));
    if max_abs < EMPTY_AUDIO_THRESHOLD {
        return vec![0i16; audio.len()];
    }
    audio
        .iter()
        .map(|&s| {
            let norm = (s / max_abs).clamp(-1.0, 1.0);
            let v = (norm * MAX_WAV_VALUE).clamp(-MAX_WAV_VALUE, MAX_WAV_VALUE);
            v as i16
        })
        .collect()
}

/// Legacy encode law (`audio_float_to_int16`): `a ×
/// 32767/max(0.01, max|a|)` → clip ±32767.0 → trunc → i16.
fn encode_int16_legacy(audio: &[f32]) -> Vec<i16> {
    let max_abs = audio.iter().fold(0f32, |m, &s| s.abs().max(m));
    let denom = max_abs.max(LEGACY_NORM_FLOOR);
    audio
        .iter()
        .map(|&s| {
            let v = s * MAX_WAV_VALUE / denom;
            v.clamp(-MAX_WAV_VALUE, MAX_WAV_VALUE) as i16
        })
        .collect()
}

// ---------------------------------------------------------------------------
// espeak-rs G2P shim (the only espeak call site; keeps lane imports tight)
// ---------------------------------------------------------------------------

mod espeak {
    /// Phonemize one T013 chunk: one IPA string per sentence (clauses
    /// accumulated, punctuation embedded; no separator — codepoint map law).
    /// Errors surface as Synth (espeak init/data problems are per-run
    /// conditions: PIPER_ESPEAKNG_DATA_DIRECTORY contract).
    pub fn phonemize(
        text: &str,
        espeak_voice: &str,
    ) -> Result<Vec<String>, crate::types::LaneError> {
        espeak_rs::text_to_phonemes(text, espeak_voice, None).map_err(|e| {
            crate::types::LaneError::Synth(format!(
                "espeak phonemize failed ({e}) — set PIPER_ESPEAKNG_DATA_DIRECTORY to the \
                 parent of espeak-ng-data (system: /usr/share)"
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — assembly + encode laws (the two verbatim ports)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn law_map() -> HashMap<String, Vec<i64>> {
        HashMap::from([
            ("_".to_string(), vec![0]),
            ("^".to_string(), vec![1]),
            ("$".to_string(), vec![2]),
            (" ".to_string(), vec![3]),
            ("t".to_string(), vec![15]),
            ("ɛ".to_string(), vec![79]),
            ("s".to_string(), vec![17]),
            (".".to_string(), vec![10]),
        ])
    }

    #[test]
    fn assembly_piper1_law_has_pad_after_bos() {
        let m = law_map();
        let mut warned = HashSet::new();
        let ids = assemble_ids(
            &m,
            (&[1i64], &[2i64], &[0i64]),
            IdLaw::Piper1Gpl,
            "tɛs.",
            &mut warned,
        );
        // 1.0.0 law: [BOS, PAD] ++ (map[ph] ++ [PAD])* ++ [EOS]
        assert_eq!(
            ids,
            vec![1, 0, 15, 0, 79, 0, 17, 0, 10, 0, 2],
            "[BOS,PAD] ++ (t,PAD)(ɛ,PAD)(s,PAD)(.,PAD) ++ [EOS]"
        );
    }

    #[test]
    fn assembly_legacy_law_has_no_pad_after_bos() {
        let m = law_map();
        let mut warned = HashSet::new();
        let ids = assemble_ids(
            &m,
            (&[1i64], &[2i64], &[0i64]),
            IdLaw::Legacy,
            "tɛs.",
            &mut warned,
        );
        // 0.2.0 law: [BOS] ++ (map[ph] ++ [PAD])* ++ [EOS]
        assert_eq!(ids, vec![1, 15, 0, 79, 0, 17, 0, 10, 0, 2]);
    }

    #[test]
    fn assembly_missing_phoneme_warns_once_and_skips() {
        let m = law_map();
        let mut warned = HashSet::new();
        let ids = assemble_ids(
            &m,
            (&[1i64], &[2i64], &[0i64]),
            IdLaw::Piper1Gpl,
            "tq", // 'q' not in map
            &mut warned,
        );
        assert_eq!(ids, vec![1, 0, 15, 0, 2]); // q skipped
        assert!(warned.contains(&'q'));
        // Second miss of the same symbol: no re-entry into warned (warned
        // dedupe), but skip still applies.
        let ids2 = assemble_ids(
            &m,
            (&[1i64], &[2i64], &[0i64]),
            IdLaw::Piper1Gpl,
            "q",
            &mut warned,
        );
        assert_eq!(ids2, vec![1, 0, 2]);
    }

    #[test]
    fn encode_piper1_peaks_and_truncs() {
        // Peak-normalize drives max to full scale; 32767.0 trunc → 32767.
        let audio = vec![0.5f32, -0.25, 1.0, -2.0];
        let pcm = encode_int16_piper1(&audio);
        assert_eq!(pcm[3], -32767, "max|-2| normalizes to -1 → ×32767 → -32767");
        let audio = vec![0.5f32, -0.25];
        let pcm = encode_int16_piper1(&audio);
        // Peak-norm law applies to EVERY clip: 0.5/max(0.5)=1.0 → 32767.
        assert_eq!(pcm[0], 32767, "0.5/0.5=1.0 → ×32767.0 truncs to 32767");
        assert_eq!(
            pcm[1], -16383,
            "-0.25/0.5=-0.5 → ×32767.0 = -16383.5 truncs to -16383"
        );
        let audio = vec![0.5f32, -1.0];
        let pcm = encode_int16_piper1(&audio);
        assert_eq!(pcm[0], 16383, "already-peaked: 0.5*32767.0 truncs to 16383");
        assert_eq!(
            pcm[1], -32767,
            "already-peaked: -1.0*32767.0 truncs to -32767"
        );
    }

    #[test]
    fn encode_piper1_below_threshold_is_silence() {
        let pcm = encode_int16_piper1(&[1e-9f32, -1e-9, 5e-10]);
        assert!(pcm.iter().all(|&s| s == 0));
        assert_eq!(pcm.len(), 3);
    }

    #[test]
    fn encode_legacy_uses_0_01_floor() {
        // max 0.0005 < 0.01 → denominator floored at 0.01 → gain ×100 over
        // naive peak-norm: 0.0005 → 0.05 → ×32767 → 1638 (trunc).
        let audio = vec![0.0005f32];
        let pcm = encode_int16_legacy(&audio);
        assert_eq!(pcm[0], 1638);
    }

    #[test]
    fn version_major_parses() {
        assert_eq!(version_major("1.0.0"), Some(1));
        assert_eq!(version_major("2.0.0"), Some(2));
        assert_eq!(version_major("0.2.0"), Some(0));
    }

    #[test]
    fn voice_cfg_defaults_are_upstream() {
        let raw = serde_json::from_str::<RawVoiceCfg>(
            r#"{
                "audio": {"sample_rate": 22050},
                "phoneme_id_map": {"_": [0], "^": [1], "$": [2]}
            }"#,
        )
        .expect("minimal legacy cfg parses");
        let cfg = PiperVoiceCfg::parse(raw).expect("defaults apply");
        assert_eq!(cfg.noise_scale, 0.667);
        assert_eq!(cfg.length_scale, 1.0);
        assert_eq!(cfg.noise_w, 0.8);
        assert_eq!(cfg.id_law, IdLaw::Legacy);
        assert_eq!(cfg.espeak_voice, "en-us");
        assert_eq!(cfg.num_speakers, 1);
        // Explicit null phoneme_type (ryan's json) also parses — upstream
        // would crash; deviation-labeled permissive default in the lane.
    }
}
