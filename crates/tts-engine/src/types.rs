//! Seam types — `docs/seam.md` §3. These definitions are law; lanes implement
//! the trait against them, the CLI builds them, the telemetry recorder
//! serializes them. Field names here mirror the chunks.jsonl schema.

use std::path::PathBuf;
use std::time::Duration;

use crate::telemetry::TelemetryError;

/// BCP-47-ish language tag, reduced to the bases our first lanes need.
/// Lane-specific tags extend via `LaneOptions`, not by widening this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    En,
    Fr,
    De,
    Es,
    Pt,
}

impl Lang {
    /// Parse `en`, `EN`, `en_US`, `fr-FR`, ... (base token decides).
    pub fn parse(s: &str) -> Option<Lang> {
        let lowered = s.trim().to_lowercase();
        let base = lowered.split(['-', '_']).next().unwrap_or("");
        match base {
            "en" => Some(Lang::En),
            "fr" => Some(Lang::Fr),
            "de" => Some(Lang::De),
            "es" => Some(Lang::Es),
            "pt" => Some(Lang::Pt),
            _ => None,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::Es => "es",
            Lang::Pt => "pt",
        }
    }
}

/// Which voice to use. `Id` = named voice in the model's voice table,
/// `Path` = external .pt/.npz/.bin embedding file, `Default` = lane's own
/// default. Missing `--voice` maps to `Default`.
#[derive(Debug, Clone)]
pub enum VoiceRef {
    Id(String),
    Path(PathBuf),
    Default,
}

impl VoiceRef {
    /// Header `voice` label for telemetry (paths are labeled by file name).
    pub fn label(&self) -> String {
        match self {
            VoiceRef::Id(s) => s.clone(),
            VoiceRef::Path(p) => p
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
            VoiceRef::Default => "default".to_string(),
        }
    }
}

/// Either an on-disk artifact (loaded mmap/zero-copy per the seam mandate)
/// or an explicit HF hub id (`--allow-hf` only). `Pseudo` is for
/// diagnostic/stub lanes only — a real lane must reject it with
/// `LaneError::Load`.
#[derive(Debug, Clone)]
pub enum ModelRef {
    Path(PathBuf),
    Hub {
        id: String,
        revision: Option<String>,
    },
    /// No weights — the model name is a lane-local identifier.
    Pseudo(String),
}

impl ModelRef {
    /// Header `model` label.
    pub fn label(&self) -> String {
        match self {
            ModelRef::Path(p) => p
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
            ModelRef::Hub { id, .. } => id.clone(),
            ModelRef::Pseudo(id) => id.clone(),
        }
    }
}

/// One pre-chunked piece of input text with its byte range in the original
/// text (the range indexes the ORIGINAL string; chunks may be re-chunked
/// lane-side only within these spans).
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub byte_range: [u32; 2],
}

/// Lane-specific dials beyond the seam contract (quantization flavor, decode
/// policy knobs, ...). Never used to smuggle in seam-side concerns.
#[derive(Debug, Clone, Default)]
pub struct LaneOptions {
    pub chunk_policy: Option<String>,
    pub extra: Vec<(String, String)>,
}

impl LaneOptions {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.extra
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Per-stage timings, fixed 4-slot vocabulary `{g2p, acoustic, vocoder, io}`
/// (`docs/seam.md` §4). Renaming per lane is forbidden — cross-lane
/// comparability is the point. Lanes report data; the seam serializes it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StageTimings {
    pub g2p: Duration,
    pub acoustic: Duration,
    pub vocoder: Duration,
    pub io: Duration,
}

impl StageTimings {
    pub fn total(self) -> Duration {
        self.g2p + self.acoustic + self.vocoder + self.io
    }
}

/// One output chunk with an honest stage split. `index` must be 0-based and
/// consecutive across the stream. `pcm` must be mono i16; `sample_rate` lives
/// on the header, not the chunk.
#[derive(Clone)]
pub struct ChunkOut {
    pub index: u32,
    pub stages: StageTimings,
    pub pcm: Vec<i16>,
}

impl std::fmt::Debug for ChunkOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkOut")
            .field("index", &self.index)
            .field("stages", &self.stages)
            .field("pcm_len", &self.pcm.len())
            .finish()
    }
}

/// Determinism class (`docs/seam.md` §7). Drives the byte-identity gate.
#[derive(Debug, Clone, PartialEq)]
pub enum Determinism {
    /// Same model+voice+text+seed ⇒ bit-identical PCM, always.
    Deterministic,
    /// Bit-identical when the same `seed` is supplied; ungated runs vary.
    Seeded,
    /// No reproducibility contract (reason reported in telemetry).
    NonDeterministic(String),
}

impl Determinism {
    pub fn as_str(&self) -> &'static str {
        match self {
            Determinism::Deterministic => "deterministic",
            Determinism::Seeded => "seeded",
            Determinism::NonDeterministic(_) => "nondeterministic",
        }
    }
}

/// Run metadata emitted as the telemetry header row.
#[derive(Debug, Clone)]
pub struct RunMeta {
    pub lane: &'static str,
    pub model: String,
    /// First 16 hex chars of the (m)map'd weight file(s) sha256.
    pub model_sha: Option<String>,
    pub voice: String,
    pub sample_rate: u32,
    pub lang: Lang,
    pub determinism: Determinism,
    pub seed: Option<u64>,
    pub lane_version: String,
    pub runtime: String,
    /// Model load wall ms — always separate from synth timing.
    pub load_ms: u64,
    pub chunk_policy: String,
    pub restart_voice: bool,
}

/// Load-phase result (kept separate from synth: load parity law).
#[derive(Debug, Clone, Copy)]
pub struct LoadInfo {
    pub load_ms: u64,
}

/// Lane errors classified by phase so the CLI can exit-code them per the seam.
#[derive(Debug, thiserror::Error)]
pub enum LaneError {
    #[error("load failed: {0}")]
    Load(String),
    #[error("synthesis failed: {0}")]
    Synth(String),
    #[error("io error: {0}")]
    Io(String),
}

impl From<TelemetryError> for LaneError {
    fn from(e: TelemetryError) -> Self {
        LaneError::Io(e.to_string())
    }
}

/// Synthesis request — seam input side.
#[derive(Debug, Clone)]
pub struct SynthRequest {
    pub text: String,
    /// Pre-chunked by the seam (`chunking::chunk_text`); empty means
    /// "lane chunks itself over the whole text".
    pub chunks: Vec<TextChunk>,
    pub lang: Lang,
    pub voice: Option<VoiceRef>,
    pub seed: Option<u64>,
    pub lane_options: LaneOptions,
}

/// Synthesis stream — seam output side. Iterate lazily; each item handed to
/// the sink is measured for TTFA/RTF by the telemetry recorder.
pub struct SynthStream {
    pub meta: RunMeta,
    pub chunks: Box<dyn Iterator<Item = Result<ChunkOut, LaneError>>>,
}

impl std::fmt::Debug for SynthStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SynthStream")
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

/// The fixed telemetry stage-slot vocabulary with the meaning each slot must
/// keep across all lanes (lane → slot mapping table in `docs/seam.md` §4).
#[derive(Debug, Clone, Copy)]
pub struct StageSlot {
    pub slot: &'static str,
    pub meaning: &'static str,
}

pub const STAGE_SLOTS: [StageSlot; 4] = [
    StageSlot {
        slot: "g2p",
        meaning: "text -> model input prep",
    },
    StageSlot {
        slot: "acoustic",
        meaning: "model input -> audio-domain intermediate",
    },
    StageSlot {
        slot: "vocoder",
        meaning: "intermediate -> PCM",
    },
    StageSlot {
        slot: "io",
        meaning: "resample/convert/flush overhead",
    },
];

/// Every lane behind the seam implements this. Model-agnostic by law: the
/// engine never knows which family lives behind it.
pub trait TtsLane: Send {
    fn lane_name(&self) -> &'static str;
    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError>;
    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError>;
}
