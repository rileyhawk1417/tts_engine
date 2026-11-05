//! T010 chatterbox lane (layer B): ResembleAI **Chatterbox Turbo** via an
//! in-process FFI to llama.cpp's Native Chatterbox TTS path.
//!
//! # Provenance pin
//!
//! - Engine: **ggml-org/llama.cpp `186eab1cb83899282bec44b837270959b56ee8f3`**
//!   (2026-11-05, PR #26254 lineage) — static `libllama.a` + `libllama-common.a`
//!   (+ its `-base` transitive) + `libmtmd.a` + `libggml{,-cpu,-base}.a`
//!   built at `/teranux/third_party/llama.cpp` (cmake Release, CPU-only,
//!   shared off). Override with `LLAMA_CPP_DIR`; the link recipe lives in
//!   `build.rs`.
//! - C++ shim: `src/shim/chatterbox_shim.cpp` — C ABI over
//!   `common_init_from_params` + `mtmd_helper_gen_audio_*` +
//!   `common_sampler_*`, mirroring `tools/tts/tts.cpp` (the llama-tts
//!   binary) line for line.
//! - Weights (q8_0, llama.cpp layout, gitignored drop dir):
//!   backbone `chatterbox-turbo-Q8_0.gguf` + `mmproj-chatterbox-turbo-F16.gguf`
//!   (both 1.03 GB total) — `/teranux/hugging_face/t012-drop/chatterbox/`.
//!
//! # Model math ownership (seam contract, docs/seam.md)
//!
//! The shim + llama.cpp own the model math; the lane owns ONLY the seam:
//! voice resolution, per-chunk orchestration, stage timings, and the
//! encode law (WAV bytes from the pipeline → i16 mono, no re-encode).
//! Determinism-class: **Seeded** (seed drives both the semantic-code
//! sampler and the CFM ODE trajectory; threads are a dial, see below).
//!
//! # Laws & deviations
//!
//! - **Faithful port** (T008 law): sampling chain = llama.cpp common
//!   defaults exactly (top_k 40, top_p 0.95, min_p 0.05, temp 0.80,
//!   penalty_repeat 1.0, penalty window 64; NOT any "improved" values),
//!   `embedding = true`, mmproj CPU-only, `n_batch` 2048. The 512-frame
//!   cap per chunk (`n_predict > 0 ? n_predict : 512`) is the tool's.
//! - **DLV-1** (labeled deviation): the shim rebuilds a
//!   `common_sampler` per synth call with the call's seed — llama-tts is
//!   one-shot per process, our lane is multi-chunk per process. Effect:
//!   each chunk is an independent decode (`set_input` wipes that
//!   sequence's KV); bytes depend on (text-chunks, seed, threads) but
//!   **not** on chunk position. Same class as the flow lanes' per-chunk
//!   seeding, verified by the byte-identity gate.
//! - **DLV-2**: the shim's WAV buffer is pipeline-owned; the lane copies
//!   it into `Vec<i16>` per chunk before the next `set_input` resets it.
//! - **Seed width**: llama.cpp sampler/CFM seeds are u32; seam-u64 seeds
//!   truncate (documented; seed 42 unaffected).
//! - **Threads dial**: ggml intra-op threads fix the reduction order →
//!   byte-identity is per (lane, quant, dial) triple with dial =
//!   threads, exactly like the flow lanes (measured: t4 ≠ t8 ≠ t16 at
//!   byte level). Default dial here = **4** (measured sweet spot;
//!   t16 loses to SMT contention, T010 layer-A log).
//! - **Worker-loop law**: one lane instance per process owns all llama
//!   contexts (`llama_backend_*` init is once-flagged in the shim); the
//!   lane is NOT concurrent-safe within `synthesize` (single decode per
//!   call), matching the resident-service worker-loop=1 design.
//! - **No G2P stage**: prompts go straight to the backbone's own text
//!   path; `StageTimings.g2p` stays ZERO.
//! - **Stage clock mapping** (§4): shim `prompt_s` + `gen_s` →
//!   `acoustic` (semantic + frame-gen are fused in `step_gen`);
//!   `wav_s` → `vocoder` (code2wav/hift); `io` = WAV parse.
//! - **Voice**: `VoiceRef::Default` == shipped voice 0 (the
//!   conds.pt-derived pack, license MIT/Boba — provenance header in
//!   `shim`); `VoiceRef::Id(name)` = shipped voice by name (turbo ships
//!   one voice, "default"); `VoiceRef::Path(wav)` = **voice cloning** —
//!   the WAV is preprocessed per `set_input` (the cloning preproc ≈
//!   0.4 s lands inside the prompt stage, T010 layer-A measurement).
//!   Cloning reference: > 5 s audio, 10–15 s recommended (upstream README).
//! - **RSS deviation (T014 contract)**: peak resident ≈ **2.4 GB**
//!   (weights 1.03 GB + KV + compute buffers) — far above the
//!   resident-service ≤300 MB warm/lane budget; recorded as a known
//!   deviation pending the resident-serve gate.
//! - **max frames per chunk**: 512 (≈10.2 s at 24 kHz, 50:1 frames:
//!   samples); an over-long T013 chunk would truncate (the tool's
//!   default cap) — chunk plans under the eval harness stay below it.
//!
//! # Telemetry contract
//!
//! `load_ms` = wall time of `tts_cbx_init` ONLY (model + mmproj +
//! pipeline ctx, cold; there is no lazy per-voice build on this lane).
//! Chunk rows carry `acoustic = prompt + gen`, `vocoder = code2wav`,
//! `g2p = 0`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::types::{
    ChunkOut, Determinism, LaneError, LoadInfo, ModelRef, RunMeta, StageTimings, SynthRequest,
    SynthStream, VoiceRef,
};

// --------------------------------------------------------------------------
// C ABI (tts_cbx_* — shim: src/shim/chatterbox_shim.cpp)
// --------------------------------------------------------------------------

#[repr(C)]
struct TtsCbxSession {
    _opaque: [u8; 0],
}

// Used only through the fully-qualified trait path (`impl crate::types::TtsLane
// for ...`); declared here for the `TtsLane` bound in `make_lane`.
#[allow(dead_code)]
unsafe extern "C" {
    fn tts_cbx_init(
        model_path: *const std::ffi::c_char,
        mmproj_path: *const std::ffi::c_char,
        threads: i32,
        n_batch: i32,
        n_predict_cap: i32,
    ) -> *mut TtsCbxSession;
    fn tts_cbx_ok(s: *mut TtsCbxSession) -> i32;
    fn tts_cbx_set_speaker_wav(s: *mut TtsCbxSession, wav_path: *const std::ffi::c_char) -> i32;
    fn tts_cbx_reset_speaker(s: *mut TtsCbxSession) -> i32;
    fn tts_cbx_sample_rate(s: *mut TtsCbxSession) -> i32;
    /// Voice inventory (chatterbox-turbo ships exactly one — "default").
    /// Surfaced for T014 voice endpoints; unused by the lane today.
    #[allow(dead_code)]
    fn tts_cbx_n_voices(s: *mut TtsCbxSession) -> i32;
    fn tts_cbx_voice_name(s: *mut TtsCbxSession) -> *const std::ffi::c_char;
    fn tts_cbx_synthesize(
        s: *mut TtsCbxSession,
        text: *const std::ffi::c_char,
        seed: u64,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        temp: f32,
        penalty_repeat: f32,
        out_sample_rate: *mut i32,
        out_wav: *mut *const std::ffi::c_char,
        out_wav_len: *mut usize,
        out_samples: *mut i64,
        out_frames: *mut i32,
        out_prompt_s: *mut f64,
        out_gen_s: *mut f64,
        out_wav_s: *mut f64,
    ) -> i32;
    fn tts_cbx_last_error(s: *mut TtsCbxSession) -> *const std::ffi::c_char;
    fn tts_cbx_free(s: *mut TtsCbxSession);
}

/// Sampler dial set resolved from lane options (llama-tts pinned defaults
/// unless overridden; `max_frames = None` = the shim's init-time cap).
#[derive(Debug)]
struct CbxDials {
    top_k: i32,
    top_p: f32,
    min_p: f32,
    temp: f32,
    penalty_repeat: f32,
    max_frames: Option<i32>,
}

/// Raw pointer newtype: the shim session owns non-Send C++ state, but the
/// lane contract restricts use to one worker thread at a time (Mutex-gated
/// sync + the resident worker-loop=1 law). Same pattern as the piper lane's
/// ORT session wrapper.
struct CbxHandle(*mut TtsCbxSession);
unsafe impl Send for CbxHandle {}

struct CbxRuntime {
    sess: CbxHandle,
    sample_rate: u32,
    shipped_voice: String,
}

unsafe impl Sync for CbxRuntime {}

impl Drop for CbxRuntime {
    fn drop(&mut self) {
        unsafe { tts_cbx_free(self.sess.0) };
    }
}

impl CbxRuntime {
    fn error(&self) -> Option<String> {
        unsafe {
            let p = tts_cbx_last_error(self.sess.0);
            if p.is_null() {
                None
            } else {
                Some(std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned())
            }
        }
    }
}

// llama-tts sampling defaults (common_params_sampling, common/common.h) —
// the measured, validated baseline (layer-A recon; do NOT change silently).
const SAMPLING_TOP_K: i32 = 40;
const SAMPLING_TOP_P: f32 = 0.95;
const SAMPLING_MIN_P: f32 = 0.05;
const SAMPLING_TEMP: f32 = 0.80;
const SAMPLING_PENALTY_REPEAT: f32 = 1.0;
const DEFAULT_THREADS: i32 = 4;
const DEFAULT_N_BATCH: i32 = 2048;
const DEFAULT_N_PREDICT_CAP: i32 = 512;

// --------------------------------------------------------------------------
// Lane
// --------------------------------------------------------------------------

pub struct GgmlFfiLane {
    runtime: Option<Arc<CbxRuntime>>,
    model_path: Option<PathBuf>,
    mmproj_path: Option<PathBuf>,
    model_label: String,
    model_sha: Option<String>,
    load_ms: u64,
}

impl Default for GgmlFfiLane {
    fn default() -> Self {
        Self::new()
    }
}

impl GgmlFfiLane {
    pub fn new() -> Self {
        Self {
            runtime: None,
            model_path: None,
            mmproj_path: None,
            model_label: String::new(),
            model_sha: None,
            load_ms: 0,
        }
    }

    /// Lane-option helper: `(key, value)` dials override the pinned
    /// llama-tts defaults. Unknown keys are a hard error (the seam law:
    /// no silent dial drift). Returns the sampler dial tuple plus the
    /// `max-frames` override (`None` = shim init cap).
    fn resolve_dials(req: &SynthRequest) -> Result<CbxDials, LaneError> {
        let mut top_k = SAMPLING_TOP_K;
        let mut top_p = SAMPLING_TOP_P;
        let mut min_p = SAMPLING_MIN_P;
        let mut temp = SAMPLING_TEMP;
        let mut penalty_repeat = SAMPLING_PENALTY_REPEAT;
        let mut max_frames: Option<i32> = None;
        let mut bad: Option<(String, String)> = None;
        for (k, v) in &req.lane_options.extra {
            match k.as_str() {
                "top-k" => match v.parse::<i32>() {
                    Ok(n) if n >= 1 => top_k = n,
                    _ => bad = Some((k.clone(), v.clone())),
                },
                "top-p" => match v.parse::<f32>() {
                    Ok(f) if (0.0..=1.0).contains(&f) => top_p = f,
                    _ => bad = Some((k.clone(), v.clone())),
                },
                "min-p" => match v.parse::<f32>() {
                    Ok(f) if (0.0..=1.0).contains(&f) => min_p = f,
                    _ => bad = Some((k.clone(), v.clone())),
                },
                "temp" => match v.parse::<f32>() {
                    Ok(f) if f >= 0.0 => temp = f,
                    _ => bad = Some((k.clone(), v.clone())),
                },
                "penalty-repeat" => match v.parse::<f32>() {
                    Ok(f) if f >= 1.0 => penalty_repeat = f,
                    _ => bad = Some((k.clone(), v.clone())),
                },
                "max-frames" => match v.parse::<i32>() {
                    Ok(n) if n > 0 => max_frames = Some(n),
                    _ => bad = Some((k.clone(), v.clone())),
                },
                _ => bad = Some((k.clone(), v.clone())),
            }
        }
        if let Some((k, v)) = bad {
            return Err(LaneError::Synth(format!(
                "lane option {k:?}={v:?} is not a known ggml-ffi dial (top-k / \
                 top-p / min-p / temp / penalty-repeat / max-frames) or out of range"
            )));
        }
        Ok(CbxDials {
            top_k,
            top_p,
            min_p,
            temp,
            penalty_repeat,
            max_frames,
        })
    }
}

fn cstr(p: &Path) -> Result<std::ffi::CString, LaneError> {
    std::ffi::CString::new(p.to_string_lossy().into_owned())
        .map_err(|e| LaneError::Load(format!("model path has interior NUL: {e}")))
}

/// mmproj twin law (llama.cpp convention): `mmproj*.gguf` sits beside the
/// backbone in the drop dir; exactly one must resolve unless the caller
/// passes `mmproj=<path>` as a lane option. Scan = sorted glob so the
/// resolution is deterministic.
fn resolve_mmproj(model: &Path, opt: Option<&Path>) -> Result<PathBuf, LaneError> {
    if let Some(p) = opt {
        let p = PathBuf::from(p);
        if !p.is_file() {
            return Err(LaneError::Load(format!(
                "mmproj lane option points to a missing file: {}",
                p.display()
            )));
        }
        return Ok(p);
    }
    let dir = model
        .parent()
        .ok_or_else(|| LaneError::Load("backbone gguf has no parent dir".into()))?;
    let mut cands: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| LaneError::Load(format!("scan {}: {e}", dir.display())))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|f| {
                    let n = f.to_string_lossy().to_lowercase();
                    n.starts_with("mmproj") && n.ends_with(".gguf")
                })
                .unwrap_or(false)
        })
        .collect();
    cands.sort();
    match cands.len() {
        1 => Ok(cands.remove(0)),
        0 => Err(LaneError::Load(format!(
            "no mmproj-*.gguf beside {} — place the mmproj twin beside the \
             backbone (or pass lane-option mmproj=<path>)",
            model.display()
        ))),
        n => Err(LaneError::Load(format!(
            "{n} mmproj-*.gguf beside {} — ambiguous (pass lane-option \
             mmproj=<path> to choose): {cands:?}",
            model.display()
        ))),
    }
}

impl crate::types::TtsLane for GgmlFfiLane {
    fn lane_name(&self) -> &'static str {
        "ggml-ffi"
    }

    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError> {
        let t0 = Instant::now();
        let backbone = match model {
            ModelRef::Path(p) => p,
            ModelRef::Hub { id, .. } => {
                return Err(LaneError::Load(format!(
                    "ggml-ffi lane requires a disk gguf pair, got hub model '{id}' \
                     (pass the chatterbox-turbo backbone gguf path)"
                )));
            }
            ModelRef::Pseudo(id) => {
                return Err(LaneError::Load(format!(
                    "ggml-ffi lane requires a disk gguf pair, got pseudo model '{id}'"
                )));
            }
        };
        if !backbone.is_file() {
            return Err(LaneError::Load(format!(
                "model file not found: {} (pass the chatterbox-turbo backbone gguf)",
                backbone.display()
            )));
        }
        // mmproj twin: explicit (pre-set) or the single mmproj-*.gguf beside
        // the backbone in the same directory.
        let mmproj = resolve_mmproj(backbone, self.mmproj_path.as_deref())?;
        self.mmproj_path = Some(mmproj);

        let threads: i32 = std::env::var("TTS_CBX_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_THREADS);

        let model_c = cstr(backbone)?;
        let mmproj_c = cstr(self.mmproj_path.as_ref().expect("checked"))?;
        let sess = unsafe {
            tts_cbx_init(
                model_c.as_ptr(),
                mmproj_c.as_ptr(),
                threads,
                DEFAULT_N_BATCH,
                DEFAULT_N_PREDICT_CAP,
            )
        };
        let sr = unsafe { tts_cbx_sample_rate(sess) };
        if unsafe { tts_cbx_ok(sess) } == 0 || sr <= 0 {
            let err = unsafe {
                let p = tts_cbx_last_error(sess);
                if p.is_null() {
                    "unknown shim error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            unsafe { tts_cbx_free(sess) };
            return Err(LaneError::Load(format!("chatterbox init failed: {err}")));
        }
        let rt = CbxRuntime {
            sess: CbxHandle(sess),
            sample_rate: sr as u32,
            shipped_voice: unsafe {
                let p = tts_cbx_voice_name(sess);
                if p.is_null() {
                    String::new()
                } else {
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            },
        };
        let load_ms = t0.elapsed().as_millis() as u64;

        self.runtime = Some(Arc::new(rt));
        self.model_path = Some(backbone.to_path_buf());
        self.model_label = backbone
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| backbone.display().to_string());
        self.model_sha = crate::lanes::model_sha16(backbone);
        self.load_ms = load_ms;
        Ok(LoadInfo {
            load_ms: self.load_ms,
        })
    }

    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError> {
        if req.text.trim().is_empty() {
            return Err(LaneError::Synth("empty text".into()));
        }
        let rt = self
            .runtime
            .clone()
            .ok_or_else(|| LaneError::Synth("lane not loaded".into()))?;
        if rt.sample_rate == 0 {
            return Err(LaneError::Load("shim has no sample rate".into()));
        }

        let d = Self::resolve_dials(req)?;
        let top_k = d.top_k;
        let top_p = d.top_p;
        let min_p = d.min_p;
        let temp = d.temp;
        let penalty_repeat = d.penalty_repeat;
        let max_frames = d.max_frames;
        // max-frames stays an init-time cap in the shim; expose it as a
        // hard error so per-request changes are explicit (init cap is 512,
        // the llama-tts default).
        if max_frames.is_some() {
            return Err(LaneError::Synth(
                "ggml-ffi lane option max-frames is set at load (shim init cap \
                 512, the llama-tts default); per-request override unsupported"
                    .into(),
            ));
        }

        // Voice is resolved PER CALL (seam law, matches every other lane):
        // `Path` installs a cloning reference in the shim; `Default`/`Id`
        // restore the shipped pack (otherwise the cloning bitmap would leak
        // into later requests of a resident process). Turbo ships exactly
        // one voice ("default"); unknown names fail honestly.
        let voice_ref = req.voice.clone().unwrap_or(VoiceRef::Default);
        let voice_label = match &voice_ref {
            VoiceRef::Path(p) => {
                let wav = cstr(p)?;
                let rc = unsafe { tts_cbx_set_speaker_wav(rt.sess.0, wav.as_ptr()) };
                if rc != 0 {
                    return Err(LaneError::Load(format!(
                        "speaker wav load failed: {} (error: {}, reference must be \
                         > 5 s audio; 10–15 s recommended)",
                        p.display(),
                        rt.error().unwrap_or_else(|| "unknown".into())
                    )));
                }
                p.file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string())
            }
            _ => {
                let rc = unsafe { tts_cbx_reset_speaker(rt.sess.0) };
                if rc != 0 {
                    return Err(LaneError::Synth(format!(
                        "shipped-voice restore failed: {}",
                        rt.error().unwrap_or_else(|| "unknown".into())
                    )));
                }
                match &voice_ref {
                    VoiceRef::Id(id) if !id.is_empty() => id.clone(),
                    _ => rt.shipped_voice.clone(),
                }
            }
        };

        // Chunks: seam-provided (T013), else one chunk over the whole text.
        let chunks_in: Vec<crate::types::TextChunk> = if req.chunks.is_empty() {
            vec![crate::types::TextChunk {
                text: req.text.clone(),
                byte_range: [0, req.text.len() as u32],
            }]
        } else {
            req.chunks.clone()
        };

        let meta = RunMeta {
            lane: "ggml-ffi",
            model: self.model_label.clone(),
            model_sha: self.model_sha.clone(),
            voice: voice_label,
            sample_rate: rt.sample_rate,
            lang: req.lang,
            determinism: match req.seed {
                Some(_) => Determinism::Seeded,
                None => Determinism::NonDeterministic(
                    "unseeded ggml-ffi run (sampler + ODE draws random) — pass \
                     --seed for byte-identity"
                        .to_string(),
                ),
            },
            seed: req.seed,
            lane_version: env!("CARGO_PKG_VERSION").to_string(),
            runtime: "llama.cpp-ffi (chatterbox-turbo, pinned 186eab1)".to_string(),
            load_ms: self.load_ms,
            // Grain = one T013 text chunk (what `tts serve` and the relay
            // parity need; llama set_input = fresh KV per chunk, DLV-1).
            chunk_policy: "text-chunks(T013)".to_string(),
            restart_voice: false,
        };

        // Per-stream synthesis state (the `SynthStream` iterator's captures).
        struct ChunkState {
            rt: Arc<CbxRuntime>,
            chunk_iter: std::vec::IntoIter<crate::types::TextChunk>,
            seed: u64,
            top_k: i32,
            top_p: f32,
            min_p: f32,
            temp: f32,
            penalty_repeat: f32,
            index: u32,
        }

        /// One chunk of synthesis. Returns `Ok(None)` at stream end; the
        /// iterator closure just `.transpose()`s this into its
        /// `Option<Result<_, _>>` shape.
        fn next_chunk(st: &mut ChunkState) -> Result<Option<ChunkOut>, LaneError> {
            let chunk = match st.chunk_iter.next() {
                Some(c) => c,
                None => return Ok(None),
            };
            // ---- g2p: none on this lane (shim owns the text path) ----
            let g2p = std::time::Duration::ZERO;

            // ---- acoustic: prompt + generation (fused stages 1–2) ----
            let text_c = std::ffi::CString::new(chunk.text.as_str())
                .map_err(|e| LaneError::Synth(format!("chunk text NUL: {e}")))?;
            let mut sample_rate: i32 = 0;
            let mut wav: *const std::ffi::c_char = std::ptr::null();
            let mut wav_len: usize = 0;
            let mut samples: i64 = 0;
            let mut frames: i32 = 0;
            let mut prompt_s: f64 = 0.0;
            let mut gen_s: f64 = 0.0;
            let mut wav_s: f64 = 0.0;
            let sess = st.rt.sess.0;
            let rc = unsafe {
                tts_cbx_synthesize(
                    sess,
                    text_c.as_ptr(),
                    st.seed,
                    st.top_k,
                    st.top_p,
                    st.min_p,
                    st.temp,
                    st.penalty_repeat,
                    &mut sample_rate,
                    &mut wav,
                    &mut wav_len,
                    &mut samples,
                    &mut frames,
                    &mut prompt_s,
                    &mut gen_s,
                    &mut wav_s,
                )
            };
            if rc != 0 {
                return Err(LaneError::Synth(format!(
                    "chatterbox synth failed: {}",
                    st.rt
                        .error()
                        .unwrap_or_else(|| "unknown shim error".to_string())
                )));
            }
            let acoustic = std::time::Duration::from_secs_f64(prompt_s + gen_s);

            // ---- vocoder (code2wav/hift, timed inside the shim) ----
            let vocoder = std::time::Duration::from_secs_f64(wav_s);

            // ---- io: WAV buffer → i16 (DLV-2: copy before next set_input) ----
            let t_i = Instant::now();
            if wav.is_null() || wav_len == 0 || sample_rate <= 0 {
                return Err(LaneError::Synth(
                    "chatterbox pipeline returned empty audio".into(),
                ));
            }
            let bytes = unsafe { std::slice::from_raw_parts(wav as *const u8, wav_len) };
            let (wrate, pcm) = crate::wav::parse_wav_buffer_mono_i16(bytes)
                .map_err(|e| LaneError::Synth(format!("wav parse: {e}")))?;
            if wrate != sample_rate as u32 {
                return Err(LaneError::Synth(format!(
                    "chatterbox wav header rate {wrate} != pipeline rate {sample_rate}"
                )));
            }
            let io = t_i.elapsed();

            // Law mirror (kokoro): empty token stream → 10 ms honest chunk.
            let pcm = if pcm.is_empty() {
                vec![0i16; sample_rate as usize / 100]
            } else {
                pcm
            };
            // Header-truth cross-check (n_samples is straight from the WAV
            // header; the parser uses the same data chunk).
            if samples > 0 && samples as usize != pcm.len() {
                return Err(LaneError::Synth(format!(
                    "chatterbox wav mismatch: pipeline reported {samples} samples, \
                     parsed {}",
                    pcm.len()
                )));
            }

            let out = ChunkOut {
                index: st.index,
                stages: StageTimings {
                    g2p,
                    acoustic,
                    vocoder,
                    io,
                },
                pcm,
            };
            st.index += 1;
            Ok(Some(out))
        }

        // Seed: per-call (shim rebuilds the sampler per chunk — DLV-1).
        // Unseeded: entropy per CHUNK (each chunk independent, so the
        // identity across runs is meaningless without --seed — as on the
        // flow lanes).
        let seed = req.seed.unwrap_or_else(rand_seed);
        let mut state = ChunkState {
            rt,
            chunk_iter: chunks_in.into_iter(),
            seed,
            top_k,
            top_p,
            min_p,
            temp,
            penalty_repeat,
            index: 0,
        };

        let iter = std::iter::from_fn(move || next_chunk(&mut state).transpose());

        Ok(SynthStream {
            meta,
            chunks: Box::new(iter),
        })
    }
}

/// Cheap local entropy (matches piper lane's unseeded convention).
fn rand_seed() -> u64 {
    // std-based fallback: no external rng dep on this feature.
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LaneOptions, TtsLane};

    fn opts(pairs: &[(&str, &str)]) -> LaneOptions {
        LaneOptions {
            chunk_policy: None,
            extra: pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn req(opts_: LaneOptions) -> SynthRequest {
        SynthRequest {
            text: "hello".into(),
            chunks: Vec::new(),
            lang: crate::types::Lang::En,
            voice: Some(VoiceRef::Default),
            seed: Some(42),
            lane_options: opts_,
        }
    }

    #[test]
    fn dials_default_to_llama_tts_law() {
        let d = GgmlFfiLane::resolve_dials(&req(Default::default())).expect("defaults");
        assert_eq!(SAMPLING_TOP_K, d.top_k, "top_k");
        assert_eq!(SAMPLING_TOP_P, d.top_p, "top_p");
        assert_eq!(SAMPLING_MIN_P, d.min_p, "min_p");
        assert_eq!(SAMPLING_TEMP, d.temp, "temp");
        assert_eq!(SAMPLING_PENALTY_REPEAT, d.penalty_repeat, "repeat");
        assert_eq!(d.max_frames, None, "no cap override");
    }

    #[test]
    fn dial_overrides_parse() {
        let r = req(opts(&[
            ("top-k", "64"),
            ("top-p", "0.9"),
            ("temp", "0.5"),
            ("penalty-repeat", "1.1"),
        ]));
        let d = GgmlFfiLane::resolve_dials(&r).expect("overrides");
        assert_eq!(d.top_k, 64);
        assert_eq!(d.top_p, 0.9);
        assert_eq!(d.temp, 0.5);
        assert_eq!(d.penalty_repeat, 1.1);
        assert_eq!(d.max_frames, None);
    }

    #[test]
    fn unknown_dial_is_hard_error() {
        let e = GgmlFfiLane::resolve_dials(&req(opts(&[("nope", "1")]))).unwrap_err();
        assert!(e.to_string().contains("not a known ggml-ffi dial"));
    }

    #[test]
    fn out_of_range_dial_is_hard_error() {
        for (k, v) in [
            ("top-k", "0"),
            ("top-p", "1.5"),
            ("temp", "-1"),
            ("penalty-repeat", "0.5"),
            ("max-frames", "0"),
        ] {
            let e = GgmlFfiLane::resolve_dials(&req(opts(&[(k, v)]))).unwrap_err();
            assert!(e.to_string().contains("out of range"), "{k}={v}: {e}");
        }
    }

    #[test]
    fn lane_name_matches_registry() {
        assert_eq!(GgmlFfiLane::new().lane_name(), "ggml-ffi");
    }
}
