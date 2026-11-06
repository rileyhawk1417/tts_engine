pub mod smoke;

#[cfg(feature = "graph-ort")]
pub mod kokoro;

#[cfg(feature = "flow-ptts")]
pub mod flow_ptts;

#[cfg(feature = "flow-candle")]
pub mod flow_candle;

#[cfg(feature = "piper")]
pub mod piper;

#[cfg(feature = "ggml-ffi")]
pub mod ggml_ffi;

#[cfg_attr(
    all(not(feature = "piper"), not(feature = "graph-ort")),
    allow(unused_imports)
)]
use crate::types::LaneError;
use crate::types::{ModelRef, TtsLane};

#[cfg(any(feature = "graph-ort", feature = "piper"))]
pub(crate) const DEFAULT_ORT_LIB: &str = "/teranux/ort-1.22.1/libonnxruntime.so";

/// The process-wide ORT api/environment — ONE per process (ort 2.0-rc,
/// load-dynamic: the dylib is initialized once). Hoisted here from the
/// kokoro lane (T004) so both ort-graph lanes (kokoro T004, piper T011)
/// share it instead of racing competing envs. Dial law: `TTS_ORT_DYLIB` →
/// `ORT_DYLIB_PATH` → `DEFAULT_ORT_LIB` if it exists → `libonnxruntime.so`
/// (host linker search; onnxruntime 1.22.x — `pip install
/// onnxruntime==1.22.1` drops the matching dylib).
#[cfg(any(feature = "graph-ort", feature = "piper"))]
pub(crate) fn ort_env() -> Result<&'static ort::environment::Environment, LaneError> {
    static ORT_ENV: std::sync::OnceLock<Result<ort::environment::Environment, String>> =
        std::sync::OnceLock::new();
    match ORT_ENV
        .get_or_init(|| {
            let dylib = std::env::var("TTS_ORT_DYLIB")
                .or_else(|_| std::env::var("ORT_DYLIB_PATH"))
                .unwrap_or_else(|_| {
                    if std::path::Path::new(DEFAULT_ORT_LIB).exists() {
                        DEFAULT_ORT_LIB.to_string()
                    } else {
                        "libonnxruntime.so".to_string()
                    }
                });
            ort::init_from(dylib)
                .map_err(|e| e.to_string())?
                .with_telemetry(false)
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
    {
        Ok(env) => Ok(env),
        Err(e) => Err(LaneError::Load(format!(
            "ort-graph lane (kokoro/piper) needs the ORT 1.22.x dylib: \
              pip install onnxruntime==1.22.1, or set TTS_ORT_DYLIB; init failed: {e}"
        ))),
    }
}

/// Map an `ort::Result` onto `LaneError` (every ort call site).
#[cfg(any(feature = "graph-ort", feature = "piper"))]
pub(crate) fn ort<T>(r: ort::Result<T>, what: &str) -> Result<T, LaneError> {
    r.map_err(|e| LaneError::Synth(format!("ort error in {what}: {e}")))
}

/// Lane-agnostic CPU dial default: logical/2 ≈ physical cores (SMT halves
/// give no compute; the te/STT law — threads=cores — transfers here).
#[cfg(any(feature = "graph-ort", feature = "piper", feature = "flow-ptts"))]
pub(crate) fn physical_core_estimate() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (logical / 2).max(1)
}

/// Lane feature registry. Each feature-gated lane registers here:
/// `graph-ort` (T004, `graph-ort` feature), `flow-ptts` (T006), `flow-candle`
/// (T008), `ggml-ffi` (T010, llama.cpp chatterbox FFI), `piper` (T011).
/// Unknown lane names are a CLI usage error, not a runtime panic; disabled
/// features fall
/// through to the same error (message notes the feature).
pub fn make_lane(name: &str) -> Option<Box<dyn TtsLane>> {
    match name {
        "smoke" => Some(Box::new(smoke::SmokeLane::new())),
        #[cfg(feature = "graph-ort")]
        "kokoro" => Some(Box::new(self::kokoro::KokoroLane::new())),
        #[cfg(feature = "flow-ptts")]
        "flow-ptts" => Some(Box::new(self::flow_ptts::FlowPttsLane::new())),
        #[cfg(feature = "flow-candle")]
        "flow-candle" => Some(Box::new(self::flow_candle::FlowCandleLane::new())),
        #[cfg(feature = "piper")]
        "piper" => Some(Box::new(self::piper::PiperLane::new())),
        #[cfg(feature = "ggml-ffi")]
        "ggml-ffi" => Some(Box::new(self::ggml_ffi::GgmlFfiLane::new())),
        _ => None,
    }
}

/// Seam law: `model_sha` = first 16 hex chars of the sha256 over the map'd
/// weight file(s). Streams the file (weights can be GBs).
pub fn model_sha16(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let digest = h.finalize();
    Some(digest.iter().take(8).map(|b| format!("{b:02x}")).collect())
}

impl ModelRef {
    /// Resolve the primary weight file for hashing (single-file lanes).
    pub fn primary_file(&self) -> Option<&std::path::Path> {
        match self {
            ModelRef::Path(p) => Some(p),
            ModelRef::Hub { .. } => None,
            ModelRef::Pseudo(_) => None,
        }
    }
}
