pub mod smoke;

use crate::types::{ModelRef, TtsLane};

/// Lane feature registry. Each feature-gated lane registers here:
/// `graph-ort` (T004, `graph-ort` feature), `flow-ptts` (T006), `flow-candle`
/// (T008), `ggml-ffi` (T010), `streaming-graph` (T011). Unknown lane names
/// are a CLI usage error, not a runtime panic.
pub fn make_lane(name: &str) -> Option<Box<dyn TtsLane>> {
    match name {
        "smoke" => Some(Box::new(smoke::SmokeLane::new())),
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
