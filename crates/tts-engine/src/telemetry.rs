//! Telemetry — seam-side only (`docs/seam.md` §5, §6).
//!
//! Lanes report `StageTimings` data; this module serializes typed JSONL rows
//! (header / chunk / summary) to a sink, computes TTFA and running RTF, keeps
//! the 10 ms RSS sampler, and hashes per-chunk PCM for the byte-identity gate.

use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::types::{ChunkOut, Determinism, RunMeta};

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
}

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

fn round3(f: f64) -> f64 {
    (f * 1000.0).round() / 1000.0
}

pub fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    let digest = Sha256::digest(bytes.as_ref());
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        write!(s, "{b:02x}").ok();
    }
    s
}

/// sha256 over PCM as contiguous little-endian i16 bytes (canonical form for
/// the byte-identity gate; independent of host endianness).
pub fn pcm_sha(pcm: &[i16]) -> String {
    let mut h = Sha256::new();
    for s in pcm {
        h.update(s.to_le_bytes());
    }
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        write!(s, "{b:02x}").ok();
    }
    s
}

// ---------------------------------------------------------------------------
// RSS sampler (10 ms loop over /proc/self/status; te load-parity lineage)
// ---------------------------------------------------------------------------

pub struct Rss {
    #[allow(dead_code)] // current-kb feed is written by the sampler; read later by lanes
    current_kb: Arc<AtomicU64>,
    peak_kb: Arc<AtomicU64>,
}

fn vm_field(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest.split_whitespace().next()?.parse::<u64>().ok();
        }
    }
    None
}

fn rss() -> &'static Rss {
    static RSS: OnceLock<Rss> = OnceLock::new();
    RSS.get_or_init(|| {
        let current_kb = Arc::new(AtomicU64::new(0));
        let peak_kb = Arc::new(AtomicU64::new(0));
        // One immediate synchronous sample so the very first chunk flush sees
        // real VmRSS, not 0 (the sampler thread may not have ticked yet).
        if let Some(kb) = vm_field("VmRSS:") {
            current_kb.store(kb, Ordering::Relaxed);
            peak_kb.store(kb, Ordering::Relaxed);
        }
        let cur = Arc::clone(&current_kb);
        let peak = Arc::clone(&peak_kb);
        thread::Builder::new()
            .name("tts-rss-sampler".into())
            .spawn(move || {
                loop {
                    if let Some(kb) = vm_field("VmRSS:") {
                        cur.store(kb, Ordering::Relaxed);
                        let mut m = peak.load(Ordering::Relaxed);
                        while kb > m {
                            match peak.compare_exchange_weak(
                                m,
                                kb,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(x) => m = x,
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            })
            .ok();
        Rss {
            current_kb,
            peak_kb,
        }
    })
}

// ---------------------------------------------------------------------------
// Typed JSONL rows (serde field order == wire order)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct HeaderRow<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    lane: &'a str,
    model: &'a str,
    model_sha: Option<&'a str>,
    voice: &'a str,
    sample_rate: u32,
    lang: &'a str,
    determinism: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    nondet_reason: Option<&'a str>,
    seed: Option<u64>,
    lane_version: &'a str,
    runtime: &'a str,
    load_ms: u64,
    chunk_policy: &'a str,
    restart_voice: bool,
}

#[derive(serde::Serialize)]
struct ChunkMs {
    g2p: u64,
    acoustic: u64,
    vocoder: u64,
    io: u64,
}

#[derive(serde::Serialize)]
struct ChunkRow<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    i: u32,
    t_synth_start_ms: u64,
    chunk_ms: ChunkMs,
    ttfa_ms: Option<u64>,
    rtf_running: f64,
    peak_rss_kb: u64,
    pcm_bytes: &'a str,
}

#[derive(serde::Serialize)]
struct GateRow<'a> {
    byte_identity: &'a str,
    run_id: &'a str,
}

#[derive(serde::Serialize)]
struct SummaryRow<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    chunks: u32,
    audio_seconds: f64,
    wall_ms: u64,
    rtf_final: Option<f64>,
    ttfa_ms: Option<u64>,
    peak_rss_kb_max: u64,
    seed: Option<u64>,
    gate: GateRow<'a>,
}

/// Aggregates returned to the caller (CLI prints these; eval harness stores
/// them in `results.json`).
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub audio_seconds: f64,
    pub wall_ms: u64,
    pub rtf_final: Option<f64>,
    pub ttfa_ms: Option<u64>,
    pub peak_rss_kb: u64,
    pub chunks: u32,
}

/// Drives a run's telemetry: wraps a sink, writes rows as chunks flow through.
///
/// `gate` is `"not-assessed"` on live runs; the T003 harness rewrites it to
/// `"pass"`/`"fail"` after comparing two same-seed runs.
pub struct RunRecorder<'w> {
    w: Box<dyn Write + 'w>,
    meta: RunMeta,
    synth_start: Instant,
    /// Wall ms since process start when `synthesize()` was called — §5.1 law:
    /// constant across all chunk rows of a run.
    t_synth_start_ms: u64,
    first_flush_done: bool,
    first_flush_wall_ms: Option<u64>,
    audio_samples: u64,
    n_chunks: u32,
    last_rtf: Option<f64>,
    last_flush_wall_ms: u64,
    run_id: String,
    gate: String,
}

impl<'w> RunRecorder<'w> {
    /// Begin a run: emit the header row.
    pub fn begin(
        meta: &RunMeta,
        process_start: Instant,
        synth_start: Instant,
        sink: Box<dyn Write + 'w>,
        run_id: String,
        gate: &str,
    ) -> io::Result<Self> {
        let model_sha = meta.model_sha.as_deref();
        let nondet_reason = match &meta.determinism {
            Determinism::NonDeterministic(r) => Some(r.as_str()),
            _ => None,
        };
        let row = HeaderRow {
            kind: "header",
            lane: meta.lane,
            model: &meta.model,
            model_sha,
            voice: &meta.voice,
            sample_rate: meta.sample_rate,
            lang: meta.lang.code(),
            determinism: meta.determinism.as_str(),
            nondet_reason,
            seed: meta.seed,
            lane_version: &meta.lane_version,
            runtime: &meta.runtime,
            load_ms: meta.load_ms,
            chunk_policy: &meta.chunk_policy,
            restart_voice: meta.restart_voice,
        };
        let mut w = sink;
        serde_json::to_writer(&mut w, &row)?;
        writeln!(w)?;
        Ok(Self {
            w,
            meta: meta.clone(),
            synth_start,
            t_synth_start_ms: process_start.elapsed().as_millis() as u64,
            first_flush_done: false,
            first_flush_wall_ms: None,
            audio_samples: 0,
            n_chunks: 0,
            last_rtf: None,
            last_flush_wall_ms: 0,
            run_id,
            gate: gate.to_string(),
        })
    }

    /// Emit a chunk row and update running aggregates.
    pub fn on_chunk(&mut self, chunk: &ChunkOut) -> io::Result<()> {
        let wall = self.synth_start.elapsed();
        let wall_ms = wall.as_millis() as u64;
        let wall_sec = wall.as_secs_f64();
        let audio_samples = self.audio_samples + chunk.pcm.len() as u64;
        let audio_sec = if self.meta.sample_rate > 0 {
            audio_samples as f64 / self.meta.sample_rate as f64
        } else {
            0.0
        };
        let rtf = if wall_sec > 0.0 && audio_sec > 0.0 {
            round3(audio_sec / wall_sec)
        } else {
            0.0
        };
        let ttfa = if self.first_flush_done {
            None
        } else {
            Some(wall_ms)
        };
        if self.first_flush_wall_ms.is_none() {
            self.first_flush_wall_ms = Some(wall_ms);
        }
        self.first_flush_done = true;
        let peak = rss().peak_kb.load(Ordering::Relaxed);
        let hex = format!("sha256:{}", pcm_sha(&chunk.pcm));
        let row = ChunkRow {
            kind: "chunk",
            i: chunk.index,
            t_synth_start_ms: self.t_synth_start_ms,
            chunk_ms: ChunkMs {
                g2p: ms(chunk.stages.g2p),
                acoustic: ms(chunk.stages.acoustic),
                vocoder: ms(chunk.stages.vocoder),
                io: ms(chunk.stages.io),
            },
            ttfa_ms: ttfa,
            rtf_running: rtf,
            peak_rss_kb: peak,
            pcm_bytes: &hex,
        };
        serde_json::to_writer(&mut self.w, &row)?;
        writeln!(self.w)?;
        self.audio_samples = audio_samples;
        self.n_chunks += 1;
        self.last_rtf = Some(rtf);
        self.last_flush_wall_ms = wall_ms;
        Ok(())
    }

    /// Emit the summary row and return final aggregates.
    ///
    /// Invariant: `rtf_final` equals the last `rtf_running` (frozen at the
    /// last chunk flush), while `wall_ms` is the true synth wall to finish.
    pub fn finish(&mut self) -> io::Result<RunStats> {
        let wall_ms = self.synth_start.elapsed().as_millis() as u64;
        let audio_seconds = if self.meta.sample_rate > 0 {
            round3(self.audio_samples as f64 / self.meta.sample_rate as f64)
        } else {
            0.0
        };
        let peak = rss().peak_kb.load(Ordering::Relaxed);
        let row = SummaryRow {
            kind: "summary",
            chunks: self.n_chunks,
            audio_seconds,
            wall_ms,
            rtf_final: self.last_rtf,
            ttfa_ms: if self.n_chunks > 0 {
                self.first_ttfa()
            } else {
                None
            },
            peak_rss_kb_max: peak,
            seed: self.meta.seed,
            gate: GateRow {
                byte_identity: &self.gate,
                run_id: &self.run_id,
            },
        };
        serde_json::to_writer(&mut self.w, &row)?;
        writeln!(self.w)?;
        self.w.flush()?;
        Ok(RunStats {
            audio_seconds,
            wall_ms,
            rtf_final: self.last_rtf,
            ttfa_ms: if self.n_chunks > 0 {
                self.first_ttfa()
            } else {
                None
            },
            peak_rss_kb: peak,
            chunks: self.n_chunks,
        })
    }

    fn first_ttfa(&self) -> Option<u64> {
        self.first_flush_wall_ms
    }
}
