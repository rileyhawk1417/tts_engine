//! tts-engine — the seam library.
//!
//! Types and services defined by `docs/seam.md`: the `TtsLane` contract,
//! `SynthRequest`/`SynthStream`/`ChunkOut`, per-lane text chunking, the
//! telemetry recorder that emits `chunks.jsonl`, and the canonical WAV writer.
//! Lanes live behind the `TtsLane` trait; telemetry never lives lane-side.

pub mod chunking;
pub mod lanes;
#[cfg(feature = "flow-candle")]
pub mod candle_ptts;
pub mod telemetry;
pub mod types;
pub mod wav;
