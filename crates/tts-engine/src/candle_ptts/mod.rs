//! Candle Phonon-31m port (T008). Oracle: ptts-0.6.0 (cached at
//! `~/.cache/tts-engine-ref/ptts-0.6.0`). Plan: `docs/plan/tasks/T008-candle-phonon-port.md`.
//!
//! Session-1 surface: `cfg` (config.json, faithful serde port) and `loader`
//! (remap table + mmap inventory). The transformer / flow head / EOS head /
//! Mimi decode arrive in later sessions; the `flow-candle` lane fails cleanly
//! until then. Decode-policy law: exact port, nothing "improved" silently —
//! deviations are experiments on top, labeled as such.
pub mod cfg;
pub mod loader;
