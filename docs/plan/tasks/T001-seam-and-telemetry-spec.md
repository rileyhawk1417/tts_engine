# T001 — Seam + telemetry spec

- **Status:** `completed`
- **Milestone:** [M0](../index.md#milestone-m0--foundation)
- **Depends on:** none
- **Deliverable:** [../../seam.md](../../seam.md)

## Goal

One written contract every lane must satisfy, so lanes are swappable and every run is measurable: CLI surface, chunk protocol, telemetry schema, output conventions. Deliverable: `docs/seam.md` (design note — no status).

## Approach

- CLI: `tts synth --lane <name> --model <path-or-hf-id> --lang <code> --voice <id> --text <str|@file> -o out.wav [--jsonl chunks.jsonl] [--bench]`.
- Output: mono WAV (24 kHz default; lane-declared sample rate recorded in chunks.jsonl header row).
- chunks.jsonl schema: one header row (`lane, model, voice, sample_rate`) then one row per audio chunk: `{i, t_synth_start, chunk_ms_by_stage {g2p, acoustic, vocoder, io}, ttfa_ms, rtf_running, peak_rss_kb, pcm_bytes_sha}`.
- TTFA definition: first PCM chunk flushed → wall clock since synth call (post-load). RTF: audio seconds / wall seconds, whole utterance. Peak RSS via `/proc/self/status` sampler thread.
- Env conventions: `HF_HOME=/teranux/hugging_face`, weights always resolved to a local path first (mmap-first mandate), HF fetch only when explicitly allowed.
- Library seam mirrors CLI: `trait TtsLane { fn synthesize(&self, req: SynthRequest) -> Result<SynthStream; }` — chunk iterator + final metrics struct.
- Streaming IPC is out of scope for M0 (in-process chunks only) — record as open question owner.

## Done-conditions

- [ ] `docs/seam.md` exists, reviewed, covers CLI, JSONL schema, metrics defs (TTFA/RTF/RSS), env, trait sketch
- [ ] At least one worked example chunks.jsonl document (hand-written) serves as golden reference for T002/T004 parsers
