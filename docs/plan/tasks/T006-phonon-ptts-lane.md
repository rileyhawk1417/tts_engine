# T006 — Phonon `ptts`-crate lane behind the seam (flow-latent step a)

- **Status:** `not-started`
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md)

## Goal

Phonon-31m generating through our seam via Gradium's first-party `ptts` crate (MIT OR Apache-2.0): fast path to streaming + telemetry, and the correctness oracle for the from-scratch port (T008).

## Approach

- `lanes/flow-ptts/`: wrap `ptts::Checkpoint::resolve` (local path under `weights/`, hf only when explicitly allowed), `builder(Lang::X).device(Cpu)`; `tts.stream(text)` → map PCM chunks to seam chunks; `say_with(SpeechOptions)` for non-stream bench runs.
- Sessions: `session_default(&options)` / `session(&options, max_seq_len)` mapping = our warm-pool; telemetry captures session reuse (voice-primed KV) vs cold as TTFA delta rows.
- chunk callbacks: chunk boundary events → chunks.jsonl per-chunk rows (voice restart per chunk per Phonon policy).
- First clear the verify-list blockers (targets.md): cargo dep weight of `ptts` (ggml pulled?), actual LICENSE files, phonon-40m gate irrelevance (31m only).
- Weights off disk: q8 GGUF (`model.q8.gguf`, 64.5 MB) default; f32 variant noted for later comparison.

## Done-conditions

- [ ] `tts synth --lane flow-ptts ...` streams intelligible audio, chunks.jsonl complete, per-chunk TTFA visible
- [ ] Determinism gate passes (fixed seed) — two runs byte-identical
- [ ] Verify-list items resolved and recorded in targets.md (no dangling license unknowns for a shipped lane)
