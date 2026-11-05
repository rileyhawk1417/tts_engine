# T004 — Kokoro-ORT lane end-to-end (graph-ort)

- **Status:** `not-started`
- **Milestone:** [M1](../index.md#milestone-m1--default-lane-graph-ort--kokoro)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md), [T013](T013-longtext-chunking-module.md)

## Goal

First real lane: Kokoro-82M int8 ONNX through `ort`, fully behind the seam with telemetry — the proven cheapest path to a judged lane. Reference checkpoint: `hexgrad/Kokoro-82M` (targets.md).

## Approach

- ONNX Runtime via `ort`, pinned build (api-21 per te gotchas; pin `libonnxruntime.so` for byte-repro); int8 model with external-data → mmap.
- Session-per-thread (`Session::run(&mut self)` constraint → session pool keyed to worker threads; int8 wins at m=1 — TTS decode is m=1, so int8 first).
- G2P: spike decision espeak-ng (proven, GPL-linking caveat → dynamic-load or subprocess boundary) vs misaki (Rust-impl maturity unknown) — measure WER + prosody impact on corpus, record the decision in targets.md verify-list, don't hand-wave it.
- Pipeline: G2P → StyleTTS2 encoder/plbert int8 → iSTFTNet vocoder, chunked through T013 module, per-stage timings into chunks.jsonl.
- Voice handling: reference-style style vectors from bundled voicebinaries; voice id resolved from lane config.

## Done-conditions

- [ ] `tts synth --lane graph-ort --model kokoro-82m ... -o out.wav` produces intelligible speech from a corpus sentence (spot-listen + ASR sanity)
- [ ] chunks.jsonl shows credible per-stage breakdown (g2p/acoustic/vocoder ms) and non-trivial TTFA
- [ ] Two cold runs byte-identical (determinism gate) before any numbers are quoted
