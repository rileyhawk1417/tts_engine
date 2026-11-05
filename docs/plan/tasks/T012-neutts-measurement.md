# T012 — NeuTTS honest measurement experiment (measurement-only lane)

- **Status:** `not-started`
- **Milestone:** [M4](../index.md#milestone-m4--streaming--measurement-lanes)
- **Depends on:** [T003](T003-roundtrip-eval-harness.md)

## Goal

Settle the disputed CPU-realtime claim with our own numbers — or falsify fast and shelve. NeuTTS Air Q4/Q8 GGUF (Apache-2.0, qwen2 arch) has no codec decode path integrated anywhere; that integration cost is part of the verdict.

## Approach

- Phase 1 (cheap falsification): run the LM backbone (qwen2 GGUF ~748 MB Q4_0) via a llama.cpp-class harness on this box; measure tokens/s and load wall — if backbone alone cannot sustain 12.5 Hz-class frame rate needed for codec decode, record falsification and stop.
- Phase 2 (only if phase 1 passes): NeuCodec decode path — check for any ONNX/ggml decode artifact; if missing, estimate ONNX-export effort; time-box it, else shelve.
- Compare against reported community complaints (~15× slower than realtime, ~3 min load, 74 s voice encode) — our numbers either confirm, refute, or sit in-between with hardware context.
- Verdict in the scoreboard + memory checkpoint: shelve-with-reason is a fully valid outcome (te discipline).

## Done-conditions

- [ ] Phase-1 measured numbers recorded (tokens/s, load wall, RSS) regardless of outcome
- [ ] Explicit adopt/shelve verdict with reason in scoreboard + memory checkpoint
- [ ]targets.md measurement-only lane row updated to reflect the verdict
