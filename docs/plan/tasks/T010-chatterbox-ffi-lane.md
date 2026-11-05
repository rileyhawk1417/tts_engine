# T010 — chatterbox.cpp FFI lane (ggml-ffi)

- **Status:** `not-started`
- **Milestone:** [M3](../index.md#milestone-m3--ggml-ffi-lane-cloning)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md)

## Goal

Voice cloning capability via the mature ggml port: `gianni-cor/chatterbox.cpp` (Turbo EN variant), behind our seam with FFI and the te-style off-disk GGUF mechanics.

## Approach

- Preflight (gate before code): verify the library exposes a stable C API (targets.md verify-list) — if C API is absent/unstable, subprocess wrapper is the fallback (te relay pattern) and this task records the swap.
- FFI wrapper crate `lanes/ggml-ffi/`: bind `ptts.h`-style C headers of chatterbox.cpp (Turbo: GPT-2-medium T3 + meanflow CFM + S3Gen + HiFT), GGUF Q4_0 T3 + f16 S3Gen per their validated recipe.
- Cloning path: voice embed extraction from reference WAV via the port's tooling; consent/provenance note recorded per bundled voice (targets.md license asymmetries rule).
- Telemetry: FFI boundary timings (inference vs decode vs resample) mapped into chunks.jsonl stages; chunked via T013; session/pool semantics documented (ggml contexts per thread, mirroring te session-per-thread).
- Round-trip WER gate like every lane; their CER 0.00 CI gates are the port's own discipline — we add hard-sentences corpus to it.

## Done-conditions

- [ ] `tts synth --lane ggml-ffi --voice <cloned> ...` yields recognized speech from a reference-WAV clone (round-trip WER reported)
- [ ] C-API (or subprocess) decision recorded with evidence in targets.md verify-list
- [ ] Cold-load wall + peak RSS logged; determinism gate for the no-cloning default voice path
