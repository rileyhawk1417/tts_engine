# T002 — Rust workspace scaffold + build gates

- **Status:** `completed` (worktree, 2026-11-05)
- **Milestone:** [M0](../index.md#milestone-m0--foundation)
- **Depends on:** [T001](T001-seam-and-telemetry-spec.md) (the CLI/trait surface it implements)
- **Deliverable:** `crates/tts-engine` + `crates/tts` + `scripts/build.sh` + `eval/fixtures/golden-chunks.jsonl`

## Goal

Compiling workspace skeleton where every future lane is a feature-gated module behind the seam, with build scripts and lint gates mirroring `te` conventions.

## Approach

- Workspace crates: `crates/tts-engine` (lib: `lane` trait, `SynthRequest/Chunk/Metrics`, telemetry sampler, chunking module placeholder), `crates/tts` (bin CLI per seam spec), lanes as features (`graph-ort`, `flow-ptts`, `flow-candle`, `ggml-ffi`, `streaming-graph`) each in its own module dir `lanes/<name>/`.
- `scripts/build.sh`: fmt + clippy -D warnings + build + `--bench` smoke (hello lane placeholder that emits silence through the seam and writes chunks.jsonl — proves telemetry end-to-end with zero model deps).
- Version pinning policy like te: lock `ort` to a working ONNX Runtime build (api-21), record `target-cpu` decision (te falsified `native`; default plain, revisit per-lane).
- `.gitignore`: `weights/`, `out/`, fixtures keep small golden WAVs only.
- Keep hello-world `src/main.rs` replaced by the workspace layout above.

## Done-conditions

- [x] `cargo build --workspace` green; `scripts/build.sh` runs fmt+clippy+tests+build clean
- [x] Placeholder lane emits valid WAV + chunks.jsonl conforming to seam spec (golden fixture checked in, `crates/tts-engine/tests/golden_fixture.rs` re-asserts §5.1 on it)
- [x] Telemetry fields verified non-zero/credible on the smoke lane (load_ms 5, peak_rss ~4.4 MB, rtf non-zero; ttfa honestly 0 ms — silence flushes instantly)

## Notes from build

- `ModelRef` gained a third variant `Pseudo(String)` for diagnostic/stub lanes (`--lane smoke --model smoke`); real lanes must reject it. `ModelRef` also lost `PartialEq/Eq` (PathBuf eq not needed) — derive is `Debug, Clone`.
- Contract fix vs T001 text: `rtf_running` (cumulative audio/wall) is NOT contractually monotone — dips are legal; seam §5.1 + golden fixture (4th chunk dips 5.155→5.148) updated. Hard invariants: indices, TTFA-once, `rtf_final == last rtf_running`, finite positive values, constant `t_synth_start_ms`.
- `t_synth_start_ms` captured once at `RunRecorder::begin`, anchored at first line of `main` (was recomputed per row — caught live).
- `pcm_bytes` rows carry the `sha256:` prefix; RSS sampler takes one synchronous sample at first use so first-chunk `peak_rss_kb` ≠ 0.
- ort pin + `target-cpu` profile decision land with the `graph-ort` feature (T004), mirroring te (`ort 2.0.0-rc.15`, `load-dynamic` + `api-21`, plain target-cpu).
