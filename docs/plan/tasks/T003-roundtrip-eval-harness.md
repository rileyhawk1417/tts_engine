# T003 — Round-trip WER eval harness

- **Status:** `completed` (worktree, 2026-11-05)
- **Milestone:** [M0](../index.md#milestone-m0--foundation)
- **Depends on:** [T002](T002-workspace-scaffold.md)
- **Deliverable:** `eval/run.sh` + `eval/corpus.toml` + `eval/wer.py` + `eval/import_hard.py`

## Goal

The adoption gate: any lane's output is transcribed by `transcription_engine`, scored against source text, and reported alongside stage timings. No lane ships without a measured round-trip WER.

## Approach

- Corpus manifest `eval/corpus.toml`: internal bench sentences (short/medium/long + numbers/dates/addresses) + import step for `gradium/tts-hard-sentences-202610` (Hugging Face dataset, per-language split).
- Judge: shell out to the `te` binary, lane = nemo int8 default, parakeet q8_0 as tiebreak; record which judge lane + its version in every report row. WER convention = te repo's (`scripts/wer.py`) — same normalization, byte-identity checked.
- Determinism: synth under seed/temp=0 or lane-declared determinism; repeated runs must be byte-identical (gate) before WER is trusted — mirrors te load-parity discipline.
- Output: `eval/results/<date>-<lane>-<corpus>.json` { per-sentence WER, mean WER, TTFA mean/p95, RTF, peak RSS } + human-readable scoreboard.md table appended (design note only).
- Judge load parity: warm the te model once per eval batch; wall includes load separately logged.

## Done-conditions

- [x] `eval/run.sh --lane smoke --corpus internal` produces valid `rows.tsv` + `results.json` + scoreboard.md row; silence WER = expected garbage 100% (mechanics proven, not meaning)
- [x] Byte-identity gate implemented (`cmp` on the two same-seed wavs per sentence) and passing on all 8 smoke sentences -> verdict "trusted"
- [x] Hard-sentences dataset imported: card 2500 rows, EN subset 500 -> `eval/data/hard-sentences-en.jsonl` + `.meta.json` counts sidecar (counts match dataset card)

## Build notes

- Judge wiring: `te transcribe --nemo --out-dir <workdir> <wav>` — required rebuilding te with `TE_FEATURES=nemo,dnnl ./scripts/build.sh build --release` (the existing target/release/te had been built bare, no nemo).
- `eval/wer.py` is a byte-identical copy of te's `scripts/wer.py` (sha256 b15eb6f060004e78…); normalization/opcodes identical — re-compare after any te change.
- Judge latency is deliberately NOT recorded in our rows (te's number, not ours). OUR TTFA/RTF/RSS come from chunksA.jsonl.
- Silence -> empty transcript -> WER 100% is the harness self-check (a non-100 WER on silence would mean broken normalization).
