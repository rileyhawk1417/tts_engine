# T003 — Round-trip WER eval harness

- **Status:** `not-started`
- **Milestone:** [M0](../index.md#milestone-m0--foundation)
- **Depends on:** [T002](T002-workspace-scaffold.md)

## Goal

The adoption gate: any lane's output is transcribed by `transcription_engine`, scored against source text, and reported alongside stage timings. No lane ships without a measured round-trip WER.

## Approach

- Corpus manifest `eval/corpus.toml`: internal bench sentences (short/medium/long + numbers/dates/addresses) + import step for `gradium/tts-hard-sentences-202610` (Hugging Face dataset, per-language split).
- Judge: shell out to the `te` binary, lane = nemo int8 default, parakeet q8_0 as tiebreak; record which judge lane + its version in every report row. WER convention = te repo's (`scripts/wer.py`) — same normalization, byte-identity checked.
- Determinism: synth under seed/temp=0 or lane-declared determinism; repeated runs must be byte-identical (gate) before WER is trusted — mirrors te load-parity discipline.
- Output: `eval/results/<date>-<lane>-<corpus>.json` { per-sentence WER, mean WER, TTFA mean/p95, RTF, peak RSS } + human-readable scoreboard.md table appended (design note only).
- Judge load parity: warm the te model once per eval batch; wall includes load separately logged.

## Done-conditions

- [ ] Running `eval/run.sh --lane <smoke> --corpus internal` produces a valid results JSON + WER even for silence (expected garbage WER — mechanics proven, not meaning)
- [ ] Byte-identity gate implemented and passing for 2 consecutive smoke runs
- [ ] Hard-sentences dataset downloaded once into `/teranux` cache and import step verified (counts match dataset card)
