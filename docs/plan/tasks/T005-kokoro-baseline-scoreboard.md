# T005 — Kokoro baseline scoreboard on Zen 2

- **Status:** `not-started`
- **Milestone:** [M1](../index.md#milestone-m1--default-lane-graph-ort--kokoro)
- **Depends on:** [T004](T004-kokoro-ort-lane.md)

## Goal

The bar every later lane must beat or justify against: measured RTF, TTFA, peak RSS, round-trip WER for our own Kokoro lane on the reference box (Zen 2 4800H 8C/16T). Also decides which `te` judge lane becomes the default.

## Approach

- Full eval matrix: internal corpus + hard-sentences, cold + warm, 1-thread vs 8-core (te dials: threads = physical cores, ROW_SPLIT/TE_DNNL analogs only where ORT exposes them), judge nemo int8 vs parakeet q8_0 head-to-head.
- Publish into `docs/notes/scoreboard.md` (design numbers, dated) + full JSON artifacts in eval/results; mirror the headline row into a memory checkpoint (numbers live in memory as tracking, docs keep the dated note — per repo-docs convention).
- Derived targets for M2 (Phonon): expected TTFA ≤ Kokoro's (streaming should crush it), RTF target, WER within ±0.5pt of Kokoro's round-trip on internal corpus.
- CPU-time tiebreak rule carried from te if RTFs are within noise.

## Done-conditions

- [ ] Scoreboard row complete: RTF / TTFA mean+p95 / peak RSS / round-trip WER (both corpora) / judge identity — each with run date + determinism gate passed
- [ ] Default judge lane chosen with stated reason
- [ ] M2 targets written down (memory checkpoint, not just chat)
