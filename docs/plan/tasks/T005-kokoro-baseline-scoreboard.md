# T005 — Kokoro baseline scoreboard on Zen 2

- **Status:** `completed` (2026-11-05)
- **Milestone:** [M1](../index.md#milestone-m1--default-lane-graph-ort--kokoro)
- **Depends on:** [T004](T004-kokoro-ort-lane.md)

## Goal

The bar every later lane must beat or justify against: measured RTF, TTFA, peak
RSS, round-trip WER for our Kokoro lane on the reference box (Zen 2 4800H
8C/16T), judged through `te`. Decides the default judge lane.

## What ran (see [docs/notes/scoreboard.md](../../notes/scoreboard.md) for the dated note)

- Full matrix: internal corpus (8 sentences) + hard-80 (stride-80 subset of the
  500-sentence gradium hard corpus), judge nemo int8 AND parakeet q8_0
  head-to-head, byte-identity gate `pass` everywhere.
- Headline: internal WER ≈ 6-7% / RTF ≈ 0.75-0.86 / TTFA ≈ 3-7 s / RSS ≈ 420-460 MB.
- Judge decision: **nemo int8 default** (tie on internal, clearly better on the
  symbol-heavy hard set: 42.03 vs 49.62).
- Key measured fact: TTFA is COMPUTE-bound (repeat experiment: 3 s TTFA across
  in-process reps with a single 741 ms load) — warm pooling does not help
  kokoro TTFA; streaming lanes win by generating progressively.
- Harness fixes landed: `--repeat N` CLI flag (rep-indexed outputs, rep 0 cold),
  aggregate-step hardening (missing cells → None), clean corpus dir stamps,
  parakeet judge precheck now reflects te's runtime gate (rebuild with
  TE_FEATURES=nemo,parakeet).

## Done-conditions

- [x] Scoreboard row complete (RTF/TTFA mean+p95/peak RSS/WER both corpora/judge
      identity) with run dates + determinism gate pass — all four runs `trusted`
- [x] Default judge lane chosen with stated reason (nemo int8)
- [x] M2 targets written down (memory checkpoint, not just chat)
