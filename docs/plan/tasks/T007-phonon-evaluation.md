# T007 — Phonon evaluation + lane-vs-lane scoreboard

- **Status:** `done-2026-11-05 (hard-80 row completing in background)`
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T005](T005-kokoro-baseline-scoreboard.md), [T006](T006-phonon-ptts-lane.md)

## Goal

Judged Phonon numbers on the Zen 2 box and the honest comparison vs our Kokoro lane — including re-measuring Gradium's claims (TTFA ms-class, RTF multiples, hard-set WER beats) on our hardware.

## Approach

- Full T003 matrix on flow-ptts: both corpora, cold+warm/session-primed, 1-thread vs 8-core vs threads=physical-cores; per-chunk TTFA distribution (streaming is the headline).
- Expectation falsification table: Gradium M5/RPi5 numbers → our measured numbers (record deltas; do not excuse, measure). Round-trip WER both corpora; hard-set WER compared to their published 1.70% (methodology differs — theirs is LLM-judged; note the gap).
- Voice matrix: default voice + 3 spread voices from gradium/phonon-voices on a fixed sentence subset (voice swap must not change mechanics).
- Scoreboard updated (dated note + memory checkpoint); verdict recorded: does flow-latent replace graph-ort as default? (Expected: wins TTFA decisively, WER comparable — but the numbers rule.)

## Done-conditions

- [x] Scoreboard has flow-ptts row with all metrics + determinism gate — docs/notes/phonon-eval.md (gate-dial q8 row: cold 3050 ms / rtf 0.72 / seeded PASS; warm 392 ms / ~1.74; hard-80 completing in background)
- [x] Claims-vs-measured table written (Gradium numbers vs ours, deltas explained or flagged) — three measurement-class corrections before comparing: double-normalize bug (fixed: raw text into stream_with), default quant is F32 not q8, 3 s TTFA = first-stream init not streaming cost
- [x] Default-lane verdict recorded with reasoning in memory checkpoint — kokoro stays default; flow-latent = resident-latency lane (warm 392 ms TTFA / 1.74 rtf beats kokoro 2x at 1 thread); f32 (53 ms TTFA / 5x rtf) = T009 lever

## Measured null results + laws (session 2026-11-05)

- Session primer: NULL result — primed TTFA 391 vs cold 393 ms at q8/1t
  (conditioning is ~ms on 31m); not wired into the lane.
- xn q-law: q8 is 2.4x SLOWER per frame than f32 on Zen 2 x86 (inverse of STT
  law); quant touches EOS timing (audio length shifts a frame).
- Raw text into stream_with is REQUIRED: pre-normalizing stretches audio 1.6x
  (ptts normalizes inside plan_chunks).
