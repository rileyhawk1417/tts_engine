# T007 — Phonon evaluation + lane-vs-lane scoreboard

- **Status:** `not-started`
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

- [ ] Scoreboard has flow-ptts row with all metrics + determinism gate
- [ ] Claims-vs-measured table written (Gradium numbers vs ours, deltas explained or flagged)
- [ ] Default-lane verdict recorded with reasoning in memory checkpoint
