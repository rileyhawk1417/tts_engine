# T011 — Piper streaming lane (streaming-graph)

- **Status:** `not-started`
- **Milestone:** [M4](../index.md#milestone-m4--streaming--measurement-lanes)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md), [T013](T013-longtext-chunking-module.md)

## Goal

The latency competition for the flow-latent lane: streaming VITS via Piper/ONNX (split encoder/decoder, mel chunks + overlap-add), as an en_US graph-based streaming alternative.

## Approach

- Study the streaming approach (piper streaming PR #255 lineage): mel chunk sizes (~45 frames + 10 padding) and overlap-add windowing ported as documented constants, not vibes.
- Lane layout mirrors graph-ort (ort sessions, session-per-thread, int8-first); decoder-dominance expectation from prior art: phoneme→mel fast, decode steps ~32 ms each — measure ours.
- TTFA headline metric (streaming should beat non-stream Kokoro decisively); compare head-to-head vs flow-ptts/flow-candle on identical corpus.
- en_US voice set minimal (2 voices); long-text runs through T013 chunking with overlap-add only at decoder level.

## Done-conditions

- [ ] Streaming output verified continuous (no seams audible on a 3-sentence paragraph — spot-listen + round-trip WER no worse than non-stream)
- [ ] TTFA + RTF row on the scoreboard vs flow-latent and Kokoro rows
- [ ] Adopt/shelve verdict: keep as lane or shelve with reason (if flow-latent dominates on both axes)
