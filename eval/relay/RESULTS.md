# relay-bench TTS phase — local-approximation runs (2026-11-06)

> **Label: NON-ADMISSIBLE.** The T014 done-condition 6 row must be measured
> **in-container** (3.5 CPU / 2 GB), daemon AND `kokoroWorker` on the same
> lines. These engine-side numbers approximate the cap with `taskset -c 6-9`
> (4 cores of a 4800H) and are first-cut context for *what to expect*, nothing
> more. Harness: `eval/relay/bench.mjs`; daemon = `tts serve` (kokoro int8 +
> piper amy-rng), 14 chat-shaped lines, seed 42, `--workers 1|2 --reps 1`.

## Results (Node v22.22.0 client → UDS → daemon warm)

| Run | Lane | TTFA med | TTFA p95 | wall med | rtf_mean (audio/wall) | rss max | Notes |
|---|---|---|---|---|---|---|---|
| w1 (serial) | kokoro int8 | 6572 ms | 10 278 ms | 6573 ms | 0.506 | 424 MB | 4-core pin: ~40% slower than the unpinned 4.6 s hard-80 TTFA; per-lane loop=1, one chunk per line → TTFA≈wall |
| w1 (serial) | piper amy | **372 ms** | 504 ms | 373 ms | **9.15** | 534 MB | seeded, deterministic repeat verified separately |
| w2 (2 in-flight) | kokoro int8 | 7025 ms | 9912 ms | 7026 ms | 0.515 | 594 MB | workers share the 4-core pin plus daemon loop=1 serialization → med TTFA ~unchanged, only queueing |
| w2 (2 in-flight) | piper amy | 374 ms | 503 ms | 374 ms | 9.12 | 594 MB | queue wait invisible at this wall scale |

Cold-lane warmup rows (`warming the lane`, 18 chars): kokoro 2588 ms, piper
155 ms — NOT comparable to the main medians (different line lengths; mode was
already warm). Full per-line rows: `results/20261106-0713-local-{w1,w2}/`
(`bench-lanes.tsv`, `summary.json`).

## What this suggests for the in-container row

- **Piper looks overwhelming for chat latency** at a 3.5-CPU cap (TTFA med
  372 ms, ~9× audio/wall, RSS ~0.5 GB). If the row holds in-container, the
  "daemon lane" decision is piper-first for read-aloud, kokoro-on-request.
- **Kokoro TTFA med ≈ 6.6 s under the cap** vs today's `kokoroWorker` in the
  same budget — expect the row to decide **daemon vs pipeline kokoro**, not
  kokoro vs piper (the pipeline pays the same ONNX graph cost; the comparison
  target is residency overhead + serialization, not the model).
- RSS peak ~0.6 GB with both lanes loaded leaves headroom under the 2 GB cap
  minus the whisper pool (unknown here — the container row measures it).

## Next runs (the admissible ones)

1. In-container daemon: `bench.mjs --sock /run/tts/tts.sock --lane kokoro --lane piper --workers 1 --workers variant` …
2. Same lines through `kokoroWorker` (`KOKORO_POOL_SIZE=1`) — replay from
   `relay-bench`'s existing harness/lines so both sides share the corpus.
3. Land both on the scoreboard as the T014 close-out row (before any swap).
