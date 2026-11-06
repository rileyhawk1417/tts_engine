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

## 2026-11-06 addendum — Dockerized daemon row (cap REAL, still not the final admissible row)

Daemon ran in Docker (archlinux base, espeak-ng via pacman, ORT + models +
binary bind-mounted) with **`--cpus 3.5 --memory 2g --memory-swap 2g`**
(cgroup-verified: cpu.max=350000/100000, memory.max=2GiB). Host client via
bind-mounted UDS. Both lanes, workers 1 & 2 → `results/20261106-0725-incontainer-{w1,w2}/`:

| Run | Lane | TTFA med | TTFA p95 | rtf_mean | rss max |
|---|---|---|---|---|---|
| docker w1 | kokoro int8 | 6785 ms | 9750 ms | 0.516 | 424 MB |
| docker w1 | piper amy | **404 ms** | 538 ms | 8.5 | 531 MB |
| docker w2 | kokoro int8 | 6832 ms | 9691 ms | 0.520 | 588 MB |
| docker w2 | piper amy | 400 ms | 534 ms | 8.6 | 588 MB |

Matches the 4-core approximation within noise (kokoro 6.6→6.8 s, piper 372→404 ms).
**Functional proof: the 14-check client harness passed 14/14 from a network-less
`node:26-slim` container against the capped daemon** (socket perms needed
`chmod 666` — in-container daemon is root; production must chmod or run as the
bot user). Remaining gap vs the admissible row: the `kokoroWorker` comparison
arm on the same lines + the real relay box hardware.

## Next runs (the admissible ones)

1. In-container daemon: `bench.mjs --sock /run/tts/tts.sock --lane kokoro --lane piper --workers 1 --workers variant` …
2. Same lines through `kokoroWorker` (`KOKORO_POOL_SIZE=1`) — replay from
   `relay-bench`'s existing harness/lines so both sides share the corpus.
3. Land both on the scoreboard as the T014 close-out row (before any swap).

## 2026-11-06 addendum 2 — flow-candle lane, in-container (first CPU-device row)

Same docker topology as the 0725 row (archlinux base, host client via
bind-mounted UDS, `--cpus 3.5 --memory 2g --memory-swap 2g` cgroup-verified:
cpu.max=350000/100000, memory.max=2GiB), new lane. Binary is the **cuda-featured
release build** (the GPU-sweep binary, Nov 6 10:29) booted with
`--lane flow-candle=/srv/models/phonon-31m-en --device cpu` (cpu is the
flow-candle default and never initializes CUDA; the DT_NEEDED GPU libs
libcuda/curand/cublas/cudart are host bind-mounts so the loader is satisfied).
The bare `archlinux` base lacks the espeak-ng runtime deps the compiled-in
piper/espeak feature links (libpcaudio, libsonic, audio stack transitives) —
bound from the host by exact `ldd` walk, all `ro`; ORT lib bound too but never
dlopened (a flow-candle-only serve creates no ORT sessions).

Runs → `results/20261106-{1112-…-w1, 1115-…-w2, 1119-…-rerun}/`:

| Run | TTFA med | TTFA p95 | wall med | rtf_mean | rss max | Notes |
|---|---|---|---|---|---|---|
| w1 (serial) | 393 ms | 535 ms | 11.8 s | 0.284 | 421 MB | ran while the GPU hard-sweep judge tail was still busy |
| w2 (2 in-flight) | 414 ms | 508 ms | 12.2 s | 0.287 | 423 MB | queueing barely moves TTFA |
| rerun, box idle | **384 ms** | **479 ms** | 11.0 s | 0.299 | 403 MB | quoted row; dev-board rtf 0.334 reproduced |

Differences vs the kokoro/piper rows: flow-candle streams **many chunks per
line** (22–61, vs chunks=1 there) — NDJSON streaming buys real early audio
here: TTFA ~0.39 s against ~11 s total synthesis. Sustained throughput stays
slower than realtime (rtf 0.30 audio/wall → ~3.3× slower at 3.5 CPUs; same
class as kokoro's 0.52, where piper's 8.5 is the only lane that keeps up).
For the relay feel this is a TTFA-first lane: first words land 17× under the
2 s bar with kokoro-class sustained backlog on long read-aloud queues.

Same remaining gap as above: `kokoroWorker` comparison arm + the real relay
box hardware — those rows move the claim from engine-side to admissible.

### Addendum 3 — flow-ptts in-container clears the relay bar (2026-11-06)

Same topology as addendum 2 (archlinux base, `--cpus 3.5 --memory 2g
--memory-swap 2g --network none`, host-lib `:ro` mounts, `--device cpu`), but
with the **flow-ptts** lane (llama.cpp backend, resolves `model.q8.gguf` Q80,
loads in 117 ms, boot RSS 211 MB). Same 14-chat-line corpus, seed 42, warm:

| row | TTFA med | TTFA p95 | wall med | rtf | RSS max | notes |
|---|---|---|---|---|---|---|
| w1, n=14 | 295 ms | 385 ms | 1.16 s | 3.117 | 354 MB | streaming, 22+ chunks/line |
| w2, n=14 | 284 ms | 367 ms | 1.14 s | 3.134 | 354 MB | 2 in-flight ≈ w1 |
| w1 reps=3, n=42 | **302 ms** | **438 ms** | ~1.17 s | 3.080 | 354 MB | firmer p95 |

Cold first-request after boot (`warmup` row): TTFA 2069 ms, rtf 1.41 — one hit
per daemon boot, right at the 2 s bar; steady state is far under.

**This clears the reviewer-stated relay bar on the same cap**: first audio
< 2 s AND rtf ≥ 1 at 3.5 CPUs — rtf 3.08 means synthesis runs ~3× faster than
playback, so an assemble-then-play client still delivers a full line's WAV in
~1.2 s end-to-end; a chunk-streaming client reaches the 0.3 s daemon-side
TTFA. Contrast with the candle port of the same family (addendum 2): the
llama.cpp q8 path is ~10× the sustained CPU throughput (3.08 vs 0.30) and
~half the memory (354 vs ~420 MB). flow-candle remains the GPU-side port;
flow-ptts is the CPU lane.

Remaining for admissibility: relay client must stream chunks to actually ship
0.3 s first audio (assemble-then-play gives ~1.2 s — still under the bar), and
the real relay box A/B is the reviewer's step by their own framing.
