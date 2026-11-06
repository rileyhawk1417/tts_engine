# Resident-service seam — daemon & CLI harmony

> Design north-star for making `tts_engine` a drop-in replacement for how bot-relay
> hosts engines (whisper pool, `kokoroWorker`, `parakeetWorker` — ThyroxDevelopment
> upstream `33f1fff`). Notes carry design only; status lives in the plan index.

## Why this note exists

Relay's prod law is the same one `te` proved for interactive STT: **warm residency
beats per-utterance batch** for interactive latency — a batch-file engine loses to a
warm resident pool on 1–8 s utterances regardless of RTF. Relay hosts *every* heavy
engine resident, tuned by measurement:

| relay component | shape | measured lesson |
|---|---|---|
| whisper pool | 2 resident whisper.cpp-server processes, POST `/inference`, nice −15, threads=2 | pool wins on 1–8 s inputs regardless of RTF |
| `kokoroWorker` | worker_threads, model loads once, `KOKORO_POOL_SIZE=1` | +1 worker = only 4–9% throughput, ~300 MB each, **slower first audio** — pool stays tiny |
| `parakeetWorker` | separate process | sync ~7 s native load would freeze the bot; native deps (own onnxruntime) get process isolation |
| `Speaker` | per-guild FIFO (drop oldest >30), chunk-streamed playback, historically globally serialized synth | client owns ordering/backpressure — keep the server dumb |
| budget | container cap **3.5 CPUs / 2048 MB**, nice discipline (`piper -n 10`) | engine must share the box with whisper |

Our seam today is one-shot `tts synth` — the right eval/scripting surface, useless to
relay. If a relay lane swap is the goal, **one binary must serve both consumers without
compromising the eval laws.**

## Design: one binary, two run modes

| mode | consumer | load | semantics |
|---|---|---|---|
| `tts synth` (batch CLI) | eval, bench, scripting | process-local, mmap loads per run | laws unchanged: load parity, `load_ms` measured separately (never baked into TTFA/RTF), byte-identity gates, seeded reproducibility |
| `tts serve` (resident daemon) | relay | lane(s) loaded once at boot or lazily | per-request synth; chunk-streamed responses; per-request telemetry with honest mode labels |

Non-negotiables:

1. **Same lane core.** `Lane::synthesize` already returns a chunk iterator
   (`SynthStream`). CLI drains it to WAV; `serve` pushes it to a socket. No
   daemon-only code paths inside lanes — ever.
2. **Eval never runs through the daemon.** Bench/eval discipline (load parity,
   byte-identity gates, round-trip WER) stays CLI-only. The daemon is a consumer,
   not a measurement surface.
3. **The daemon admits what it is.** Every response carries `mode` (cold|warm),
   `determinism_class`, and fresh `load_ms` *only on a true cold start* (e.g. after
   idle-unload). A consumer must never be able to conflate a warm TTFA with a cold one.

## IPC shape (resolves the plan's open question)

Decided direction:

- **Transport: HTTP/1.1 over a Unix domain socket** (TCP opt-in via flag). Zero-dependency
  clients (Node `fetch`/undici, `curl` for probes), fits relay ops, and a relay
  `TtsEngine` impl in `tts.ts` stays ~1 file — same shape as the sherpa engine swap
  (PR #87 pattern: opt-in engine behind the existing interface, fallbacks intact).
- **Framing: NDJSON events** — `{type:"meta",...}` → `{type:"chunk", seq, pcm_b64}` × N →
  `{type:"done", metrics}`. Base64 costs 33% on a local socket — irrelevant. Keep the
  protocol dumb; the client owns sequencing and backpressure (relay's Speaker already
  serializes and drops oldest >30).
- **Concurrency: one worker loop per lane by default** — relay measured that kokoro
  workers beyond `KOKORO_POOL_SIZE=1` buy 4–9% throughput for ~300 MB each and slower
  first audio, and our `xn` grain race shifts chunk boundaries *under load*. Pool only
  lanes with a deterministic-at-threads>1 profile; serialize the rest;
  `determinism_class` in meta tells consumers which they got.
- **Rejected alternatives**, recorded so we don't relitigate:
  - *NAPI / in-process in Node* — couples Rust release cycles to JS; relay's own lesson
    is that multi-second CPU synth must not run where playback runs (`kokoroWorker.ts`
    exists for exactly this reason).
  - *Per-utterance `tts synth` spawn* — cold load each time; the batch loss relay proved
    with whisper. Residency or nothing.
  - *WebSocket* — state and deps for zero benefit on a single host; HTTP verbs suffice
    (GET `/voices`, POST `/synth`).

## Resource discipline (hooks the daemon requires, why measured)

| # | requirement | grounded in |
|---|---|---|
| 1 | **RSS bounding**: optional idle session reset / arena cap per lane; telemetry reports reset count | kokoro ORT arena grows with chunk count (~250 → ~458 MB in one paragraph run); ORT init is cheap (~0.74 s measured) vs growth |
| 2 | **Thread dials target the CPU cap, not the host** (`--threads`) | relay caps at 3.5 CPUs; our numbers are on an uncapped 4800H 8C/16T; te law: ORT thread optimum is sharp — re-sweep under the production cap before quoting any relay number |
| 3 | **ptts-class determinism default = threads 1** in the daemon unless the consumer opts out | xn byte-identity only trusted at 1t; grain race is load-timing dependent |
| 4 | **`--lazy` per lane, `--idle-unload <s>`, prefetch at boot** | parakeet-style lazy process is relay's pattern for heavy/native lanes; `prefetch-kokoro.mjs` bakes models into the image; our mmap-first loading keeps resident RSS near weights (page-cache shared) |
| 5 | **`--nice <N>`** | piper runs `nice -n 10` today; whisper pool `nice -15`; the daemon must be the same kind of citizen |
| 6 | **Memory budget: ≤ ~300 MB warm-resident target per active lane** | flow lanes 182–219 MB fit comfortably; kokoro-ort ~457 MB is the outlier (trim or pair with idle-unload); neutts-air 1.07 GB disqualified, nano 744 MB borderline-out |

## Seam deltas this implies (keep-in-mind checklist)

- [x] `speed` field in `SynthRequest` + a per-lane mapping table — **T014
      landed**: kokoro direct rate (clamp 0.5–2.0), piper `1/s` → length_scale
      (clamp 0.25–4.0); flow lanes hard-error (no rate control verified).
- [x] RunMeta additions + wire honesty: `mode` (cold|warm), `load_ms` (fresh
      on cold ONLY, absent key on warm), `determinism_class`, `rss_after_kb`,
      `resets` — landed in serve.rs (types.rs untouched; serve-side meta).
- [x] Lane capability surface for the swap decision — **scoreboard.md §T014
      table** (harmony, ttfa, rtf, RSS per lane; determinism class from
      meta). Per-lane eval capability rows stay on the existing scoreboard.
- [x] `serve` builds WITHOUT a dedicated feature flag — lanes gate themselves
      (feature-list-is-truth survives: a lane absent from the build errors
      "not built into this binary" at request time; no mtime games needed).
- [x] **Harmony gate**: daemon replay byte-identical for kokoro, piper
      (same dial), flow-candle/pocket; ptts-class stays CLI-only (grain-race
      budget documented, 1.4–2.2% under load) — see scoreboard §T014.
- [ ] `voices`/`doctor` subcommands already parked for "later" in seam.md — same vocab
      becomes GET endpoints on the daemon; no separate surface.

## Plan impact

- **T011 (piper streaming lane)** — done phase A 2026-11-05. It landed the
  chunk-streaming seam shape (T013 chunks + `SynthStream`, seeded noise,
  byte-stable under load) and — with amy's ≈ 0.063 wall/audio + 500/500
  identity — made the daemon viable with no decode-speed work. The IPC
  protocol itself is T014's, not T011's (an early draft had T011 carrying
  the protocol; that split off when the lane landed without it).
- **T014 (`tts serve`) — now a board task (M5)**: task file
  [plan/tasks/T014-resident-serve.md](../plan/tasks/T014-resident-serve.md);
  deps T002/T003/T013; done-conditions: harmony gate green, resource table
  hooks landed, relay-bench rows measured. ~~Sequence after session-4
  speed~~ **Amended 2026-11-05**: piper phase A removes the dependency —
  serve ships on kokoro + piper warm lanes; session-4 speed (candle
  batching/gemm) is throughput headroom, not a prerequisite. The original
  concern (flow-candle 0.334 backing up a FIFO queue) applies only if piper
  falls through.
- **relay-bench TTS phase (when built) must measure the daemon**, not the CLI: in-container
  (3.5 CPU / 2 GB cap), warm, worker-loop=1, versus the `kokoroWorker` pipeline on the
  same lines (chat text, links/emoji paths, speaker prefix). CLI cold numbers would
  overstate our win by the load term.

## Shipped protocol (T014, 2026-11-06) — the wire, as landed

- `target/release/tts serve --lane NAME=MODEL … --socket path \| --tcp host:port
  [--lazy] [--idle-unload s] [--threads n] [--nice n]`. Boot log prints one
  `loaded in <ms>` line per pre-warmed lane, then `listening` with
  `idle-unload <s>, lazy <bool>`.
- `POST /synth` body: `{lane?, text, voice?, lang?, seed?, speed?,
  chunk_policy?}` — `lane` absent = first configured lane (single-lane
  daemons can omit it).
- Response 200 with NDJSON (chunked): meta → `chunk,seq,pcm_b64` × N → done.
  **The `\n` is INSIDE each HTTP chunk payload** — bare `\r\n` is transport
  framing that curl/python strip during decode (E2E lesson: naive framing
  merged every event onto one line). Clients should decode byte-wise, not
  line-wise.
- Keys: meta = `type,lane,model,model_sha,voice,lang,seed,sample_rate,
  runtime,chunk_policy,determinism_class,nondet_reason,restart_voice,mode,
  load_ms? (only on cold),resets,rss_after_kb,vm_peak_kb,protocol:"ndjson-chunks/1"`;
  done = `type,chunks,audio_s,wall_ms,ttfa_ms,rtf,mode,resets,sample_rate,
  rss_after_kb,vm_peak_kb`.
- `GET /health`: `{type:"health",pid,idle_unload_s,threads_dial,
  rss_after_kb,vm_peak_kb,lang_default:"en",lanes:[{lane,model,loaded,
  resets,idle_ms_ago}]}`.
- Errors: plain HTTP before streaming (404 unknown path, 405 bad method,
  400 parse, 413 oversize, 409 unknown lane/not loaded, 500/422 lane errors),
  NDJSON `error` events after meta.
- **speed**: piper `length_scale = 1/speed` (clamp 0.25–4.0); kokoro direct
  rate (clamp 0.5–2.0); flow lanes refuse (hard error) — T014 measured only
  piper (3.192 → 2.774 s at speed 1.25).
- **Determinism**: daemon == CLI byte-identical for kokoro, piper (same
  dial), flow-candle/pocket — harmony gate green. Cross-dial: kokoro
  identical even across intra 4 vs 16; piper differs (intra-sensitive
  matmuls). Relay clients that care pin the daemon's `--threads` and reuse
  one daemon for a synth profile.
- **Lifecycle gates measured**: lazy boot 15 MB RSS; cold 508 ms honest;
  warm without `load_ms`; idle > 4 s unload, `resets:1`, arena freed.
- **Hand-off to the relay swap (the ONE open T014 done-condition)**:
  measure this daemon in-container (3.5 CPU / 2 GB cap, warm, worker-loop=1)
  vs the `kokoroWorker` pipeline on the same lines and land the scoreboard
  row BEFORE any lane swap. Zero-dep Node client = `http.request` with
  `socketPath` (Node's builtin http supports UDS), byte-wise NDJSON decode,
  ~1 file in `bot-relay/src/tts.ts` behind the existing `TtsEngine` seam.
  **Status 2026-11-06 (example client): the client now EXISTS and is
  tested — `examples/node-client/tts-client.js` (+ 14-check harness
  `examples/node-client/tts-client.test.mjs`, 14/14 green incl.
  client-assembled-WAV == CLI byte-identity `80241f1c…`; embedding notes in
  `examples/node-client/README.md`). Chat-replay harness `eval/relay/bench.mjs`; first
  local-approximation runs in `eval/relay/RESULTS.md` — NON-ADMISSIBLE
  (taskset 4-core), in-container row still owed.**

## Cross-refs

- `docs/seam.md` — one-shot seam + chunk protocol (SynthStream, exit codes).
- `docs/notes/scoreboard.md` — RSS/TTFA/RTF facts quoted above; grain-race identity budgets.
- `docs/plan/index.md` — board; M5/T014 row; resolved IPC open question links here.
- Relay reality: `software/ThyroxDevelopment/bot-relay/src/{kokoroWorker,parakeetWorker,speaker,tts}.ts`,
  `docs/RELAY.md` "Read-aloud pipeline" §; te case study: transcription_engine `refs/relay/BENCH_NOTES.md`.
