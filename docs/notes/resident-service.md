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

- [ ] `speed` field in `SynthRequest` + a per-lane mapping table (kokoro: rate =
      1/length_scale; piper: length_scale passthrough; flow lane: verify phonon rate
      control exists before promising it).
- [ ] `RunMeta`/chunks.jsonl additions: `mode` (cold|warm), `load_ms` (fresh on cold),
      `determinism_class`, `rss_after`, `resets`.
- [ ] Lane capability matrix in `targets.md`: streaming? speed? determinism class? warm
      RSS? warm TTFA? — one row per lane, the swap-decision surface.
- [ ] `serve` behind a build feature flag — remember the feature-strip landmine: mtime
      lies via hardlink relink; *the feature list is the only truth*.
- [ ] **Harmony gate**: daemon replay of a CLI-recorded corpus must produce byte-identical
      PCM for deterministic profiles (candle 80/80; ptts@1t idle), with a documented
      under-load fail-budget for ptts-class lanes (known 1.4–2.2%).
- [ ] `voices`/`doctor` subcommands already parked for "later" in seam.md — same vocab
      becomes GET endpoints on the daemon; no separate surface.

## Plan impact

- **T011 (piper streaming lane)** carries the streaming-IPC seam design — land the
  protocol there first; it is the smallest honest vehicle for it.
- Candidate new task (M4): **T014 resident-service lane (`tts serve`)** — deps T002/T013;
  done-conditions: harmony gate green, resource table hooks landed, relay-bench rows
  measured. **Sequence after session-4 speed**: flow-candle wall (rtf 0.334 vs kokoro
  0.80) would back up relay's FIFO queue today; throughput first, residency second.
- **relay-bench TTS phase (when built) must measure the daemon**, not the CLI: in-container
  (3.5 CPU / 2 GB cap), warm, worker-loop=1, versus the `kokoroWorker` pipeline on the
  same lines (chat text, links/emoji paths, speaker prefix). CLI cold numbers would
  overstate our win by the load term.

## Cross-refs

- `docs/seam.md` — one-shot seam + chunk protocol (SynthStream, exit codes).
- `docs/notes/scoreboard.md` — RSS/TTFA/RTF facts quoted above; grain-race identity budgets.
- `docs/plan/index.md` — open question this note resolves; candidate T014.
- Relay reality: `software/ThyroxDevelopment/bot-relay/src/{kokoroWorker,parakeetWorker,speaker,tts}.ts`,
  `docs/RELAY.md` "Read-aloud pipeline" §; te case study: transcription_engine `refs/relay/BENCH_NOTES.md`.
