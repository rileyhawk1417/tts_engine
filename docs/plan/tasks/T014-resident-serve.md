# T014 — `tts serve` (resident daemon at the seam)

- **Status:** `done (2026-11-06)`
- **Milestone:** [M5](../index.md#milestone-m5--resident-daemon)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md), [T013](T013-longtext-chunking-module.md)
- **Design north-star:** [notes/resident-service.md](../../notes/resident-service.md) (read it first — this file only pins scope + done-conditions)

## Goal

Make `tts_engine` a drop-in replacement for how bot-relay hosts engines
(`kokoroWorker` et al): one binary with two run modes — the existing `tts
synth` batch CLI (the ONLY eval/bench surface, unchanged laws) and `tts
serve`, a resident daemon that loads lanes once and streams the SAME
`SynthStream` chunk iterator to a socket.

**UNBLOCKED as of 2026-11-05**: piper phase A (rtf 0.075 wall/audio ≈
0.063 amy / 0.268 ryan, warm TTFA 204–663 ms med, seeded/byte-stable,
500/500 identity) removed the session-4-speed prerequisite — serve can ship
on kokoro + piper warm lanes. Session-4 speed (candle batching/gemm) is
throughput headroom, decided separately.

## Design constraints (all measured, in resident-service.md)

- **HTTP/1.1 over Unix domain socket** (TCP opt-in flag) — zero-dep clients
  (Node `fetch`, `curl`); a relay `TtsEngine` client stays ~1 file.
- **NDJSON framing**: `{type:"meta",…}` → `{type:"chunk",seq,pcm_b64}` × N →
  `{type:"done",metrics}`. Client owns ordering/backpressure.
- **Worker loop = 1 per lane** by default (relay's `KOKORO_POOL_SIZE=1`
  lesson: +1 worker = 4–9% throughput for ~300 MB; xn grain races shift
  chunk boundaries under load). Pool only lanes with a deterministic-under-
  load profile; `determinism_class` in meta tells the consumer.
- **The daemon admits what it is**: every response carries `mode`
  (cold|warm); fresh `load_ms` ONLY on a true cold (post-idle-unload) —
  never let a warm TTFA pass for a cold one.
- **Eval never runs through the daemon** (load parity, byte-identity gates,
  round-trip WER stay CLI-only).

## Scope

1. `Cmd::Serve` in `crates/tts` — flags: `--lane` (repeatable, boot-prefetch),
   `--socket PATH`, `--tcp host:port` opt-in, `--lazy`, `--idle-unload <s>`,
   `--threads`, `--nice <N>`; behind feature flag (feature-list-is-truth
   landmine — no mtime games).
2. Seam server loop: parse request → `SynthRequest` → lane `synthesize()` →
   stream chunks → `done` metrics (same RunRecorder rows, labeled `mode:warm`).
3. `SynthRequest.speed` + per-lane mapping (kokoro rate = 1/length_scale;
   piper length_scale passthrough; flow lane: verify rate control first).
4. `RunMeta` additions: `mode`, fresh-on-cold `load_ms`, `rss_after`,
   `resets` (idle session reset hook for ORT arena growth ~250→458 MB).
5. Resource hooks: `--idle-unload` + reset counter, `--nice`, thread dials
   targeting the CPU cap (relay container = 3.5 CPU / 2 GB), ≤ ~300 MB
   warm-resident per active lane target (kokoro int8 ~457 MB = pair with
   idle-unload or trim). **T010 ggml-ffi deviation: peak ≈ 2.4 GB (weights
   1.03 GB + KV + compute buffers) — the lane cannot live under this target;
   if served, it needs its own headroom rule or stays CLI-only. Second data
   point (T015): pocket-tts-en ≈ 666 MB peak (BF16→F32 materialization +
   KV voices) — 2× the ≤300 MB target, still servable with idle-unload.
   Kokoro 457 MB / flow-candle-phonon 219 MB stay in-budget.**
6. Relay-bench TTS phase (separate harness work): measure the DAEMON
   in-container, warm, worker-loop=1, vs `kokoroWorker` on the same lines.
   CLI cold numbers overstate the win by the load term — never quote them.

## Out of scope

- WebSocket (rejected: state + deps for zero benefit on single host).
- NAPI/in-process Node binding (couples release cycles; relay's own lesson).
- Voices/doctor endpoints (parked with the CLI subcommands, same vocab).
- Any daemon-only lane code path — forbidden by the seam.

## Done-conditions

- [x] `tts serve --socket …` green on kokoro + piper amy-rng (at minimum), warm
      requests stream NDJSON `meta→chunk×N→done` (33-chunk flow-candle stream
      verified too); curl over UDS **and** TCP works end-to-end.
- [x] Harmony gate: daemon replay of a CLI synth **byte-identical** — pocket
      (flow-candle) sha-256 match, piper amy-rng sha-256 match at the same
      `--threads` dial. ptts-class exposure: the daemon's worker-loop=1
      serializes decode so the T009 grain race stays CLI-batch-only; any
      flow-ptts eval rows keep the documented hard-500/1t fail budget
      (~1.4%) and never run through the daemon (eval stays CLI-only).
- [x] Honesty fields live: `mode` cold|warm on the wire, `load_ms` present
      ONLY on a true cold (post-lazy/post-idle-unload), absent on warm,
      `determinism_class`, `rss_after_kb`/`resets` — curl transcripts in
      notes/resident-service.md.
- [x] `--idle-unload` reset measured: lazy boot 15 MB RSS → cold 508 ms →
      warm (no load_ms) → idle > 4 s `loaded:false` `resets:1`. E2E bug
      found+fixed: `last_used` stamped post-synth (lazy iterator) — a 10 s
      synth outran a 4 s window and the next request paid a gratuitous cold.
- [x] `speed` dial wired on piper (`length_scale = 1/speed`, clamp
      [0.25, 4.0]) + kokoro (direct rate, clamp [0.5, 2.0]); documented in
      notes/piper.md contract block + notes/resident-service.md.
- [ ] Relay-bench row measured in-container (3.5 CPU / 2 GB) vs kokoroWorker
      pipeline; numbers landed in a scoreboard note before any relay swap.
      **OPEN by design — this is the relay-side half of the swap. This task
      ships the engine half. Hand-off status in notes/resident-service.md.**

## Result (2026-11-06)

**VERDICT: ADOPT.** `tts serve` ships as the resident engine host. M5 code
done; the one open done-condition (relay-bench in-container row) is the
relay-side measurement that consumes this daemon, tracked in
notes/resident-service.md → relay swap order.

- **Wire**: `POST /synth` → 200 `Content-Type: application/x-ndjson`,
  `Transfer-Encoding: chunked` — each event INSIDE its HTTP chunk carries its
  own trailing `\n` (E2E-proven: a bare `\r\n` is transport framing that
  curl/python strip during decode, and naive framing merged every event onto
  one line). `GET /health` = single JSON: pid, idle_unload_s, threads_dial,
  rss_after_kb/vm_peak_kb, per-lane `{lane, model, loaded, resets,
  idle_ms_ago}`.
- **Honesty law (on the wire)**: `mode` cold|warm; `load_ms` present ONLY on
  a true cold (lazy first-use OR post-idle-unload reload); warm requests
  carry NO `load_ms` key (absent, never null). `resets` = cumulative
  unload-and-reload count.
- **Measured on the 4800H, release, seed 42**:
  - **kokoro** warm: 3.525 s audio, ttfa 4,641 ms, rtf 1.32 —
    **daemon == CLI byte-identical (sha256 `6bea4778…`)**, even ACROSS intra
    dials (4 vs 16) — for kokoro, intra is a perf dial, not a determinism dial.
  - **piper amy-rng** warm: ttfa **197–278 ms** (vs 204 ms CLI cold load),
    rtf 0.062–0.087 — **daemon == CLI byte-identical at the same dial**;
    across dials (4 vs 16) bytes differ (VITS matmuls are intra-sensitive).
    Speed 1.25: 3.192 s → 2.774 s audio.
  - **flow-candle(pocket)** warm: 33 chunks, ttfa 512 ms, wall 12.2 s,
    RSS ~600–786 MB — **daemon == CLI byte-identical**.
  - **lazy+idle-unload**: boot ~15 MB RSS, `loaded:false`; first synth cold
    `load_ms=508`; idle > 4 s → `loaded:false`, `resets:1` (arena freed).
- **Boot gates**: `--threads` sets `TTS_ORT_INTRA` (fail-fast if already
  set), `--nice` → `setpriority(PRIO_PROCESS, 0, N)`; zero-lane boot exits 2;
  both/neither listener exits 5 with usage text.
- **Tests**: 12 unit tests in serve.rs (parse_request incl. oversize-body
  413, base64 RFC-4648 vectors + std roundtrip, speed mapping ×3 incl.
  refusal on flow lanes, ndjson chunked framing, lane_slot selection,
  next_unload_wait); clippy `-D warnings` clean (all-targets);
  `scripts/build.sh` OK (all five lanes).
- **libc**: deliberate — `setpriority`/`poll` live here (`crates/tts`
  Cargo dep, NOT tts-engine).
