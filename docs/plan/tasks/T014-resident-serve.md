# T014 — `tts serve` (resident daemon at the seam)

- **Status:** `not-started`
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
   idle-unload or trim).
6. Relay-bench TTS phase (separate harness work): measure the DAEMON
   in-container, warm, worker-loop=1, vs `kokoroWorker` on the same lines.
   CLI cold numbers overstate the win by the load term — never quote them.

## Out of scope

- WebSocket (rejected: state + deps for zero benefit on single host).
- NAPI/in-process Node binding (couples release cycles; relay's own lesson).
- Voices/doctor endpoints (parked with the CLI subcommands, same vocab).
- Any daemon-only lane code path — forbidden by the seam.

## Done-conditions

- [ ] `tts serve --socket …` green on kokoro + piper amy (at minimum), warm
      requests stream NDJSON `meta→chunk×N→done`, relay-style client probe
      (curl or tiny .mjs) works end-to-end on the socket.
- [ ] Harmony gate: daemon replay of a CLI-recorded corpus byte-identical
      for deterministic profiles (piper seeded; candle), with the documented
      under-load fail-budget stated for any ptts-class exposure.
- [ ] Honesty fields live: `mode` cold|warm, `load_ms` fresh only on cold,
      `determinism_class`, `rss_after`/`resets`.
- [ ] `--idle-unload` reset measured (arena stops growing across requests).
- [ ] `speed` dial wired end-to-end on at least one lane with its mapping
      documented in the lane's contract block.
- [ ] Relay-bench row measured in-container (3.5 CPU / 2 GB) vs kokoroWorker
      pipeline; numbers landed in a scoreboard note before any relay swap.
