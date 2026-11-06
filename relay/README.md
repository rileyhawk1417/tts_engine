# relay/ — zero-dep TTS daemon client (relay-bot prep)

The relay side of T014's hand-off: the paste-ready client for
[`tts serve`](../docs/notes/resident-service.md), plus the verification
harness that proves it against a live daemon from **Node — the relay bot's
own runtime**.

## Files

| File | What |
|---|---|
| `tts-client.js` | The client. ESM JS + JSDoc — valid TypeScript, zero dependencies (builtin `http` over UDS `socketPath` or TCP `host/port`). Paste into `bot-relay/src/` and it compiles as `tts.ts`. |
| `tts-client.test.mjs` | 14 live checks: health gate, event shapes, warm/cold honesty, per-lane determinism at a seed, speed dial + client-side clamp, error surfaces, timeout, entropy draws. |
| `package.json` | `{ "type": "module" }` so Node executes `tts-client.js` as ESM in *this* repo (relay's package is already ESM). |

## Verify locally

```sh
# 1. boot the daemon (engine side; see docs/usage.md §3):
target/release/tts serve \
  --lane kokoro=models/kokoro-82m-v1.0-onnx/model_quantized.onnx \
  --lane piper=/teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx \
  --socket /tmp/tts/tts.sock

# 2. run the client test from Node:
TTS_SOCK=/tmp/tts/tts.sock node relay/tts-client.test.mjs

# 3. chat-shaped replay bench:
node eval/relay/bench.mjs --sock /tmp/tts/tts.sock --lane kokoro --lane piper \
  --workers 1 --reps 1 --out eval/relay/results/<stamp>
```

Last run (2026-11-06, Node v22): **14/14 passed**, and the byte-identity
convergence check that matters most for relay — seed-42 kokoro WAV assembled
by this client == `tts synth` WAV, sha16 `80241f1c00fc097e` (85 200 B).

## Drop-in plan for the relay repo (governing order: bench row FIRST)

`bot-relay/src/tts.ts` at HEAD `33f1fff` dispatches `TtsEngineId =
"kokoro" | "piper"` with fallback kokoro→piper — worst case flat voice. The
daemon arm slots in **without touching the chain**:

1. `bot-relay/src/types.ts`: extend `TtsEngineId` with `"daemon"` (settings
   dropdown surfaces it; existing guilds keep their value).
2. `tts.ts`: import this client; add a `DaemonRoute` inside `synthesize()`
   **before** the kokoro arm:
   ```ts
   if (opts.engine === "daemon" && await daemonAvailable(DAEMON_SOCK)) {
     try {
       const lane = /* map: kokoroVoice present → "kokoro", else "piper" */;
       const r = await daemonSynthesize(DAEMON_SOCK, {
         lane, text, lang, voice: opts.kokoroVoice, speed: opts.speed,
       });
       return { data: r.wav, format: "wav" };           // SynthResult unchanged
     } catch (err) { console.error("[relay] daemon synth failed:", (err as Error).message); }
   }
   // …existing kokoro arm, then piper fallback — unchanged…
   ```
3. Failure semantics stay relay-shaped: daemon down / 4xx / 5xx → `catch` →
   kokoro arm → piper engine, each independently fallible.
4. Ops: the daemon runs as its **own container process** (like whisper/
   parakeet), never spawned by Node; `--nice 10`, `--idle-unload 0`
   (resident), one listener (`--socket /run/tts/tts.sock`).
5. `speed` maps straight through: client clamps to the per-lane ranges in
   `SPEED_CLAMP` (kokoro 0.5–2.0, piper 0.25–4.0) so the daemon never 400s.

## Status (honest ledger, 2026-11-06)

- [x] Client written + 14/14 live tests green (this repo, Node v22).
- [x] Daemon==CLI byte-identity confirmed **through the client's WAV path**.
- [x] Chat-replay bench harness + first local-approximation numbers
      (`eval/relay/RESULTS.md` — **non-admissible**, 4-core taskset).
- [ ] **relay-bench TTS phase IN-CONTAINER** — T014 done-condition 6; run
      `bench.mjs` against the in-container daemon AND `kokoroWorker` on the
      same lines; that row, alone, is the admissible comparison.
- [ ] Drop-in into `ThyroxDevelopment` (blocked on the row; relay remote
      fetch/push currently SSH-blocked from this machine).
