# examples/node-client/ — zero-dependency Node client for `tts serve`

An example showing **any Node.js app** how to talk to the resident daemon
([`tts serve`](../../docs/usage.md)) with zero dependencies — Node's builtin
`http` speaks chunked NDJSON over a unix domain socket (`socketPath`) or TCP
(`host`/`port`). Plain ESM JavaScript + JSDoc, i.e. valid TypeScript: paste
`tts-client.js` into a TS project and it compiles as-is.

Wire/protocol details: [`docs/notes/resident-service.md`](../../docs/notes/resident-service.md).

## Files

| File | What |
|---|---|
| `tts-client.js` | The client. `daemonSynthesize(conn, req)` → `{wav, pcm, meta, done}`; `daemonAvailable(conn, {timeoutMs, cacheMs})` health gate; WAV assembly from LE mono i16 PCM; per-lane `speed` clamps (the daemon 400s out of range); hard request timeout; availability cache. |
| `tts-client.test.mjs` | 14 live checks against a running daemon: health gate, event shapes, warm/cold honesty, per-lane determinism at a seed, speed dial + client-side clamp, error surfaces, timeout, entropy draws. |
| `package.json` | `{ "type": "module" }` so Node executes the ESM file as-is. |

## Try it

```sh
# 1. boot the daemon (see docs/usage.md §3) — paths are yours to fill in:
target/release/tts serve \
  --lane kokoro=<models-dir>/kokoro-82m-v1.0-onnx/model_quantized.onnx \
  --lane piper=<models-dir>/piper/en_US-amy-medium-rng.onnx \
  --socket <socket-path>          # the examples default to /tmp/tts/tts.sock

# 2. run the client test from Node:
TTS_SOCK=<socket-path> node examples/node-client/tts-client.test.mjs

# 3. chat-shaped replay bench (see eval/relay/RESULTS.md for the row ledger):
node eval/relay/bench.mjs --sock <socket-path> --lane kokoro --lane piper \
  --workers 1 --reps 1 --out eval/relay/results/<stamp>
```

Last full harness run (2026-11-06, Node v22): **14/14 passed**, including the
byte-identity convergence check that matters most for any client — a seeded
kokoro WAV assembled by this client is byte-identical to `tts synth`'s WAV at
the same (seed, dial).

## Notes for embedding the client in an app

- The client owns: WAV assembly (44-byte mono 16-bit LE header + PCM),
  `SPEED_CLAMP` per lane (kokoro rate 0.5–2.0, piper 0.25–4.0 — mirror any
  user-facing speed meaning before sending), request timeout, health gate
  with a cache window. It does NOT own chunk planning — the daemon chunks.
- Connection is either `{ socket: "/path/to/tts.sock" }` or
  `{ host, port }`; pick by configuration, not code.
- Seeded requests (`seed` field) are byte-reproducible for a fixed
  (binary, model, dial); unseeded draws entropy per request.
- Failure handling is the embedder's responsibility: the daemon is a dumb,
  honest server — treat any non-2xx or timeout as "daemon unavailable" and
  route to whatever fallback your app defines.
