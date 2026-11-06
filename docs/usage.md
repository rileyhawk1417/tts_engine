# Usage — `tts synth`, `tts serve`, and the eval harness

> Practical reference for operating the engine. The wire/telemetry **contract**
> lives in [`docs/seam.md`](seam.md) (CLI + chunks.jsonl) and
> [`docs/notes/resident-service.md`](notes/resident-service.md) (daemon
> protocol as shipped); the lane-by-lane research record lives in
> `docs/notes/*.md`. This file is the how-to.

---

## 0. Build

The workspace has two crates: `tts-engine` (the library + lanes) and `tts`
(the CLI + daemon). Lanes live behind cargo features — **the feature list is
the only truth** (the binary size is your staleness signal):

```sh
# The canonical gate build (fmt → clippy -D warnings → tests → e2e):
./scripts/build.sh

# Or build by hand — always release for real runs (debug xn is ~8× slower):
cargo build --workspace --release \
  --features graph-ort,flow-ptts,flow-candle,piper,ggml-ffi

# Binary: target/release/tts   (all five lanes ≈ 33 MB)
```

Ort needs the 1.22.1 runtime shared library; resolve it with
`TTS_ORT_DYLIB=/teranux/ort-1.22.1/libonnxruntime.so` (or wherever your
`libonnxruntime.so` lives) — the ONRT-graph lanes probe the default
locations first.

---

## 1. `tts synth` — one-shot CLI (the ONLY eval/bench surface)

```
tts synth --lane LANE --model MODEL
          [--lang en] [--voice NAME|PATH]
          --text "text" | - | @file.txt
          -o out.wav
          [--jsonl out/chunks.jsonl | --bench]
          [--seed N] [--allow-hf] [--repeat N]
```

| Flag | Meaning |
|---|---|
| `--lane` | `smoke` \| `kokoro` \| `flow-ptts` \| `flow-candle` \| `piper` \| `ggml-ffi` |
| `--model` | Disk path (mmap-loaded) or, with `--allow-hf`, a hub id. **Per-lane quirks below.** |
| `--lang` | Language tag; defaults `en` (`fr de es pt` also parsed) |
| `--voice` | Named voice id, a path to an embedding/KV file, or omit for the lane default |
| `--text` | Inline text, `-` for stdin, or `@relative/path/to/file.txt` |
| `-o` / `--out` | Output WAV (mono 16-bit at the lane's sample rate); parent dirs auto-created |
| `--jsonl PATH` | Telemetry sink: one header row, N chunk rows, one summary row |
| `--bench` | Same telemetry rows to **stdout** (mutually exclusive with `--jsonl`) |
| `--seed N` | Set for anything you will compare; without it seeded lanes draw entropy (the byte-identity gate only runs with a seed) |
| `--allow-hf` | Allow explicit hub ids; first use downloads into the HF cache |
| `--repeat N` | N syntheses in one process (rep 0 = cold, pays load; warm-pool measurement) |

Exit codes: `0` ok · `2` usage · `3` load · `4` synth · `5` io.

### Examples

```sh
B=target/release/tts

# Smoke lane — honest silence, mechanics-only (no weights):
$B synth --lane smoke --model smoke --text "hello" -o out/s.wav --jsonl out/s.jsonl --seed 42

# Kokoro (ONNX graph lane) — model is the .onnx FILE, voices live in <dir>/voices/*.bin:
$B synth --lane kokoro \
  --model models/kokoro-82m-v1.0-onnx/model_quantized.onnx \
  --voice af_heart --text "Hello there." -o out/kokoro.wav --seed 42

# flow-candle, phonon-31m family (dir with config.json + safetensors + tokenizer):
$B synth --lane flow-candle --model models/phonon-31m-en \
  --text "Hello there." -o out/phonon.wav --seed 42

# flow-candle, pocket family (second checkpoint in the SAME lane):
$B synth --lane flow-candle --model models/pocket-tts-en \
  --voice alba --text "Hello there pocket." -o out/pocket.wav --seed 42

# piper — fused VITS graph; must be an *-rng.onnx (seeded-noise patched, see §5):
$B synth --lane piper --model /teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx \
  --text "Hello piper." -o out/amy.wav --seed 42

# chatterbox (cloning lane) — backbone gguf, mmproj-*.gguf beside it:
$B synth --lane ggml-ffi --model /teranux/hugging_face/t012-drop/chatterbox/backbone-q8_0.gguf \
  --text "Hello chatterbox." -o out/cbx.wav --seed 42 \
  # + (optional) --voice /path/to/reference.wav  (clone from a ≥5 s WAV)
```

Telemetry of a run (`--jsonl`): `header` → one `chunk` row per chunk
(`i, t_synth_start_ms, ttfa_ms (first chunk only), wall_ms, bytes, …`) →
`summary` (`load_ms, te…`). TTFA = wall to first PCM byte; RTF is published
by the seam as `audio/wall`; `load_ms` is NEVER baked into TTFA.

---

## 2. Lanes — what `--model` / `--voice` expect, and the dials

| Lane | Checkpoint family | `--model` | `--voice` | Sample rate |
|---|---|---|---|---|
| `kokoro` | Kokoro-82M int8 ONNX | path to the **`.onnx` file** (dir → `model file not found`) | voice id from `voices/*.bin` (default `af_heart`) or `.bin` path | 24 000 |
| `flow-ptts` | phonon-31m via the `ptts` crate (oracle lane) | dir with `model.q8.gguf` + config | voice embedding `.safetensors` or named | 24 000 |
| `flow-candle` | from-scratch candle port; **two families**: `phonon-31m-en` and `pocket-tts-en` | model **dir** with `config.json` | named (e.g. `alba`, `michael` — resolved from `embeddings/`/`voices/`), an embedding `.safetensors` path, or omit (registered default) | 24 000 |
| `piper` | fused VITS (rhasspy piper voices) | **`*-rng.onnx`** seeded-noise-patched graph; the `.onnx.json` voice config must sit next to it | n/a (voice identity is the graph pair) | 22 050 |
| `ggml-ffi` | Chatterbox Turbo GGUF (llama.cpp FFI) | backbone `.gguf`; a single `mmproj-*.gguf` beside it (or lane option `mmproj=<path>`) | path to a reference WAV (clone) or omit → bundled voice | 24 000 |
| `smoke` | none (diagnostic) | literally `smoke` | n/a | 24 000 |

### Environment dials

| Var | Lane(s) | Effect |
|---|---|---|
| `TTS_ORT_DYLIB` | graph lanes | Path to `libonnxruntime.so` (1.22.1) |
| `TTS_ORT_INTRA` | graph lanes | ORT intra-op threads; default = physical-core estimate |
| `TTS_XN_THREADS` | `flow-*` | xn scheduler threads (pinned on the ptts lane for determinism) |
| `TTS_PTTS_QUANT` | `flow-ptts` | Quantized weights variant selector |
| `TTS_CBX_THREADS` | `ggml-ffi` | llama.cpp threads for the chatterbox pipeline |
| `TTS_ENGINE_SEED` | all (CLI) | Fallback seed when `--seed` is absent |
| `TTS_ESPEAK` | `piper`/kokoro G2P | Override espeak-ng data dir |

The byte-identity gate law: **re-run it per (lane, quant, dial) triple.**
Piper VITS matmuls and ptts decode are intra-thread sensitive; kokoro's
graph is not.

---

## 3. `tts serve` — the resident daemon (interactive surface)

One binary, two modes: `tts synth` stays the measurement surface; `tts serve`
loads lanes **once** and streams synthesis over HTTP/1.1 (NDJSON chunks) on a
Unix domain socket by default (TCP opt-in). Relay's pattern: a dumb, honest
server — the client owns sequencing and backpressure.

```
tts serve --lane NAME=MODEL [--lane NAME2=MODEL2 …]
          --socket PATH | --tcp HOST:PORT
          [--lazy] [--idle-unload S] [--threads N] [--nice N]
```

| Flag | Meaning |
|---|---|
| `--lane NAME=MODEL` | Repeatable. Same `NAME`/`MODEL` semantics as `synth` (§2). Loads at boot unless `--lazy`. |
| `--socket PATH` | UDS listener (owner-scoped filesystem permissions). Relay hosts the daemon in the same container. |
| `--tcp HOST:PORT` | TCP listener instead (opt-in). Pass **not both** listeners; the daemon refuses. |
| `--lazy` | Defer lane loads to first request — boot stays instant (~15 MB RSS). |
| `--idle-unload S` | Unload a lane after S idle seconds; the reload is a fresh, honestly-reported `mode:"cold"` (0 = resident forever). |
| `--threads N` | Pin the ORT intra-op dial (fail-fast) — **must match your CLI dial for piper byte-parity.** |
| `--nice N` | `setpriority(2)` niceness (relay runs `-n 10`). |

Boot log example:

```
serve: lane "models/kokoro-82m-v1.0-onnx/model_quantized.onnx" loaded in 745 ms
serve: lane "/teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx" loaded in 210 ms
serve: listening on unix:/tmp/tts/tts.sock — lanes kokoro,piper, idle-unload 0s, lazy false
```

### 3.1 `POST /synth`

Request body (all fields optional except `text`):

```json
{
  "lane":  "kokoro",      // omit on a single-lane daemon (first lane wins)
  "text":  "Hello there.",
  "lang":  "en",          // en fr de es pt
  "voice": "af_heart",    // voice id or embedding path; omit = lane default
  "seed":  42,            // set for byte-reproducible responses
  "speed": 1.25,          // optional rate dial — lane-dependent (below)
  "chunk_policy": null    // reserved (chunking policy override)
}
```

Response: `200 OK` with the body streamed as **NDJSON events** —
`meta` → one `chunk` per PCM event → `done`. The payload is HTTP
chunked-encoded, and the newline that separates events is **inside** each
chunk payload, so after transport decoding you simply split on `"\n"`.

**meta** (shape as shipped; `load_ms` appears only on a true cold request —
it is *omitted from the JSON on the wire* on warm hits, never `null`):

```json
{"type":"meta","protocol":"ndjson-chunks/1","lane":"piper","model":"…-rng.onnx",
 "model_sha":"…16 hex…","voice":"en_US-amy-medium","lang":"en","seed":42,
 "sample_rate":22050,"runtime":"ort","chunk_policy":"sentence",
 "determinism_class":"seeded","nondet_reason":null,"restart_voice":false,
 "mode":"cold","load_ms":210,"resets":0,
 "rss_after_kb":288210,"vm_peak_kb":343000}
```

**chunk** — `pcm_b64` is little-endian mono i16 at `meta.sample_rate`,
base64-padded (RFC 4648):

```json
{"type":"chunk","seq":0,"pcm_b64":"/wG9AQ==","bytes":6,"from":0}
```

**done**:

```json
{"type":"done","chunks":3,"audio_s":3.192,"wall_ms":277.4,"ttfa_ms":197.0,
 "rtf":11.51,"mode":"warm","resets":0,"sample_rate":22050,
 "rss_after_kb":288210,"vm_peak_kb":343000}
```

RTF convention on the daemon matches the seam: `rtf = audio/wall`
(larger = faster); `ttfa_ms` is wall to the first chunk emission, the same
honesty laws as the CLI (`load_ms` only on cold).

The simplest two-line smoke test:

```sh
curl --unix-socket /tmp/tts/tts.sock -sS -X POST http://localhost/synth \
  -d '{"text":"Hello there.","seed":42}' -o resp.ndjson
cat resp.ndjson | python3 -c "import sys,json,[print(e['type']) for e in map(json.loads,sys.stdin.read().splitlines())]"
```

### 3.2 `GET /health`

```json
{"type":"health","pid":4123,"idle_unload_s":4,"threads_dial":16,
 "rss_after_kb":452000,"vm_peak_kb":470000,"lang_default":"en",
 "lanes":[{"lane":"kokoro","model":"…/model_quantized.onnx","loaded":true,
           "resets":0,"idle_ms_ago":1200}]}
```

### 3.3 Errors

Honesty to the end: **plain HTTP errors before any streaming starts**, NDJSON
`error` events only after `meta` has gone out mid-stream.

| Code | When |
|---|---|
| 404 | unknown path (`message` names the POST/GET surfaces) |
| 405 | known path, wrong method — use `POST /synth` |
| 400 | malformed JSON / missing or empty `text` / unsupported `lang` / speed out of range / lane without a speed dial | 
| 409 | unknown `lane` (message lists served lanes), or lane present but not loaded |
| 413 | body over the 4 MB cap (text heads capped at 16 KB) |
| 500 / 422 | lane load / synthesis failure (lane error text passed through) |

### 3.4 The `speed` dial (per lane)

Semantics are **rate** (bigger = faster) uniformly on the wire; per-lane
mapping and clamps are documented and hard-refused where unimplemented:

| Lane | Mapping | Range | Refused? |
|---|---|---|---|
| `kokoro` | direct rate dial | 0.5–2.0 | — |
| `piper` | `length_scale = 1/speed` (VITS duration scale) | 0.25–4.0 | — |
| `flow-*`, `ggml-ffi`, `smoke` | — | — | **hard error** (400) — no dial, pass no `speed` |

Measured: piper speed 1.25 took a 3.192 s utterance to 2.774 s (daemon E2E).

### 3.5 Determinism: the daemon/CLI harmony law

`serve` reuses the same lane cores as `tts synth` — **no daemon-only code
paths** — so:

- **kokoro** daemon output == CLI output byte-for-byte, *even across intra
  thread dials* (the graph is intra-insensitive here; `--threads` is a
  performance dial only).
- **piper** daemon == CLI **at the same intra dial**; bytes differ across
  dials (VITS matmuls). If you compare, pin `--threads N` on the daemon and
  the matching `TTS_ORT_INTRA=N` (same N) on the CLI side.
- **flow-candle** (either family) daemon == CLI byte-identical.
- `flow-ptts` is the ptts-class exception: long-text decode has a documented
  grain-race fail budget under load (~1.4% at 1 thread, ~2.2% at >1 on
  hard-500). Treat it as warm-only/CLI-eval class (eval runs stay CLI).

### 3.6 Lifecycle mechanics

- Worker loop = **one thread per lane**; requests serialize per lane by
  design (relay's own measurement: worker pools beyond one buy little and
  cost ~300 MB each on kokoro).
- The idle clock stamps at **request end** (streams materialize lazily —
  a long synth cannot be "overtaken" by its own idle timer).
- Unloads free the weights arena (`resets` increments; surfaced in `/health`
  and every `meta`/`done`).
- With `--lazy`, boot loads nothing; the first request pays load and reports
  `mode:"cold"` with an honest `load_ms`.

### 3.7 A zero-dependency Node client

Node's builtin `http` handles Unix-socket paths and chunked transfer
decoding; since the NDJSON newline is inside each payload, a byte-concat +
line split is the whole protocol:

```js
const http = require("node:http");

function ttsSynth(socketPath, body) {
  return new Promise((resolve, reject) => {
    let bufs = [];
    const req = http.request(
      { socketPath, path: "/synth", method: "POST",
        headers: { "content-type": "application/json" } },
      (res) => {
        res.on("data", (b) => bufs.push(b));
        res.on("end", () => {
          const events = Buffer.concat(bufs).toString("utf8")
            .split("\n").filter(Boolean).map(JSON.parse);
          const meta = events[0], done = events[events.length - 1];
          const pcm = Buffer.concat(events
            .filter((e) => e.type === "chunk")
            .map((e) => Buffer.from(e.pcm_b64, "base64")));
          resolve({ meta, done, pcm });   // wrap pcm into a WAV on your side
        });
        res.on("error", reject);
      });
    req.on("error", reject);
    req.end(JSON.stringify(body));
  });
}
```

TCP daemons are the same call with `host`/`port` instead of `socketPath`.

---

## 4. Evaluation (never through the daemon)

Round-trip WER via the sibling `transcription_engine` judge — the adoption
gate for every lane:

```sh
# internal corpus (mechanics + quick QA):
eval/run.sh --lane flow-candle --model models/pocket-tts-en --limit 80

# hard corpus (500 sentences), trust gate:
eval/run.sh --lane piper \
  --model /teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx \
  --corpus hard --seed 42 --tag t011-amy

# resume an interrupted run:
eval/run.sh --out eval/results/<stamp>-<tag> …
```

- Per sentence: synth twice (same seed) → **`cmp` byte-identity gate** →
  `te` transcribes synthA → WER vs source → metrics from `chunksA.jsonl`.
- Verdicts: a row whose identity gate FAILs is kept but marked `untrusted`
  (and on hard-500, any FAIL ⇒ verdict untrusted by law).
- Outputs land in `eval/results/<stamp>-<tag>/` (`rows.tsv`, `results.json`,
  per-sentence workdirs) and are appended to `eval/results/scoreboard.md`.
- The daemon is a **consumer**, never a measurement surface: no load parity,
  no byte-identity discipline there. Benchmark the daemon in its host
  container separately (that is the open relay-bench row).

---

## 5. Troubleshooting (the measured gotchas)

| Symptom | Cause → fix |
|---|---|
| `model file not found: … (pass e.g. models/kokoro-82m-v1.0-onnx/model_quantized.onnx)` | Kokoro (and piper) want the **`.onnx` file**, not the model directory. |
| `graph … is missing inputs ["noise_0","noise_1"] — stock piper graph` | Piper graphs need seeded-noise graph surgery: `python3 eval/patch_piper_rng.py <model.onnx> <out>-rng.onnx`, then load the `*-rng.onnx`. |
| Daemon and CLI produce different bytes (piper) | **Intra-dial mismatch.** Start the daemon with `--threads N` matching the CLI's `TTS_ORT_INTRA` (= physical cores by default). Kokoro is insensitive; piper is not. |
| Every NDJSON event merged onto one line | You split on the raw HTTP chunk boundaries. Decode the chunked transfer **first** (Node `http`, curl, and `requests` all do it), then split on `"\n"` — the newline is inside each payload. |
| `lane "nonexistent" not served` / 409 | The daemon serves only the lanes given at boot; check `GET /health` → `lanes[]`. |
| `meta` shows `mode:"cold"` unexpectedly after a pause | The lane idled out (`--idle-unload`); the reload is honest. Raise the timeout or pass `0` to stay resident. |
| Nothing written but stderr was silenced | Never `2>/dev/null` a synth/serve you are debugging — load errors only go to stderr. |
| Speed refused with 400 | That lane has no rate dial (§3.4); only kokoro and piper accept `speed`. |
