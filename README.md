# tts_engine

A model-agnostic, CPU-first TTS engine in Rust: **many engines, one seam**.
Every lane speaks the same contract — load → `text, lang, voice → WAV +
per-chunk telemetry` — so lanes are built, measured, and judged *identically*,
and a program never has to know which engine is on the other side of the
interface.

Shipped surfaces (one binary, `target/release/tts`):

- **`tts synth`** — one-shot CLI: the eval, benchmarking, and scripting surface.
- **`tts serve`** — resident daemon: lanes loaded once (or lazily), synthesized
  per request over HTTP/1.1 + NDJSON on a Unix domain socket (TCP opt-in), with
  honest cold/warm telemetry, idle-unload, a `speed` dial, and byte-identical
  (with the CLI) output.

Full usage — flags per lane, wire protocol, speed dials, determinism laws,
eval harness, troubleshooting — in **[docs/usage.md](docs/usage.md)**.

## Why this was built

Three reasons, in ascending order of ambition.

**1. Interactive latency is a residency problem, not an RTF problem.** Our
Discord relay ("Thyrox") hosts every heavy engine resident — a warm
whisper.cpp pool, a warm `kokoroWorker`, a lazy parakeet process — because a
batch engine that re-loads per utterance loses to a warm resident pool on
1–8 s inputs no matter how fast it synthesizes. That lesson was proved in the
sibling [`transcription_engine`](../transcription_engine) (STT side); `tts
serve` is the TTS side of the same architecture: a dumb, honest engine host a
bot can keep in its container, with worker-loop-per-lane, `--nice`, RSS
telemetry, and idle-unload so it shares a 3.5-CPU / 2 GB budget politely.
(`kokoroWorker` inside the bot works today; this repo is the engine-side
successor — an external daemon instead of an in-process Node pipeline, so the
same binary serves any client.)

**2. The 2025–26 open-weight TTS zoo needed an honest bake-off, not vibes.**
Kokoro, Phonon, PocketTTS, Piper, Chatterbox, NeuTTS — every repo quotes
different numbers on different GPUs with different judges. This engine exists
to pin everything: same seed, same chunk plan, same round-trip WER judge
(our own `te`, nemo int8 by default), same telemetry laws (TTFA never bakes
in `load_ms`, RTF = audio/wall, RSS per run), and a hard **byte-identity
determinism gate** that must pass before any WER row is trusted. The
measure-first culture is inherited from the `te` project: *adopt with proof,
falsify and shelve otherwise* — one of six candidate engines was measured
through the seam and honestly shelved (NeuTTS, backbone-bound at ~4.3×
realtime).

**3. The seam must stay boring while engines change underneath it.** Lanes
load weights off disk (mmap-first — never fully resident unless measurement
says so), chunk long text at sentence boundaries, stream chunks as they
finish, and report every stage honestly. The CLI and the daemon share the
same lane cores with no daemon-only code paths, so `daemon == CLI`
byte-identity is a *tested* property, not a hope. Ports are faithful first:
the from-scratch candle port of Phonon passes 48-step parity against the
oracle before any "improvement" is even allowed to be labeled an experiment.

## What's inside

| Crate | What |
|---|---|
| `crates/tts-engine` | The seam library: lane trait, chunking, telemetry (`chunks.jsonl`), WAV I/O, and the lanes |
| `crates/tts` | The binary: `tts synth` (CLI) + `tts serve` (resident daemon, ~1050 lines) |
| `eval/` | Round-trip WER harness (`run.sh`), corpus definitions, the piper RNG graph-surgery patcher, WER scorer |
| `docs/` | Seam contract (`seam.md`), usage guide (`usage.md`), per-lane notes (`docs/notes/*.md`), the build plan (`docs/plan/`) with every task and decision recorded |
| `scripts/build.sh` | The gate: fmt → clippy `-D warnings` → tests → release build → smoke e2e + byte-identity check |

### Lanes (all behind one seam; features: `graph-ort,flow-ptts,flow-candle,piper,ggml-ffi`)

Verdicts are measured (hard-80 unless noted; same round-trip WER judge,
seed 42, Zen-2 4800H CPU, release builds). RTF here = audio/wall (bigger is
faster):

| Lane | Engine | Verdict | Headline numbers |
|---|---|---|---|
| `kokoro` | Kokoro-82M int8, ONNX Runtime | **ADOPT** — quality default, the reference row | WER med 25.0 / mean 42.0 · rtf 0.8 · ttfa ≈ 4.6 s · 457 MB |
| `flow-candle` (phonon-31m) | from-scratch candle port (48-step parity vs oracle) | **ADOPT** — resident-latency anchor | WER med 42.9 / mean 60.3 · rtf 0.33 · ttfa 352 ms · 219 MB |
| `flow-candle` (pocket-tts-en) | kyutai pocket-tts as a second checkpoint family in the same lane | **ADOPT** — best flow-class quality | WER mean 37.1 / med 38.8 · rtf 0.22 · ttfa 503 ms · 666 MB |
| `piper` (amy medium / ryan high) | fused VITS graphs + seeded-noise graph surgery + in-process espeak-rs | **ADOPT phase A** — fastest lane; first 500/500 identity at any thread count | amy: rtf ≈ 16× realtime, warm ttfa ≈ 200 ms; ryan: WER med 20.0 hard-500 |
| `ggml-ffi` | Chatterbox Turbo via llama.cpp FFI | **ADOPT as the voice-cloning lane only** (last-place WER, ~2.4 GB) | WER med 60.0 / mean 67.5 |
| `flow-ptts` | `ptts` crate oracle lane (phonon) | **CLOSED** — parity reference; ptts-class decode grain race documented | hard-500 fail budget ~1.4% @1t / ~2.2% @>1t → WER rows untrusted |
| (measurement) NeuTTS | llama.cpp-class qwen2 backbone + ONNX codec | **SHELVED as lane** | backbone ~4.3× realtime → backbone-bound |

Board status: **every build-plan task is done** (M0 foundation → M5 daemon;
NeuTTS measured and shelved, chatterbox adopted for cloning only); the one
open condition is measuring the daemon in the relay's container — relay-side
work.

## Quick start

```sh
# Build (feature list is the only truth):
./scripts/build.sh

# Synthesize (kokoro; model is the .onnx FILE, voices from <dir>/voices/):
target/release/tts synth --lane kokoro \
  --model models/kokoro-82m-v1.0-onnx/model_quantized.onnx \
  --voice af_heart --text "Hello there." -o out/hello.wav --seed 42

# Serve (UDS default; TCP opt-in):
target/release/tts serve \
  --lane kokoro=models/kokoro-82m-v1.0-onnx/model_quantized.onnx \
  --lane piper=/teranux/hugging_face/t012-drop/piper/en_US-amy-medium-rng.onnx \
  --socket /tmp/tts/tts.sock --idle-unload 300 --nice 10

# Then from any process in the same container:
curl --unix-socket /tmp/tts/tts.sock -sS -X POST http://localhost/synth \
  -d '{"lane":"piper","text":"Warm piper, ~200 ms to first audio.","seed":42}'

# Or evaluate a lane honestly (synth twice → byte-identity gate → WER):
eval/run.sh --lane flow-candle --model models/pocket-tts-en --corpus hard
```

Full reference: **[docs/usage.md](docs/usage.md)** · Wire contract:
[docs/notes/resident-service.md](docs/notes/resident-service.md) · Seam spec:
[docs/seam.md](docs/seam.md)

## Laws this repo runs on

- **Byte-identity before trust.** Any evaluation row is only trusted if two
  seeded synth passes are bit-identical; verdicts state the fail budget
  measured for their lane class.
- **Honest telemetry.** `load_ms` is reported separately, never baked into
  TTFA; the daemon reports `mode` cold|warm and emits `load_ms` only on a
  true cold request (absent key on the wire, never `null`).
- **Faithful ports.** Nothing is "improved" silently; deviations are labeled
  experiments with a measured delta.
- **mmap-first memory.** Weights stay mapped, never fully resident, unless
  evidenced otherwise.
- **`tts synth` is the only measurement surface.** The daemon serves; it
  never judges.

## Acknowledgements

This engine stands entirely on the open-weight TTS zoo and a lot of shared
infrastructure. Licenses as verified during the build (see
`docs/plan/targets.md`):

- **hexgrad/Kokoro-82M** (Apache-2.0) — the quality baseline; consumed via
  the onnx-community/Kokoro-82M-v1.0-ONNX int8 artifacts.
- **kyutai-labs Poling-TTS / Phonon** (`gradium/phonon-31m-en`, Apache-2.0)
  and the **ptts** reference implementation used as the parity oracle; the
  `xn` CPU scheduler underneath it. Phonon-40m weights are CC BY-NC 4.0 and
  are deliberately not a target.
- **kyutai-labs PocketTTS** — code MIT; english checkpoint weights CC-BY-4.0
  (our `pocket-tts-en` uses the *without-voice-cloning* release, voice-cloning
  weights with the encoder zeroed). The Mimi codec module inside the flow lane
  ports their decoder layout (verified equivalent).
- **rhasspy Piper** (MIT runtime) and the **Piper voices** (amy medium, ryan
  high — per-voice MIT files); `espeak-ng` (via system data dir and
  `espeak-rs`) for G2P, `pcaudiolib`/`sonic` linked at build time.
- **Resemble AI Chatterbox** (MIT) — the cloning lane, via the
  **ggml-org Chatterbox-Turbo-GGUF** conversion and **llama.cpp** (MIT) whose
  `tools/tts` pipeline we drive through a small C++ shim; bundled shipped
  voice provenance recorded per `targets.md`.
- **candle** (0.11.1) — the flow lane's tensor engine; **ort**
  (2.0.0-rc.15) + **onnxruntime 1.22.1** — the graph lanes' runtime;
  **tokenizers**, **rand**, and the usual Rust ecosystem crates.
- **espeak-ng** — shared G2P substrate for kokoro and piper.
- The sibling **transcription_engine** project and its `te` binary — the
  round-trip WER judge (nemo int8 default) and the source of most of this
  repo's measurement discipline.

Model and voice licenses are recorded per artifact in
[`docs/plan/targets.md`](docs/plan/targets.md); bundled voices follow the
per-voice-terms rule. This repository itself is private/unreleased — no
license is granted until published.
