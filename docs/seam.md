# The tts_engine seam — CLI, data, and telemetry contract

> Design note (T001). Settled decisions live here; no current-state tracking.
> Every lane ships against this contract. Golden example at the bottom is the
> parser reference for T002 — illustrative numbers, not measurements.

## 1. Goal

One seam, many lanes. A lane is any engine shape wired to this contract
(graph/ONNX, flow-latent/candle, ggml FFI, streaming graph). The seam owns:
CLI surface, request/response data shapes, chunk protocol, telemetry
emission, determinism gates, env/dial conventions. Lanes own only their
model math and per-chunk stage timings.

Principle: **telemetry is seam-side, never lane-side.** Lanes report timings
as data; the seam serializes JSONL, computes TTFA/RTF, samples RSS.

## 2. CLI surface (v0)

```
tts synth --lane <name> --model <path-or-hf-id> [--lang <bcp47>] [--voice <id>]
          --text <str | @file | ->  -o out.wav
          [--jsonl chunks.jsonl] [--seed <n>] [--bench] [--allow-hf]
```

- `-o` required (WAV). `--jsonl` optional; if absent, `synth` prints nothing
  machine-readable (human summary only).
- `--bench`: machine mode — JSONL rows go to **stdout**, WAV still written.
  Eval harness (T003) drives the binary via `--bench`.
- `--lang` default `en`; supported set tracked in targets.md (en, fr, de, es, pt).
- `--voice` optional; absent = lane default voice. Values: voice id (`af_heart`),
  registry name (`default`), or path (cloned-voice embed for ggml lane).
- Exit codes: 0 ok · 2 usage · 3 load failure · 4 synthesis failure · 5 io.
- Subcommands beyond `synth` (later): `voices`, `doctor` (env/dial printout).

## 3. Library seam

```rust
pub struct SynthRequest {
    pub lang: Lang,
    pub voice: Option<VoiceRef>,      // id, path, or lane-registered default
    pub text: String,                 // pre-chunked by seam (T013) unless lane overrides
    pub chunks: Vec<TextChunk>,       // shared chunking output (empty => lane-default whole text)
    pub seed: Option<u64>,            // determinism contract, §6
    pub lane_options: LaneOptions,    // per-lane key/value, dials in §7
}

pub struct ChunkOut {
    pub index: u32,
    pub stages: StageTimings,         // per-stage durations below
    pub pcm: Pcm16,                   // i16 LE mono @ stream.sample_rate
}

pub struct SynthStream {
    pub meta: RunMeta,                // sample_rate, determinism_class, lane versions
    pub chunks: Box<dyn Iterator<Item = Result<ChunkOut, LaneError>>>,
}

pub trait TtsLane {
    /// One-time load. Load wall is measured here, separately from synth — never baked into TTFA/RTF.
    fn load(&mut self, model: &ModelRef) -> Result<LoadInfo, LaneError>;
    fn synthesize(&mut self, req: &SynthRequest) -> Result<SynthStream, LaneError>;
    fn stages(&self) -> &'static [StageMapRow]; // declares lane's slot mapping, §5.2
}
```

In-process chunk streaming only for M0; IPC (named pipes / WS) is deferred
and tracked as an open question in the plan index.

## 4. Data shapes

- PCM: mono, 16-bit LE. The lane may compute at any internal rate but must
  emit at `RunMeta.sample_rate` (resample at the lane boundary, never mid-stream).
- `TextChunk`: `{ text: String, byte_range: (u32, u32) }` — byte_range indexes
  the original request text so provenance survives chunking.
- Stage timings are per-chunk cumulative durations spent producing that chunk.

## 5. chunks.jsonl schema

One file, three typed rows (every row carries `"type"`):

**header** (first row) — identity + determinism contract:

```json
{"type":"header","lane":"graph-ort","model":"kokoro-82M-onnx-int8","model_sha":"a1b2c3d4e5",
 "voice":"af_heart","sample_rate":24000,"lang":"en",
 "determinism":"seeded","seed":1234,
 "lane_version":"0.1.0","runtime":"onnxruntime-1.x.y","load_ms":1413}
```

**chunk** rows (one per audio chunk, in flush order):

```json
{"type":"chunk","i":0,"t_synth_start_ms":1491,
 "chunk_ms":{"g2p":3,"acoustic":168,"vocoder":61,"io":2},
 "ttfa_ms":245,"rtf_running":4.898,"peak_rss_kb":317440,
 "pcm_bytes":"sha256:9f86d081884c7d65…"}
```

**summary** (last row) — run totals:

```json
{"type":"summary","chunks":3,"audio_seconds":3.412,"wall_ms":662,
 "rtf_final":5.155,"ttfa_ms":245,"peak_rss_kb_max":317440,
 "seed":1234,"gate":{"byte_identity":"pass","run_id":"2026-11-05T0912-kokoro-golden"}}
```

### 5.1 Field definitions (these definitions are law; parsers assume them)

- `t_synth_start_ms`: wall ms since process start when `synthesize()` was called
  (post-load). Load is the separate `load_ms` header field.
- `ttfa_ms`: wall ms from `synthesize()` call until the **first chunk is handed
  to the output sink** (not until it's computed). Set on chunk `i` where
  `i = 0` of the run, `null` on later chunk rows. Definition excludes load by construction.
- `rtf_running`: cumulative `audio_seconds / synth wall seconds` (excluding
  load), at the moment this chunk flushed. OVER 1 = faster than realtime
  (Gradium-family convention; te's published 0.144-style numbers are the
  inverse — wall/audio — so 1/te_rtf converts). `rtf_final` on summary =
  end-state of the same quantity. Audio seconds = `Σ chunk_i →
  samples/sample_rate`.
- `chunk_ms.g2p | acoustic | vocoder | io`: per-chunk durations, see slot semantics.
- `peak_rss_kb`: max RSS observed so far at flush time (sampler, §5.3).
  Summary carries the run max. Sampler measures `VmRSS` from `/proc/self/status`
  every 10 ms; VmHWM cross-checked at teardown.
- `pcm_bytes`: sha256 of the chunk's exact PCM bytes — byte-identity comparisons
  across runs/lanes are per-chunk, not just whole-file.
- `model_sha`: sha256 (first 16 hex) of the primary weights artifact — ties a
  result row to exact on-disk bytes (te byte-identity lesson).

### 5.2 Stage slot semantics (fixed keys so lanes compare; lanes declare their mapping)

| Slot | Semantics | graph-ort | flow-latent (ptts/candle) | ggml-ffi | streaming-graph |
|---|---|---|---|---|---|
| `g2p` | text → model input prep | espeak/misaki G2P | SentencePiece encode (≈0) | text tokenizer | phonemizer |
| `acoustic` | input → audio-domain intermediate | encoder/PLBERT → mel | flow-LM + flow head | T3 + meanflow CFM | encoder → mel |
| `vocoder` | intermediate → PCM | iSTFTNet | Mimi decoder | S3Gen + HiFT | decoder + overlap-add |
| `io` | resample/convert/flush overhead | — | — | FFI copy | — |

A lane unused slot reports `0`. Renaming slots per lane is forbidden —
comparability is the point.

### 5.3 RSS sampler

One daemon thread per process, 10 ms tick, reads `/proc/self/status`
(`VmRSS`, `VmHWM`). Lanes never self-report RSS. Threads are excluded from
stage timings; their overhead is invisible-correct (sub-µs reads).

## 6. Determinism & byte-identity gates

- Lanes declare `determinism` class in the header: `deterministic`
  (given seed, bit-identical), `seeded` (deterministic given seed + fixed thread
  count), `nondeterministic` (must state why; eval rows rejected unless gate passes).
- Gate (enforced by T003 harness, exercised by lane CI): two runs with the same
  seed/model/voice/text → all `pcm_bytes` shas identical. Gate must pass before
  any number from that lane is quoted (te load-parity discipline).
- Thread-count-sensitive lanes: `peak_rss_kb` and RTF may differ across thread
  configs; PCM bytes must not.

## 7. Env & dials (new dials land here only after measured proof)

- `HF_HOME` — default `/teranux/hugging_face` on this box; hub only touched
  when `--allow-hf` is passed plus `--model <hf-id>`.
- `TTS_ENGINE_MODEL_DIR` — local weights root, default `weights/` (gitignored).
  Model resolution order: exact path → model dir → (explicitly allowed) hub.
- `TTS_ENGINE_THREADS` — worker threads for compute; default physical cores
  (te law). Lanes must honor it or document why they can't.
- mmap mandate: weights map off disk (`memmap2` / ORT external-data / GGUF mmap);
  full-resident loads must prove their wall-time win to ship.

## 8. Chunking policy appendix (T013 hook)

- Seam pre-chunks text via the shared module: sentence split → clause split →
  hard split, target ~50 tokens/chunk, >200-token pieces split at clause marks,
  single words cut between characters (Phonon-derived law, targets.md).
- Lanes may override via `LaneOptions.chunk_policy` but MUST log the override
  in the header row as `chunk_policy:"lane:<name>"`.
- Voice-restart semantics per chunk are the lane's declaration (`restart_voice: true`
  for flow-latent; mel-overlap lanes false) — recorded in `RunMeta`.
- No empty chunks; total text preserved modulo the whitespace policy in T013.

## 9. Worked golden example (parser reference for T002/T003 tests)

> Fictional kokoro run, 4 chunks — the fourth deliberately shows an honest
> `rtf_running` dip. Numbers illustrative. This exact text (canonicalized)
> and row shape live in `eval/fixtures/golden-chunks.jsonl` when T002 lands;
> the fixture test parses it and fails on any schema drift.

```jsonl
{"type":"header","lane":"graph-ort","model":"kokoro-82M-onnx-int8","model_sha":"deadbeef00000000","voice":"af_heart","sample_rate":24000,"lang":"en","determinism":"seeded","seed":20261105,"lane_version":"0.1.0","runtime":"onnxruntime-1.22.1","load_ms":1413,"chunk_policy":"seam-default","restart_voice":false}
{"type":"chunk","i":0,"t_synth_start_ms":1491,"chunk_ms":{"g2p":3,"acoustic":168,"vocoder":61,"io":2},"ttfa_ms":245,"rtf_running":4.898,"peak_rss_kb":317440,"pcm_bytes":"sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}
{"type":"chunk","i":1,"t_synth_start_ms":1491,"chunk_ms":{"g2p":2,"acoustic":144,"vocoder":58,"io":1},"ttfa_ms":null,"rtf_running":5.111,"peak_rss_kb":317440,"pcm_bytes":"sha256:2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"}
{"type":"chunk","i":2,"t_synth_start_ms":1491,"chunk_ms":{"g2p":2,"acoustic":151,"vocoder":57,"io":1},"ttfa_ms":null,"rtf_running":5.155,"peak_rss_kb":319488,"pcm_bytes":"sha256:fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf2fb9"}
{"type":"chunk","i":3,"t_synth_start_ms":1491,"chunk_ms":{"g2p":2,"acoustic":149,"vocoder":60,"io":1},"ttfa_ms":null,"rtf_running":5.148,"peak_rss_kb":319488,"pcm_bytes":"sha256:60303ae22b998861bce3b28f33eec1be758a213c86c93c0761929ff385a79021"}
{"type":"summary","chunks":4,"audio_seconds":4.525,"wall_ms":879,"rtf_final":5.148,"ttfa_ms":245,"peak_rss_kb_max":319488,"seed":20261105,"gate":{"byte_identity":"pass","run_id":"2026-11-05T0912-kokoro-golden-v2"}}
```

Consistency of the sample: chunk flush walls 245/450/662/879 ms against
chunks of 1.200/1.100/1.112/1.113 s audio → running RTF 4.898/5.111/5.155/
5.148 (the last leg dips — 4.525/0.879 = 5.148 — as real jitter allows);
per-chunk stage sums (234/205/211/214 ms) fit their flush deltas.
`gate.byte_identity` is `not-assessed` on live single runs; `pass`/`fail` is
written by the T003 harness after comparing two same-seed runs.

Sanity invariants a parser must assert: header first, summary last, chunk `i`
monotonic from 0, `ttfa_ms` non-null only on `i` where `i=0`, summary `chunks`
== chunk count, `rtf_final` == last `rtf_running`, all `rtf_running` and
`audio_seconds` finite and positive. Note: `rtf_running` is the cumulative
ratio audio/wall; it is NOT contractually monotone — small denominators let
an honest run dip (e.g. 735.3 → 734.7 across two chunks). Only trends
matter at report time; parsers must not reject dips.

## 10. Checklist → T002 acceptance

- [ ] `crates/tts-engine` exports the §3 types + `StageTimings` with the §5.2 slots
- [ ] Telemetry wrapper emits §5 rows from any `TtsLane`; smoke lane (silence) round-trips §9 fixture shape
- [ ] RSS sampler meets §5.3; determinism classes plumbed per §6
