# Kokoro lane baseline scoreboard — Zen 2 (4800H 8C/16T), 2026-11-05

First honest numbers for `tts_engine` (T005). Machine: one laptop, release
builds, seed 42, load parity respected (runs serialized). Full artifacts in
`eval/results/<run>/` (per-run results.json, rows.tsv, wave files gitignored);
headline rows also live in `eval/results/scoreboard.md` (gitignored, machine
state — the DATED numbers of record are this note and the memory checkpoint).

## Matrix (kokoro lane, model_quantized.onnx int8, af_heart, 24 kHz)

| run date | corpus | n | judge | WER mean % | TTFA mean/p95 ms | RTF mean | RSS max KB | identity |
|---|---|---|---|---|---|---|---|---|
| 20261105-0834 | internal | 8 | nemo int8 | 7.10 | 7396/12223 | 0.785 | 457604 | pass |
| 20261105-1003 | internal | 8 | parakeet q8_0 | 6.06 | 7799/13439 | 0.749 | 457528 | pass |
| 20261105-0939 | hard-80 | 80 | nemo int8 | 42.03 | 5085/9680 | 0.805 | 420056 | pass |
| 20261105-1006 | hard-80 | 80 | parakeet q8_0 | 49.62 | 5091/9248 | 0.803 | 420668 | pass |

Conventions: RTF here = `audio_seconds / wall_seconds` (seam convention; OVER
1.0 = faster than realtime; te publishes the inverse `wall/audio`, i.e. these
runs are wall/audio ≈ 1.18-1.34). Telemetry (TTFA/RTF/RSS) is OUR synthesizer's
`chunks.jsonl` only — judge latency never enters our rows. hard-80 = stride-80
subset of the 500-sentence gradium hard corpus (deterministic regeneration in
`eval/data/hard-80.toml`, gitignored; see `eval/import_hard.py` for provenance).

## Judge decision — nemo int8 is the default judge

Internal corpus is a statistical tie (7.10 vs 6.06 at n=8, same synth runs).
The digit/symbol-heavy hard set separates them: 42.03 (nemo) vs 49.62
(parakeet) — nemo's transcriptions match the literal reference text (esp. in
punctuation/digits) closely enough to score lower round-trip WER. Default =
nemo int8; parakeet q8_0 kept as tiebreak lane (both via `te`; judge identity
sha recorded per run in scoreboard.md).

## What the hard-corpus WER actually means

Per-category round-trip WER (nemo, n=4 each):

- `plain`: **0.00 %** — natural sentences are transcribed verbatim.
- Symbol-heavy categories blow up: `files` 91 %, `mixed` 148 %, `slashed` 111 %,
  `phones` 69 %, `hash` 42 %, `versions` 52 %, `short`/dates/`alnum` 20-63 %.
- Cause is CONVENTION, not engine voice-quality errors: refs like `README.md`,
  `3.5/10`, `555-0182` are SPOKEN in expanded forms ("read me dot md"), and a
  literal-text WER punishes the mismatch. Lanes must be compared under this
  SAME convention; absolute values only have meaning relative to other lanes
  measured identically. Category medians are preserved in results.json.

## Performance shape (measured facts a later lane must beat or falsify)

1. **TTFA is compute-bound, not load-bound.** `--repeat 3` in-process runs:
   TTFA 3018 / 3001 / 3064 ms with a single 741 ms load — the ~3 s TTFA of a
   short sentence is the first acoustic chunk's graph run, not ORT init.
   Warm pooling only helps TTFA for lanes whose load dominates (not kokoro).
2. **RTF ≈ 0.75-0.86** on this box (int8, intra=8 threads default from
   `physical_core_estimate`, inter=1). Longer audio ≈ same RTF (chunks amortize).
3. **RSS grows with chunk count within one run** (ORT arena per graph run):
   ~250 MB after 1-chunk runs, ~458 MB for a 5-chunk paragraph. A resident
   warm pool must bound this (session reset or arena cap) — flagged for the
   future service-lane work; not a lane-law violation of the seam.
4. **Stage split is honest**: `g2p` 13-250 ms/chunk (espeak spawn ~12 ms +
   phonemize + punct restore), `acoustic` = whole graph run incl. in-graph
   iSTFT (dominates), `vocoder` = batch concat + pause insertion (~0-2 ms),
   `io` f32→i16 ~0-1 ms.

## M2 targets derived (Phonon-31m flow-latent lane; to be re-stated per lane)

- TTFA: kokoro's ~3.0 s cold TTFA on short text is the bar. Phonon's
  self-reported streaming (40-100 ms) would beat it 30×; measured target
  ≤ 500 ms on identical runs, else falsify the self-report and record ours.
- RTF: phonon target < 0.5 kokoro-RTF on same corpus/convention (≥ 2× faster
  than kokoro's 0.79); only adopt with proof (te law: CPU-time tiebreak if RTFs
  within noise).
- WER: within ±0.5 pt of kokoro's normalized round-trip WER on internal; 0 % on
  `plain` category; do not regress the hard-80 convention row.
- Determinism byte-identity gate `pass` required for any lane to publish rows.

## Harness notes logged this session (already fixed in run.sh)

- `--repeat N`: rep-indexed outputs `*.repN`; rep 0 is cold (pays load).
- Aggregate step hardened: rows with missing metric cells no longer crash
  aggregation (None instead of KeyError).
- Corpus file → clean dir name (basename stem, no path slashes in stamps).
- Judge flag runtime-gated: `te` help shows `--gguf` even without the
  parakeet compile feature — the grep precheck passes but the run fails; the
  head-to-head needed `TE_FEATURES=nemo,parakeet scripts/build.sh build`
  (+ `TE_PARAKEET_LIB=/teranux/parakeet.cpp/build/libparakeet.so`).
## 2026-11-05e — flow-ptts gate-dial correction (T007)

The T006 flow-ptts numbers were measured in a corrected class (see
notes/phonon-eval.md). PTEXT, q8, TTS_XN_THREADS=1 (seeded, byte-identity PASS):

| row | TTFA | rtf | note |
|---|---|---|---|
| flow-ptts cold (gate) | 3050 ms | 0.72 | first-stream init ~2.7 s lands here |
| flow-ptts warm (resident) | 392 ms | ~1.74 | beats kokoro rtf 2x at 1 thread |
| kokoro int8 (gate ref) | 3118 ms | 0.84 | unchanged |

f32 row (T009): TTFA 53 ms probe / 252 ms lane, rtf 3.36, but FAILS
byte-identity at threads=1 (nondet emissions) — ungated, experimental dial
TTS_PTTS_QUANT=f32. Default stays q8.


## 2026-11-05h — hard-500 flow-ptts complete (T009, untrusted gate as measured)

Run `eval/results/20261105-1152-flow-ptts-hard-t007` (q8, default threads,
seed 42): 500/500 rows, **identity 489 pass / 11 FAIL → verdict untrusted
(gate law held — the 11 FAILs are the known threads>1 xn nondeterminism
[q-law L2]; FAIL rows cluster in low-RTF/load windows)**.

PASS-only medians (489 rows, nemo judge):

| metric | value |
|---|---|
| WER med | 40.00 |
| raw WER med | 60.00 |
| TTFA med | 2314 ms (mean ~2250) |
| rtf med | 0.949 |
| peak RSS med | 177 MB |

Category med WER (pass rows): percents/plain_dates/versions 0.00 ·
ordinals/quantities/times 16.67 · abbreviations/units 20.00 ·
hash_numbers 33.33 · currency/alnum_codes/order_numbers/ranges 40–42.86 ·
math_symbols 46.43 · slashed_dates 50.00 · temperature 60.00 ·
mixed_fields 111.11 · short_replies 150.00 · phones 157.14 ·
files_urls 100.00.

Reading: q8 flow-ptts decodes the digit-heavy hard corpus but WER medians
are dominated by normalization-judge disagreements on formatted text
(phones / short_replies / mixed_fields >100 = judge transcript mismatch,
not just audio quality — same pattern as kokoro hard-80 42.03/49.62 but
larger). Trusted verdict + M2 row needs the `TTS_XN_THREADS=1` rerun
(seeded path, measured rtf 0.72 / identity PASS on 1t text).

## 2026-11-05k — T009 close-out: M2 finished (hard-500 1t + parity battery)

Trusted threads=1 hard-500 rerun: `eval/results/20261105-1540-flow-ptts-hard-t009-1t`
— 500 rows, **493 pass / 7 FAIL identity** (verdict untrusted as measured;
FAIL root cause below). WER stats unchanged vs default-threads.

| hard-500 flow-ptts 1t | med | mean |
|---|---|---|
| WER / raw (all 500) | 40.00 | 56.98 |
| WER / raw (493 pass) | 40.00 / 60.00 | 56.94 |
| TTFA | 2226 ms | 2238 |
| rtf | 1.054 | 1.099 |
| peak RSS | 177 MB | — |

**Identity FAIL root cause (both runs)**: NOT xn FP order — the ptts
decode-batch **grain race**. Row 129: 57 vs 58 chunks across reps; the same
sentence is 4-for-4 byte-identical standalone; FAIL rows are random across
runs (7 ≠ 11, disjoint sets). Under load, drain timing shifts chunk
boundaries → per-chunk voice restart changes bytes. WER/duration unaffected.
Fail budget: 0 % (hard-80/1t, idle) · 1.4 % (hard-500/1t) · 2.2 % (500/8t).

### M2 close-out scoreboard (hard-80, nemo judge, seed 42, 80/80 identity all)

| lane (quant) | WER med | WER mean | rtf med | TTFA med | RSS max | identity |
|---|---|---|---|---|---|---|
| kokoro int8 ORT (t005a ref) | 25.00 | 42.03 | 0.80 | 4596 ms | ~457 MB | pass |
| flow-ptts q8 1t | 46.43 | 64.24 | 1.163 | 2085 ms | 182 MB | 80/80 |
| **flow-candle f32** | **42.86** | **60.28** | **0.334** | **352 ms** | 219 MB | 80/80 |

Internal n=8 (nemo): candle 0.00/0.00/7.69/108.33/0/10/0/0 — 6 rows **exact
match** with ptts t006, `address` outlier root-caused (candle verbalizes the
address — norm-correct rendition; judge transcribed ptts's as digits; candle
@seed7 → WER 0.00). Excl. address: candle 2.53 vs ptts 3.00 (Δ 0.47 pt).

Parity battery: probe 48-step latents ≤ 1e-4; cross-lane PCM (candle-f32 vs
ptts-q8): same plans/durations, rms|d| 0.0086 ("a.") / 0.029 (fox), med ≈ 0,
p99 4.4 % / 11.6 %, max 0.145 / 0.54 — q8-vs-f32 weight delta amplified by
Mimi; bounded, not hidden.

**M2 verdict**: parity target MET (candle ≈ ptts WER everywhere except the
explained outlier; better on hard-80 mean/median). TTFA target MET (352 ms,
13× kokoro). WER-vs-kokoro on the hard corpus regresses for the flow-latent
class (42.86 vs 25.00 med — both flow lanes, so model-class behavior, not a
port artifact). Wall-RTF gap open (0.334 vs 0.80, ≈2.4× slower) → session-4
(decode batching + gemm). **Kokoro stays default lane; flow-candle becomes
the resident-latency lane.** M2 closed 2026-11-05.

### Landmine found during close-out

`eval/run.sh` rebuilt the binary with only `graph-ort,flow-ptts` → silent
**feature strip** (flow-candle gone, 12.8 MB, mtime lies via hardlink
relink). Fixed: run.sh now builds all three features on every invocation —
the feature list is the only truth; size/mtime are not staleness signals.

## 2026-11-05m — T012 NeuTTS measurement: backbone clears budget, lane SHELVED

Design corrections found on contact: NeuCodec is **50 tokens/s** (FSQ) not
12.5 Hz; Neuphonic ships official **neucodec-onnx-decoder / -int8** (gated:auto)
→ phase-2 integration = existing graph-ort machinery, not a custom export.

Phase-1 rates (llama.cpp 0.7.0-dev, 4800H/16T Zen 2, fixed seed):

| source | quant | gen t/s | vs 50 Hz | RSS |
|---|---|---|---|---|
| our box, ungated Qwen2-0.5B proxy (identical qwen2 path) | Q4_0 | 215–285 | 4.3–5.7× | 876 MB |
| our box, same | Q8_0 | 229 | 4.6× | 1.05 GB |
| authors, official neutts-air Q4_0 (llama-bench, HX 370) | Q4_0 | 119 | 2.4× | — |

Decode is 1-core-effective at this size (user 0.93 s ≈ wall-gen; rate flat
across -t 1..16). Load ≈ 1.3 s mmap vs community torch load 174.93 s.

**Verdict**: CPU-realtime claim CORRECTED (backbone ≥2× budget on this class;
public "15× slower" is the Python/transformers chain — load 174.93 s, voice
encode 74.56 s). **Lane SHELVED**: 3rd decode stack (T010 already tests
ggml/FFI), gated artifacts (no token on box → codec decode rate unmeasured),
voice-encode chain has no ONNX artifact, flow-candle owns resident-latency,
kokoro owns default. Numbers in T012 task file; targets.md row updated.

Proxies: `~/models/t012-neutts/qwen2-0_5b-instruct-{q4_0,q8_0}.gguf`.

**T012 upgrade — official artifacts measured** (`/teranux/hugging_face/hub`
mount had `neutts-air-Q4_0.gguf` 527 MB + `neutts-nano-Q4_0.gguf` 195 MB,
both qwen2-arch): air **217.4 t/s = 4.35× the 50 Hz budget**, RSS 1.07 GB,
wall 3.01 s; nano **342.4 t/s = 6.8×**, RSS 744 MB. Proxy rows stand as
corroboration. The Q4_0 file is 527 MB, not the 748 MB originally assumed.
Codec-decode rate stays unmeasured (`neucodec-onnx-decoder` gated and absent
from the mount; two torch-neucodec pip attempts blew the time-box — the
1.16 GB neucodec pth IS on the mount for a later pass). Verdict unchanged:
SHELVE as lane; claim corrected.

**T012 phase-2 measured — codec decode through OUR graph-ort machinery.**
User gated in (`hf auth` as r1l37); artifacts dropped to
`/teranux/hugging_face/t012-drop/` (int8 312 MB, fp32 782 MB; also piper
amy/medium + ryan/high → T011 model logistics prepped). Graph contract
verified exact: codes int32 [b,1,T] → audio f32 [b,1,480·T−480] (24 kHz).
Bench via new example `neucodec_ort_bench.rs` (ort rc15 api-21 + ORT
1.22.1 dylib — kokoro-identical path), t=8, 20 reps:

| decoder | 50 tok (1 s) | 250 tok (5 s) | tok/s @250 | × budget | RSS |
|---|---|---|---|---|---|
| **int8** | **77.3 ms** | 335.9 ms | 744 | **12.9–14.9×** | 556 MB |
| fp32 | 101.9 ms | 413.1 ms | 605 | 9.8–12.1× | 1.44 GB |

Marginal decode ≈ 1.3 ms/token (int8) — never gates a ~20 ms backbone step.
Rate value-independent (all-zero ids verified). **Chain picture complete**:
backbone-bound at ~4.3× realtime (217 t/s), head hidden; the public "15×
slower" is now fully attributed to the Python stack. Verdict unchanged:
SHELVE as lane (third stack, gated supply, voice-encode torch-only, slots
filled — and corrected 4.3× profile would regress WER like the other flow
lane). Voice encode remains unmeasured by deliberate scope (one-time setup
cost, torch-only, no ONNX artifact).

## 2026-11-05n — T011 piper lane: hard-500 500/500 identity PASS, ADOPT (phase A)

Both drop voices, Rust lane (`lanes/piper.rs`, feature `piper`, espeak-rs
0.2 in-process G2P, seeded caller-noise, T013 chunks, release, seed 42,
nemo-int8 judge `te @38b3d623`):

| voice | corpus | n | identity | WER med | WER mean | TTFA med/p95 ms | seam rtf (audio/wall) | wall/audio | RSS max |
|---|---|---|---|---|---|---|---|---|---|
| piper amy medium (hard) | hard | 500 | **500/500 pass** | 30.95 | 44.07 | 204/359 | 15.94 | ≈0.063 | 288 MB |
| piper ryan high (hard) | hard | 500 | **500/500 pass** | 20.00 | 33.01 | 663/1308 | 3.726 | ≈0.268 | 343 MB |

First-80 slices (apples-to-apples vs the hard-80 rows): amy-80 WER med
40.00 / mean 43.80 / TTFA med 238; ryan-80 WER med 22.50 / mean 32.41 /
TTFA med 813. First lane with a **500/500 identity pass at any thread
count** — the ORT session + per-sentence caller-supplied noise (graph
surgery from session-1) has no grain race: chunking = whole text or T013
chunks, rng advances across chunks deterministically within a process AND
across processes (2 CLI runs byte-identical on a 6-sentence, 4-chunk text).

Judged-context (same judge shas only): piper rows and the flow rows ran on
`te @38b3d623` → ryan 22.50/20.00 vs flow-candle 42.86 / flow-ptts 46.43
(hard-80 meds) — piper ryan is the best measured WER on this judge.
Kokoro t005a med 25.00 ran on `te @cdb5b135` — cross-judge-sha, treat as
approximate (ryan-80 med 22.50 would still be best if it maps 1:1, which is
plausible given nemo's stability, but not proven within one judge sha).

**Verdict: ADOPT phase A.** ryan = the en_US quality+speed lane (best WER
among same-judge rows, 3.7× seam rtf, RSS 343 MB; every axis measured at or
better than kokoro). amy = the resident latency champion (seam rtf 15.9×,
warm TTFA ≈ 200 ms, RSS 288 MB) at a visibly weaker WER (medium-quality
voice). Kokoro stays the board's default-lane reference (same-judge
caveat + non-en voices); re-run kokoro on `38b3d623` if the en_US default
switch is ever adjudicated. Phase B (mel streaming) remains blocked on
checkpoint artifacts, not engineering.

## 2026-11-06 — T010 ggml-ffi (chatterbox) lane: hard-80 trusted, cloning capability adopted

`lanes/ggml_ffi.rs` + `src/shim/chatterbox_shim.cpp` (llama.cpp
`mtmd_helper::gen_audio` in-process; pinned `186eab1`), Chatterbox Turbo
Q8_0 gguf, shipped voice ("Boba", bundled conds.pt), `TTS_CBX_THREADS=4`,
release, seed 42, nemo-int8 judge `te @38b3d623` (same judge sha as the
piper table above):

| voice | corpus | n | identity | WER med | WER mean | TTFA med/p95 ms | seam rtf (audio/wall) | RSS max |
|---|---|---|---|---|---|---|---|---|
| chatterbox turbo default | hard | 80 | **80/80 pass** | 60.00 | 67.54 | 8344/9725 | 0.406 | 2,494,480 KB |

Same-judge-sha (`38b3d623`) side-by-side, hard-80 slice: ryan 22.50 (500-run
slice) ≪ amy 40.00 (slice) < flow-candle 42.86 < flow-ptts 46.43 (idle
budget) ≪ **chatterbox 60.00 — last place on the WER board**. Expressive
T3 sampling over whisper-derived semantic tokens reads poorly on the
adversarial hard corpus; this is the shipped-voice baseline, not a tuned
one. Speed is mid-pack (audio/wall 0.406: ≈2× slower than kokoro 0.80,
faster than flow-candle 0.334). RSS ≈ 2.4 GB peak = the T014-resident
deviation (≈8× the ≤300 MB/lane warm budget; lane stays CLI/serve-own-rule).

Verdict (M3): **ADOPT as the board's voice-cloning lane (the only one);
NOT a default-TTS quality contender.** Cloning done-condition met through
the CLI: `--voice <wav>` (18.99 s piper reference) → verbatim round-trip
WER (nemo-int8), 2× byte-identical across invocations under concurrent
eval load. Probe gates: 3× shipped identity (sha `507c9ea1…`, stable across
probe and CLI processes), 2× clone identity, shipped-voice restore after
clone (no sticky speaker), thread dial t8 ≠ t4 (expected). Cloned runs
include the speaker preproc inside the prompt stage (wall 8–18 s for a
4-s sentence depending on contention). **Hard-500 NOT run:** ≈3.5 h at
~30 s/row; hard-80 is 80/80 and the adoption case (cloning) does not hinge
on hard-500; if the cloning voice ever becomes a production path, run
hard-500 then (the ptts-class FAIL-budget amendment applies only to ptts
rows; llama-class reduction order is fixed per thread dial and per-chunk
KV resets make cross-row state impossible).

## 2026-11-06 — T015 pocket-tts as a second flow-candle checkpoint: hard-80 trusted, fastest flow-class row

Run `20261106-0141-flow-candle-hard-t015-pocket-alba` (hard-80, seed 42,
nemo-int8 `te @38b3d623437347e0`, alba fallback voice): **80/80 byte-identity
pass → trusted.** WER mean **37.06** / median 38.75; TTFA med 503 / p95
521 ms; rtf med **0.220** (wall/audio ≈ 4.5× — the fastest flow-class number
measured); peak RSS 666,376 KB. Determinism 3× byte-identical standalone
(`43976644dbf857ea…`); phonon-31m regression byte-equal to pre-T015 outputs.

Same-judge hard-80 comparison (mean, lower is better): ryan-high 33.01·(500)
≪ **pocket-alba 37.06** ≪ kokoro 42.03 ≪ phonon-candle 60.28 ≪ ptts 64.24 ≪
chatterbox-cbx-t4 67.54. Pocket-alba WER mean beats every flow-latent row and
sits between piper ryan and kokoro; the adversarial-outlier tail (WER max
125) is smaller than its siblings'. This replaces phonon-31m as the
best-behaved flow-latent voice; 31m stays the Apache-2.0 portability
reference and pocket's weights are CC-BY-4.0 (code MIT).

Verdict (T015): **ADOPT** — pocket-tts-en is the second checkpoint family of
the flow-candle lane (`--model models/pocket-tts-en`); the ptts-class
FAIL-budget amendment applies to ptts rows only (candle inline decode is
grain-stable: 0% identity FAIL at idle here, and the hard-500 caveat was a
ptts decode-batch effect). Note the lane's `model` telemetry label is the
weights filename (`model.safetensors`) — disambiguate checkpoint family via
the run dir / config.

## 2026-11-06 — T014 `tts serve` resident daemon: shipped, daemon==CLI byte-identical

Not lane-eval rows (eval stays CLI-only by law); these are the daemon's
harmony/lifecycle gates on the 4800H, release, seed 42
(`notes/resident-service.md` holds the wire detail):

| lane (daemon warm) | ttfa | rtf | harmony vs CLI | notes |
|---|---|---|---|---|
| kokoro int8 | 4,204–4,641 ms | 1.28–1.32 | **sha `6bea4778…` byte-identical** — even across intra dials 4 vs 16 (intra = perf dial only for this graph) | 3.525 s audio, RSS ~390 MB |
| piper amy-rng | **197–278 ms** | 0.062–0.087 | **byte-identical at the same `--threads` dial**; ACROSS dials bytes differ (VITS matmuls are intra-sensitive) | 3.192 s audio; speed 1.25 → 2.774 s |
| flow-candle (pocket-alba) | 512 ms | ~0.22 | **byte-identical** | 33 chunks; warm RSS 600–786 MB; speed refused (flow lane) by design |

Lifecycle: lazy boot 15 MB RSS (`loaded:false`) → cold 508 ms honest → warm
(no `load_ms` key on the wire) → idle > 4 s → `loaded:false`, `resets:1`
(arena freed). NDJSON newline INSIDE the HTTP chunk payload (a bare `\r\n`
is transport framing that curl/python strip). Relay-side in-container bench
row = the one OPEN T014 condition (hand-off in notes/resident-service.md).

## 2026-11-06 — flow-candle on CUDA (session chunks 2+3): device knob, GPU rows, quality gate

Device seam: `TTS_CANDLE_DEVICE` (synth env, default `cpu`; `cuda` needs cargo
feature `flow-candle-cuda` = candle-core/cuda **plus candle-nn/cuda** — nn's
own cuda feature is what carries its softmax/sigmoid/rms kernels; layer-norm
got a manual device-branch route `nns::ln_forward` mirroring candle-nn's
fallback math, CPU keeps the library op byte-faithful). `tts serve --device
cpu|cuda` validates at boot and forwards via the same env; cuda on a
non-cuda build fails lane load loudly (operator config error).
`determinism_class` stays `seeded` — byte-identity gates are **per-device**
(same binary+seed+device): GPU bytes ≠ CPU bytes BY DESIGN, never compare
cross-device shas. RunMeta `runtime` carries the device label. Relay builds
stay NON-cuda (cuda-linked binary carries ~+190 MB host RSS even on CPU runs;
peak VRAM 369 MiB of 4096). No per-process JIT cold penalty (kernels compile
at build time). GPU cold load: 469 ms (serve boot, honest `load_ms`).

Single-clip smoke (phonon-31m-en, 11.1 s, seed 42): GPU rtf ≈ 5.0 = **15.6×
CPU (0.327)**, warm TTFA 25–50 ms. Eval rows (nemo-int8 te@38b3d623, seed 42,
`TTS_CANDLE_DEVICE=cuda`, run.sh swaps the feature itself, identity all pass →
trusted):

| corpus | GPU (this session) | CPU reference | TTFA GPU | rtf GPU |
|---|---|---|---|---|
| hard-500 | **57.42** (`20261106-1029-flow-candle-hard-gpu`) | flow-ptts 56.94/56.98 untrusted (identity FAIL) | — | — |
| hard-80 | **62.99 / med 42.86** (`20261106-1113-flow-candle-hard-80-gpu`) | 60.28 / med 42.86 (`…t009-parity`) | 42 ms mean | 4.78 |

Paired per-sentence (80/80 common, same seed+judge): **75/80 judged
identically**; 4 worse / 1 better; median delta 0.00. The +2.71 mean shift is
digit-normalization style drift, not intelligibility loss — GPU pronounces
`4502-1187-33902` as `four five zero two one one eight seven three three nine
zero two` and `march 14th` as `march fourteenth, twenty twenty six` where CPU
grouped digits; all five moved sentences sit at ≥ 50 WER on CPU already
(number/date classes, the checkpoint's known weak tail). **Quality gate: GPU
row passes the WER-class gate** — same class as CPU, 16× faster, at the price
of a new per-device byte class.
