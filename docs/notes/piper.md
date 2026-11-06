# Piper lane groundwork notes (T011 session-1.5 findings)

Collected 2026-11-05, ahead of the session-2 (`piper` feature) build. Facts verified
against upstream the day they were needed, not from memory.

## Phonemizer: espeak-rs status and role

- Crate `espeak-rs` 0.2.0 lives inside `thewh1teagle/piper-rs` (`crates/espeak-rs` +
  `crates/espeak-rs-sys`). Repo **not archived** (GitHub API: pushed 2026-05-21).
  The standalone `thewh1teagle/espeak-rs` repo 404s — folded into piper-rs; that is
  likely the "archived" impression. MIT wrapper.
- Dep hygiene: `espeak-rs` depends on exactly `espeak-rs-sys` (bindgen + cmake + glob).
  **No `ort` coupling** — piper-rs's pinned `ort =2.0.0-rc.12` never reaches us.
- Architecture: NOT a subprocess. Builds vendored espeak-ng via cmake in-tree, links in-process.
  Runtime provisioning: one `espeak-ng-data` directory (few MB).
- License: espeak-ng GPLv3 → GPL-linking caveat; same class as kokoro's espeak question.
  Alternatives if it ever matters: `espeak-ng-rs` (eugenehp) pure-Rust port, updated
  2026-09-05, fidelity + license unverified (experiment class); phonemizer sidecar
  subprocess (licensing-clean, slight IPC cost).

## G2P is not a perf surface (measured, don't relitinigate)

System espeak-ng 1.52.0, warm in-process calls (ctypes), 20 sentences × 10 reps:

| route | per-sentence cost | one-time |
|---|---|---|
| in-process lib call (warm) | **0.070 ms med / 0.135 ms max** | 15.4 ms init |
| CLI one-shot (process + data load) | 10.0 ms | — |
| reference: amy VITS wall per chunk (rtf 0.075) | 233 ms | — |

- Throughput ~54 kchar/s → >1,400 sentences/s of G2P budget.
- **G2P share of the VITS wall: 0.03%** (even the dumb CLI architecture is ~4%).
- Rule-based G2P: no gemm/thread/quant/session dials. The optimization campaigns on
  this repo (ptts, candle, kokoro, neutts) were all neural-decode stages; G2P is not
  that class. Verdict: phonemizer work = **build integration + licensing + correctness
  (id assembly)**, never speed.
- API fact: system share-lib 1.52.0 exports only `espeak_TextToPhonemes` (3-arg);
  `espeak_TextToPhonemesWithTerminator` is NOT in the shared-build export table.
  Verify in the vendored build; T013 chunking already splits clauses at the text layer,
  so terminator granularity is a nicety, not a dependency.
- **Speed dial (T014 serve-side)**: `SynthRequest.speed` maps to
  `length_scale = 1/speed` clamped to `[0.25, 4.0]` (`speed_options`, `serve.rs`) —
  passes through the existing VITS length-scale dial, no graph or seam change.
  Daemon E2E: speed 1.25 → 3.192 s → 2.774 s of the same sentence (per-phoneme
  integer rounding keeps the ratio off the exact 1.25).

## Three determinism lineages (why our surgery exists)

| lineage | noise source | determinism | seed variety |
|---|---|---|---|
| stock fused graphs (shipped voices) | 2× unseeded `RandomNormalLike` | none — differs every run | random |
| our surgery (noise-input Slice) | caller's seeded rng | byte-identical per seed (proven 3×) | full dial |
| streaming lineage (PR #255) | `torch.manual_seed(1234)`, randn traced to constants | deterministic by bake | **frozen forever** |

- piper itself cannot make byte-identical audio; the streaming lineage is deterministic
  but locks voice character per export. Our noise-input surgery is strictly friendlier
  for eval (gate law) and product (seed is a live dial).
- Streaming lineage mechanics (from `infer_onnx_streaming.py`, fetched 2026-11-05):
  encoder runs once per utterance; decoder chunks of **45 mel frames** (hop 256 →
  0.523 s @ 22050), **chunk_padding CLI default 5 frames/side** (class docstring says
  10 — discrepancy preserved), **pad-and-TRIM** (`audio[start_pad:-end_pad]`), not
  overlap-add; short inputs (≤ chunk + 2·pad frames) skip streaming. Boundary-quality
  prior art: mush42/sonata (alive, 2024), marty1885/paroli (similarity-search concat);
  mush42 himself suggests boundary smoothing/fades.
- piper1-gpl (`OHF-Voice/piper1-gpl`) is the current reference implementation (6,181
  stars, pushed 2026-11-02) — its phonemizer/id-assembly conventions are the porting
  ground truth for 1.0.0-format voices. piper-rs's README credits mush42/sonata as
  inspiration; either way, we port id assembly verbatim from piper source (both
  lineages available), run the graphs in our own ORT sessions, and adopt only espeak-rs.

## Groundwork ledger for the current board

- kokoro / flow lanes: groundwork done (drop files, surgery, parity) — closed.
- piper: **CLOSED phase A (2026-11-05)** — groundwork banked in session-1, lane
  + gates landed in session-2 (see close-out above). Phase B (mel streaming)
  blocked on checkpoint artifacts only.
- chatterbox (T010): groundwork **not done** — the real remaining non-Rust work
  (clone + build chatterbox.cpp, verify C-API surface) ≈ one dedicated session before
  any FFI Rust.
- neucodec: shelved (T012); resident-service (T014): design done, board task
  landed at `docs/plan/tasks/T014-resident-serve.md` (M5), unblocked by piper
  phase A; session-4 speed: optional, decoupled (see resident-service.md).

## Session-2 layer-B close-out (2026-11-05, lane + gates)

Lane shipped: `crates/tts-engine/src/lanes/piper.rs` behind feature `piper`
(ort-dylib session + espeak-rs 0.2 in-process G2P + seeded caller-supplied
noise + T013 text chunks + both io-encode laws). Probe:
`crates/tts-engine/examples/piper_lane_probe.rs`.

Integration gotchas found in build/link/run (none predicted by the notes):

1. **Graph output name is `output`, not `audio`** — the session-1 note
   described the tensor ("out audio f32") and the name got conflated. Lane
   verifies `output` presence and fetches `outs.get("output")`.
2. **Voice-config twin is `<full-filename>.onnx.json`** (both lineages append
   `.json` to the FULL ONNX filename). `.with_extension("json")` is WRONG
   (swaps after the last dot AND drops the `.onnx` component). For patched
   `*-rng.onnx` graphs (no own json) the lane falls back to the original's
   twin (`en_US-amy-medium.onnx.json`).
3. **espeak-rs-sys link patch required**: the vendored CMake build links
   static `libespeak-ng.a` against system `libpcaudio.so` +
   `libsonic.so` (`USE_LIBPCAUDIO=ON`, CMakeCache), but never forwards
   `-lpcaudio`/`-lsonic` to the final binary link → undefined
   `audio_object_*` for every binary. Fixed by
   `crates/tts-engine/build.rs` emitting `rustc-link-lib=pcaudio+sonic`
   under the `piper` feature (workspace binaries all inherit).
4. **Runtime data dir**: `PIPER_ESPEAKNG_DATA_DIRECTORY=/usr/share`
   (system espeak-ng 1.52.0 data matches the vendored lib version).
5. **Probe hash ≠ python-acceptance hash** (`da99672a…` vs `021686cc…`): the
   Rust lane draws noise from its own StdRng stream (ChaCha) with a
   different draw order/format and hashes canonical i16 LE — cross-language
   hash equality was never the contract; in-seam repeatability is. Gated:
   3× byte-identical per seed on both voices, seed 43 differs.
6. Unit-test law check: the 1.0.0 peak-norm law rescales EVERY clip
   (`[0.5,-0.25] → [1.0,-0.5] → [32767,-16383]`), not just clips where
   max|a| < 1 — test initially asserted straight scaling; fixed to the law.

Results (release, seed 42, `eval/run.sh` battery, TTS_ORT_INTRA unset default):

| voice | corpus | n | identity | WER med | WER mean | TTFA med/p95 | seam rtf (audio/wall) | wall/audio | RSS max |
|---|---|---|---|---|---|---|---|---|---|
| amy medium | hard | 500 | **500/500 pass** | 30.95 | 44.07 | 204 / 359 ms | 15.94 | ≈0.063 | 288 MB |
| ryan high | hard | 500 | **500/500 pass** | 20.00 | 33.01 | 663 / 1308 ms | 3.726 | ≈0.268 | 343 MB |

First-80 slices (apples-to-apples vs the hard-80 rows):
amy-80 WER med 40.00 / mean 43.80, TTFA med 238; ryan-80 WER med 22.50 /
mean 32.41, TTFA med 813.

Scoreboard context (same judge `te @38b3d623` for piper + flow rows):
flow-candle hard-80 med 42.86 / flow-ptts hard-80 med 46.43 (both worse than
either piper voice). Kokoro t005a (med 25.00) ran on an older judge sha
(`cdb5b135`) — cross-sha WER comparisons are approximate only; ryans' 22.50
hard-80 med is still the best measured. Long text streaming check: 6
sentence paragraph → 4 T013 chunks (indices 0-3), TTFA 181 ms, 18.99 s
audio in 987 ms wall, 2 runs byte-identical.

**Verdict: ADOPT phase A.** ryan = best measured en_US quality
(judge-identical WER) + 3× faster than kokoro and lower RSS; amy = latency
champion (~0.063 wall/audio, warm TTFA ≈ 200 ms) with the weakest WER of the
two (medium quality tier). Phase B (mel streaming) stays blocked on
checkpoint artifacts.
