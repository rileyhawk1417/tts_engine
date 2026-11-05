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
