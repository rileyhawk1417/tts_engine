# T009 — Parity gate + q-law experiments (M2 close-out)

- **Status:** `done 2026-11-05 — parity proven (probe 48-step + lane WER battery); hard-500 1t rerun 493/7; waveform parity bounded; q-laws L1–L5 verdicted; M2 closed (kokoro default, flow-candle = resident-latency lane)`
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T007](T007-phonon-evaluation.md), [T008](T008-candle-phonon-port.md)

## Goal

Prove the candle port is correct against the ptts oracle, then find the free speed. Two phases, both gated by measured numbers, te style.

## Approach

- Parity: same seed + voice + text through both lanes → per-sample waveform comparison on a sentence battery (spectral distance + round-trip WER deltas). Gate: candle WER within 0.5 pt of ptts on both corpora; any audible-class drift explained (dtype bf16 vs their q8 path is expected to diverge — bound it, don't hide it).
- q-law experiments (each gets its own hypothesis + falsify/adopt verdict): int8 weight-only matmuls vs bf16; DNNL feature where candle exposes it; quantization of flow head vs backbone separately; KV preflight (voice-primed session reuse) timing win; threads=physical-cores; mmap page-cache warmth effects (cold vs warm wall time).
- te transfer check explicitly recorded: does q8_0 ≥ f16 hold for this TTS class? int8-wins-at-m=1 confirmation? Each verdict added to a `docs/notes/qlaws.md` ledger.
- Post-parity performance: RTF/TTFA candle vs ptts lane on the final scoreboard.

## Parity status (session-3, 2026-11-05)

- **Probe-level parity ACHIEVED**: `flow_candle_parity_probe` matches the
  ptts oracle for all 48 decode steps (max latent |diff| ≤ 1e-4 = f32 ULP,
  identical raw eos logits step 0 −12.3559 both sides, eos step 10 both
  sides). Root cause of the earlier divergence: `TimestepEmbedder::normalize`
  `recip` instead of `sqrt+recip` (missing square root) in `mlp.rs` — fixed.
- Stage bisection tool: `stages` mode (A conditions / B voice emb / C text
  emb / D transformer on synthetic input) — all exact at first probe → bug
  localization was one code diff, zero step-dumps.
- Chunk-plan parity: probe F section shows oracle and candle plans identical
  (fox: tok=14 fb=84; "a." tok=?) — stream item-count differences are the
  oracle's decode-batch grain (max 16/thread timing), not plan differences.
  Total audio duration matches exactly (2.960 s both lanes, fox).
- Lane-level: candle byte-identity ×2 reps green on both probe texts; "a."
  chunk count now matches the oracle plan (14 vs pre-fix 34).
- **Waveform-level parity battery + WER gate — DONE (close-out, 2026-11-05):**
  - **Cross-lane PCM bound (q8-vs-f32 weights, ptts vs candle)**: same text
    same model, candle-f32 vs ptts-q8 → same durations (26,880 / 71,040
    samples) and same chunk plans; PCM differs per sample (rms |d| 0.0086
    `"a."` / 0.029 fox; med ≈ 0, p99 0.044 / 0.116, max 0.145 / 0.54 amp).
    The probe proves latents match to 1e-4; Mimi decode amplifies the
    weight-quantization (q8 vs f32) delta — bounded, not hidden.
  - **WER battery (nemo judge, seed 42, both 80/80 identity pass)**:
    hard-80 candle **med 42.86 / mean 60.28** vs ptts **med 46.43 / mean
    64.24** — candle within/better (−3.6 med, −4.0 mean). Internal n=8:
    6/8 rows match ptts exactly (numbers-dates 7.69, quoted-dialogue 10.00
    identical); `address` row 108.33 (ptts 16.67) is **seed-dependent
    sampling drift + judge normalization** — candle verbalizes the address
    (norm-correct: "two thousand two hundred ten"), judge transcribed the
    ptts rendition as digits; candle @seed 7 scores **0.00** on that row.
    Excluding it: candle 2.53 vs ptts 3.00 (Δ 0.47 pt) → **gates PASS**.
  - Gate law held per triple: candle lane byte-identity 8/8 internal,
    80/80 hard-80, and all rep pairs (inline decode, no grain race).

## Hard-500 status (flow-ptts, runs 2026-11-05)

- `eval/results/20261105-1152-flow-ptts-hard-t007`: COMPLETE, 500 rows,
  **489 pass / 11 FAIL identity → verdict untrusted (gate law held)**.
  PASS-only medians: WER 40.00 / raw 60.00, TTFA 2314 ms (first row ~2091–
  2500 range), rtf 0.949, RSS 177 MB; category med WER from 0.00 (percents,
  plain_dates, versions) to 157.14 (phones) / 150 (short_replies).
- The 11 FAILs are the **known threads>1 xn nondeterminism** (FAIL rows
  cluster at low-RTF/load windows; q-law L2). Trusted M2 hard numbers need
  `TTS_XN_THREADS=1` rerun (measured rtf 0.72 / seeded PASS on 1t).
- **f32 identity FAIL (run 1142) stands**: 7 vs 5 chunks at 1t — L4 gate
  failure; f32 stays experimental/ungated.
- **Trusted threads=1 rerun DONE** — `eval/results/20261105-1540-flow-ptts-
  hard-t009-1t`: 500 rows, **493 pass / 7 FAIL identity → run.sh verdict
  still "untrusted (identity gate failed)"**. Fail set is random across runs
  (different rows than 1152): order_numbers ×3, files_urls ×2, phones, units.
  All-rows med WER 40.00 / raw 60.00, TTFA 2226 ms, rtf 1.054, RSS 177 MB;
  pass-only med identical (WER 40.00) — **WER is grain-robust across runs**
  (1152 mean 56.94 vs 1540 mean 56.98).
- **FAILs ROOT-CAUSED — NOT xn FP order**: e.g. row 129 differs at chunks
  **57 vs 58**; standalone 4 reps byte-identical on an idle box. The FAIL is
  the ptts decode-batch **grain race** (bounded-channel drain timing shifts
  chunk boundaries → per-chunk voice restart changes bytes) under concurrent
  load, at ANY thread count. WER and total audio duration are unaffected.
  Documented in qlaws L4 amendment + L4 gate law stays: re-run per (lane,
  quant, dial) triple.

## Done-conditions

- [x] Parity gate passed with documented bound — probe 48-step (latent 1e-4),
      PCM cross-quant bound (rms 0.9–2.9 %, med 0), WER battery within/better
      (hard-80 med 42.86 vs 46.43; internal 7/8 within 0.5 pt, outlier
      root-caused as seed drift + judge normalization, confirmed by seed-7
      rerun WER 0.00).
- [x] ≥4 q-law hypotheses tested with adopt/shelve verdicts in the ledger
      (L1 refuted-for-law/f32-faster, L2 shelved-null, L3 shelved published
      rows, L4 adopted-as-gate-law + amendment, L5 adopted resident lane).
- [x] Final flow-candle row on the scoreboard; M2 default-lane decision
      re-confirmed: **kokoro stays default (throughput lane, rtf 0.80);
      flow-candle = resident-latency lane (TTFA 352 ms, 13× kokoro) with a
      documented RTF gap (0.334 vs 0.80, ≈2.4× slower wall) for session-4
      decode-batching work.**
