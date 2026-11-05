# T009 — Parity gate + q-law experiments (M2 close-out)

- **Status:** `in-progress (q-law ledger done; T008 parity probe GREEN 2026-11-05: 48-step candle==oracle; hard-500 finished 489/11 → untrusted at default threads, threads=1 rerun for trusted M2 numbers pending)`
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
- **Waveform-level parity battery + WER gate still open** (approach section
  above) — the honest grain labels (`frame:1920` / `ptts-decode-batch(max16)`
  in lanes) make lane-row WER comparable via identical chunk plans.

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

## Done-conditions

- [ ] Parity gate passed with documented bound (or divergence root-caused and fixed)
- [ ] ≥4 q-law hypotheses tested with adopt/shelve verdicts in the ledger
- [ ] Final flow-candle row on the scoreboard; M2 default-lane decision re-confirmed or revised
