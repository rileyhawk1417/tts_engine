# T009 — Parity gate + q-law experiments (M2 close-out)

- **Status:** `not-started`
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T007](T007-phonon-evaluation.md), [T008](T008-candle-phonon-port.md)

## Goal

Prove the candle port is correct against the ptts oracle, then find the free speed. Two phases, both gated by measured numbers, te style.

## Approach

- Parity: same seed + voice + text through both lanes → per-sample waveform comparison on a sentence battery (spectral distance + round-trip WER deltas). Gate: candle WER within 0.5 pt of ptts on both corpora; any audible-class drift explained (dtype bf16 vs their q8 path is expected to diverge — bound it, don't hide it).
- q-law experiments (each gets its own hypothesis + falsify/adopt verdict): int8 weight-only matmuls vs bf16; DNNL feature where candle exposes it; quantization of flow head vs backbone separately; KV preflight (voice-primed session reuse) timing win; threads=physical-cores; mmap page-cache warmth effects (cold vs warm wall time).
- te transfer check explicitly recorded: does q8_0 ≥ f16 hold for this TTS class? int8-wins-at-m=1 confirmation? Each verdict added to a `docs/notes/qlaws.md` ledger.
- Post-parity performance: RTF/TTFA candle vs ptts lane on the final scoreboard.

## Done-conditions

- [ ] Parity gate passed with documented bound (or divergence root-caused and fixed)
- [ ] ≥4 q-law hypotheses tested with adopt/shelve verdicts in the ledger
- [ ] Final flow-candle row on the scoreboard; M2 default-lane decision re-confirmed or revised
