# T008 — Candle from-scratch Phonon-31m port (flow-latent step b)

- **Status:** `not-started`
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T006](T006-phonon-ptts-lane.md) (ptts lane = correctness oracle)

## Goal

Our own implementation of Phonon-31m in candle — weights mmap'd straight off disk via `MmapedSafetensors`, decode policy ported exactly, all telemetry ours. This is the tuning lane te-style: no black boxes.

## Approach

- Loader: bf16 safetensors from `gradium/phonon-31m-en`, lazy per-layer paging (M2 lessons transfer), matmul contiguity + `broadcast_add` + `index_select` candle gotchas honored.
- Modules: SentencePiece (4001 vocab, raw text, no G2P — our chunking feeds it), backbone causal transformer (6 layers, 512 width, 8 heads, RoPE, GQA config per checkpoint config.json), flow head (`lsd` type, one-step Euler from noise ε ~ N(0, T²), temp 0.3), EOS head (`eos_threshold`), Mimi decoder (1920 samples/frame → 24 kHz PCM).
- Decode policy: exact port — chunking (T013), voice latents (num_frames, 512) prepended, per-chunk voice restart. Nothing "improved" silently; deviations are experiments on top, labeled as such.
- KV cache + conv streaming state kept in a `Session` struct (our own, shape-documented; parity-checked against ptts primed sessions).
- Mimi decoder in a shared module (no lane-specific deps) so PocketTTS checkpoint (targets.md lane 2b) can ride the same decoder later.

## Done-conditions

- [ ] `tts synth --lane flow-candle ...` produces audio for a corpus sentence, chunks.jsonl complete
- [ ] Self-review checklist vs config.json: every hyperparameter sourced from the checkpoint config, none hardwired by accident
- [ ] Cold load wall time + peak RSS logged (mmap mandate: weights never fully resident unless evidenced as win)
