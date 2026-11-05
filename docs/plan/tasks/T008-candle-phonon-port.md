# T008 — Candle from-scratch Phonon-31m port (flow-latent step b)

- **Status:** `in-progress` — session 1 (2026-11-05): scaffold banked (commit `48e9552`); transformer/flow/Mimi port in sessions 2+
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

## Progress log
- **Session 1 (2026-11-05, commit `48e9552`):** `candle_ptts` module behind the
  `flow-candle` feature — `cfg.rs` is a faithful serde port of the
  ptts-0.6.0 TTSConfig graph (enums Gelu/TimeRmsNorm/FlowType included);
  `loader.rs` ports the remap table verbatim + mmap safetensors inventory
  (candle `MmapedSafetensors`, headers only — nothing materialized). Lane
  skeleton `flow_candle` passes `load()` and fails `synthesize()` cleanly.
  Probe `flow_candle_probe` audit on models/phonon-31m-en: all knobs
  config-sourced (temp 0.8, lsd_steps 1, eos 4.0, gelu Erf, time_rms Rms,
  flow Lsd), 185 tensors / 183 remap-used / 98 MiB f32, remapped
  backbone/decoder keys present (90). Caught + fixed: samples/frame is the
  oracle's `frame_size` = round(24000/12.5) = 1920 (seanet hop 120 → encoder
  200 fps → downsample stride 16), NOT ratios×kernel (840).
- **Session 2 (2026-11-05, commit `0a9efbe`):** full port modules landed and
  compiling behind `flow-candle` — `nns.rs` (linear/LayerNorm/RoPE with the
  interleaved-pair rotation + offset-from-absolute-position contract),
  `attention.rs` (StreamingTransformer + MhaState KV cache, mask elision at
  t==1 per oracle, GQA expand-before-cache, `gelu_tanh` threaded per
  checkpoint), `mlp.rs` (SimpleMLPAdaLN/ResBlock/FinalLayer/modulate, exact
  erf GELU path, LayerNorm eps 1e-6 vs 1e-5 split), `rng.rs` (NormalRng over
  rand 0.9 + ReplayRng), `tok.rs` (HF tokenizers 0.23), `model.rs`
  (Conditioners/LutConditioner/FlowLM/TTSModel/prompt_text/_null
  /sample_next_latent_cfg/EOS). Lane `load()` is REAL now: cfg ->
  `Weights::from_safetensors` (remap into owned f32 tensors — documented
  deviation: candle 0.11 has no zero-copy path) -> `VarBuilder` ->
  tokenizer.json -> `TTSModel::load`; synth still fails cleanly until the
  decode/Mimi module. Probe audits on phonon-31m-en: remap 185 raw -> 177
  kept (8 `flow.w_s_t` drops) + 183 used-by-inventory (semantics note in
  loader.rs), 177/177 key presence sweep through VarBuilder, TTSModel built
  in ~50 ms (~120 ms full lane load incl. 98 MiB materialize; 0.17 s CLI
  wall, ~225 MB RSS). Load-path bugs found by the CLI round-trip: LUT +
  continuous conditioner `output_proj` needed `.pp("output_proj")`, and
  `out_eos`/`input_linear` needed their submodule prefixes. Deps pinned to
  oracle parity: rand 0.9.0, rand_distr 0.5.0, tokenizers 0.23.1
  (fancy-regex), candle-core/candle-nn 0.11.1. 19+1 tests green, clippy
  clean, workspace release build with graph-ort,flow-ptts,flow-candle ok.

## Done-conditions

- [ ] `tts synth --lane flow-candle ...` produces audio for a corpus sentence, chunks.jsonl complete
- [ ] Self-review checklist vs config.json: every hyperparameter sourced from the checkpoint config, none hardwired by accident
- [ ] Cold load wall time + peak RSS logged (mmap mandate: weights never fully resident unless evidenced as win)
