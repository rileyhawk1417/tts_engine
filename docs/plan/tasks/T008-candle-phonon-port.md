# T008 — Candle from-scratch Phonon-31m port (flow-latent step b)

- **Status:** `done` — session 3 (2026-11-05): parity proven step-for-step vs oracle, lane e2e + byte-identity green, config audit 25/25
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
- **Session 3 (2026-11-05):** decode + Mimi + synth driver landed. Oracle
  reading covered mimi.rs decode path, plan.rs chunking, synth.rs driver; port
  modules `conv.rs`/`seanet.rs`/`resample.rs`/`attention.rs` (MimiKvCache
  append+trim)/`mimi.rs`/`plan.rs`/`synth.rs` written; conv bias is
  broadcast-added (candle convs carry none); `conv_transpose1d` dilation must
  be `1` not `0` (candle's `l_out` collapses the kernel term at 0 — bug #4).
  Lane `synthesize()` e2e wired: weights_label/sha + load_ms into RunMeta.
  **Runtime bug chain (each found via probe, fixed):** (1) 0-frame input
  `Linear::forward` errors → pass text embeddings directly when latents empty;
  (2) `complete_kv` must return the WHOLE filled prefix `[0,current_end)`, not
  the last frames; (3) `Tensor::full` infers dtype from value — pass bare f32
  not f64-converted; (4) upsample convtr dilation 0→1; (5) **the parity
  root cause**: `TimestepEmbedder::normalize` used `.recip()` (1/(mean+eps))
  where the oracle Rms divides by √(mean+eps) — rewrote as
  `sqrt()?.recip()?` in both Rms and RmsVar branches; time embeddings were
  off by ~1e5, poisoning every AdaLN cond.
  **Parity proven:** `flow_candle_parity_probe` (same fixed `Vec<f32>` replay
  noise, shared oracle-tokenizer chunk) matches the ptts oracle for all 48
  steps — latent max|diff| ≤ 1e-4 (pure ULP), identical eos_l (step 0 −12.3559
  both sides), eos at step 10 both sides. Stage bisection (A conditions / B
  voice / C text emb exact, D transformer at f32 noise) pinned the sampler in
  two probes; step-tracing never needed.
  **Chunk-grain root cause:** candle lane and oracle produce IDENTICAL chunk
  plans (probe F section: tok=14 fb=84 both sides); the 37-vs-13 ChunkOut
  difference is oracle SpeechStream yielding per decode-batch drain (up to 16
  latents, thread-timing dependent) vs our per-frame inline decode. Honest fix
  = truthful labels: candle `chunk_policy: "frame:1920"`, ptts
  `"ptts-decode-batch(max16)"`.
  **Post-fix lane numbers (byte-identical ×2 reps each):** "a." → 14 chunks /
  1.120 s / ttfa 390 ms / rtf 0.309 / RSS 212 MB (matches oracle 14-chunk
  plan — pre-fix produced 34); fox → 37 chunks / 2.960 s / ttfa 377 ms /
  rtf 0.314 / RSS 217 MB — total audio EXACTLY matches oracle 2.960 s.
  flow-ptts fox: 13 chunks / 2.960 s / rtf 2.491 / RSS 253 MB (same duration,
  different grain).
  **Config audit:** 25/25 hyperparameters sourced from config.json
  (d_model 512/8/6/2048/rope 10000, n_bins 4000, lut/flow 512/512, depth 6,
  ldim 32, rms, erf, lsd 1, eos 4.0, temp 0.8, mimi 24k/12.5/512/32/[6,5,4],
  mimi-tf 512/8/2/2048) + conditioners/fuser tables match.
  **Done-condition 3 (load + RSS):** cold load 0.17 s wall, ~225 MB RSS load /
  ~218 MB peak synth; weights materialized once into owned f32 storage (~98
  MiB) — documented deviation from the mmap mandate: candle 0.11 has no
  zero-copy path; measured WIN vs kokoro 457 MB, revisit if a candle mmap
  path lands. RTF 0.31 vs kokoro 0.785 is the known M2 speed gap → session 4
  (decode batching + thread dial), not papered over here.

## Root causes / laws

- `TimestepEmbedder` Rms norm divides by **√(mean+eps)** — compose as
  `sqrt()?.recip()?` (candle has no `rsqrt`); `.recip()` alone corrupts time
  embeddings by ~1e5.
- `conv_transpose1d` dilation must be `1` (candle `l_out` collapses at 0).
- `complete_kv` returns the whole filled prefix; 0-frame `Linear` must be
  guarded; `Tensor::full` infers dtype from the value.
- Chunk count ≠ chunk plan: the oracle's channel grain is decode-batch
  drain timing; compare plans and total audio duration, not stream item
  counts.
- Parity = same replay noise + same chunk + same voice + step-by-step
  latent/eos diffs; stage bisection (conds/voice/text/transformer) over
  step-tracing.

## Done-conditions

- [x] `tts synth --lane flow-candle ...` produces audio for a corpus sentence, chunks.jsonl complete
- [x] Self-review checklist vs config.json: every hyperparameter sourced from the checkpoint config, none hardwired by accident (25/25 verified, session 3)
- [x] Cold load wall time + peak RSS logged (mmap mandate: weights never fully resident unless evidenced as win)
