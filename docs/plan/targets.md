# Build targets — the models tts_engine builds against

> Decision record, 2026-11-05. Static content (what we build against and why);
> live status lives in `index.md` + the project memory store. Research provenance:
> project memories `tts-zoo-expansion-verdict…` and `phonon-gradium-is-the-flow-latent-lane-anchor…`.

## Lanes → checkpoints

| Lane | Infra shape | Primary checkpoint | License | Files we consume | Role |
|---|---|---|---|---|---|
| **graph-ort** (M1) | ONNX via `ort`, session-per-thread | `hexgrad/Kokoro-82M` (StyleTTS2 + iSTFTNet, 24 kHz) | Apache-2.0 | int8 ONNX (~169 MB external-data mmap), espeak-ng/misaki G2P | Quality default; cheapest proven end-to-end |
| **flow-latent** (M2) | candle, `MmapedSafetensors`; `ptts` crate as oracle lane | `gradium/phonon-31m-en` — 31M causal LM (6L×512×8h), SentencePiece 4001 raw-text, one 32-dim latent/frame @ 12.5 Hz, one-step flow head (`lsd`), EOS head, small Mimi decoder → 24 kHz | Apache-2.0 | `model.q8.gguf` 64.5 MB (ptts lane), bf16 `safetensors` (candle port), `tokenizer.json`, `default-voice.safetensors`, `gradium/phonon-voices` (~100 voices) | Latency anchor (TTFA headline), from-scratch tuning target |
| **flow-latent** (M2, same lane) | shared flow-head + Mimi decoder module | `kyutai-labs/pocket-tts` — 100M + minis 18.2M/6.9M/1.9M, `languages/<lang>/model.safetensors` | CC-BY-4.0 (gated, prohibited-use clause) | per-language safetensors + per-voice embedding safetensors | Second checkpoint; multi-language + tiny-footprint variants |
| **ggml-ffi** (M3) | FFI to `chatterbox.cpp` (C API) | Chatterbox Turbo EN (GPT-2-medium T3 + meanflow CFM + S3Gen + HiFT) | MIT (Resemble) + port MIT | GGUF: Q4_0 T3 + f16 S3Gen (their quant recipe) | Voice-cloning lane; ggml quantization testbed |
| **streaming-graph** (M4, T011 phase A done 2026-11-05) | fused ONNX VITS (both voices) + `eval/patch_piper_rng.py` seeded-noise graph surgery; espeak-rs 0.2 in-process G2P | Piper en_US drop-dir voices (amy medium 1.0.0, ryan high 0.2.0-legacy) | MIT runtime; per-voice files | fused `*.onnx` + `*.onnx.json` (append-json law), `*-rng.onnx` patched graphs | **ADOPTED phase A** (ryan = en_US quality lane: WER med 20.00 hard-500, wall/audio ≈ 0.268; amy = latency lane: ≈ 0.063, warm TTFA ≈ 200 ms; 500/500 identity both). True mel streaming (phase B, split encoder/decoder) BLOCKED on training-checkpoint artifacts, not engineering |
| **measurement-only** (M4, T012 done) | llama.cpp-class qwen2 + ONNX codec head | `neuphonic/neutts-air` Q4/Q8 GGUF | Apache-2.0 | qwen2-arch GGUF (~748 MB Q4_0) + official `neucodec-onnx-decoder(-int8)` (gated:auto) | **T012 verdict: backbone 217 t/s (4.35× budget, official Q4_0) + codec decode int8 646–744 tok/s (12.9–14.9×) through graph-ort — chain would be backbone-bound ~4.3× realtime; public 15×-slower = Python stack; SHELVED as lane (3rd decode stack + gated supply + voice-encode torch-only)** |

## Parked (not targets)

- `microsoft/VibeVoice` (MIT) — 0.5B realtime exists but diffusion-head, Python/vLLM stack, no ggml/ONNX/Rust port. Revisit on any port.
- VoxCPM2 (OpenBMB, Apache-2.0) — 2B diffusion-AR, CPU-hostile. GPU-lane material only.

## Decode policies ported exactly, never simplified

- Phonon: temperature 0.3 (flow start-noise variance), `eos_threshold` EOS head, ~50 text-token chunks at sentence boundaries, >200-token chunks split at clause marks, single words cut between characters; voice = (num_frames, 512) speaker latents prepended per chunk.
- OuteTTS lesson stands (64-token repetition-penalty window) for any future LM lane.
- PocketTTS: `text_chunking.py` policy when its checkpoint is enabled.

## Evaluation targets (who judges us)

- Round-trip WER: `transcription_engine` lane judges our output (nemo int8 default; parakeet q8_0 secondary; temp 0, fixed corpus, te repo WER conventions).
- Corpus: internal bench sentences + `gradium/tts-hard-sentences-202610` (public hard-text set: numbers, dates, codes, units).
- Baseline to beat or justify against: Kokoro-ORT numbers from T005 on the Zen 2 box.

## Hardware reference

- Dev/target box: Zen 2 4800H 8C/16T (same class as te reference numbers: RTF 0.10–0.22, WER 14–18%).
- Weights cache env: `HF_HOME=/teranux/hugging_face`. Weights load off disk (mmap-first) — never bake into memory without the standard-load comparison.

## License asymmetries to respect

- Phonon-31m Apache-2.0 (commercial OK); ALL phonon-40m = CC BY-NC 4.0 + HF-gated → 40m never a default target.
- Speaker codec not released → no on-device cloning on Phonon; cloning belongs to chatterbox lane.
- Voices have per-voice terms (gradium/phonon-voices, kyutai/tts-voices) — record per bundled voice.

## Open verify-list

- [x] `ptts` crate dependency weight (does it drag ggml/embedded runtimes?) — **resolved 2026-11-05 (T006): no ggml, no embedded runtimes.** Dep tree: `ptts` → `xn` (+`xn-gemm`/`xn-gemm-f16`/`xn-gemm-c64` pure-Rust GEMM) → `tokenizers`, `safetensors`, `gguf` (parser only), `rayon`, `memmap2`. The `hub` crate feature (network download) is OFF in our Cargo.toml; T3's earlier `ggml` worry came from the pyo3/python bindings, which we don't compile.
- [x] xn-ptts root LICENSE files (README states MIT OR Apache-2.0) — **resolved 2026-11-05 (T006): dual LICENSE-MIT and LICENSE-APACHE present at repo root; `ptts` crate file headers + Cargo `license = "MIT OR Apache-2.0"` agree. Vendorable.**
- [ ] phonon-40m gate terms (only needed if 40m ever used, NC)
- [ ] pocket-tts code LICENSE before vendoring anything from kyutai-labs Rust
- [ ] Phonon/PocketTTS Mimi decoder safetensors layout equivalence
- [x] G2P choice for Kokoro lane: espeak-ng vs misaki — **resolved 2026-11-05 (T004): espeak-ng subprocess** (the misaki espeak-fallback materialization), phonemizer-style punctuation restore + vocab filter; GPL-caveat = subprocess invocation (no linking), `TTS_ESPEAK` override. piper lane (T011) uses espeak-rs 0.2 IN-PROCESS vendored espeak-ng 1.52.0.1 instead — GPLv3 LINKING caveat documented in that lane's contract block.
