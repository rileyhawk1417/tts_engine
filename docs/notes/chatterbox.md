# Chatterbox lane — T010 groundwork notes (ggml-ffi)

State: layer-A preflight DONE 2026-11-05. Task file: `docs/plan/tasks/T010-chatterbox-ffi-lane.md`.
Verdict evidence also recorded in `docs/plan/targets.md` verify-list.

## Landscape (measured 2026-11-05)

Two ggml-class Chatterbox ports exist:

1. **`gianni-cor/chatterbox.cpp`** — standalone repo, 0–1 stars. Static `tts-cpp` lib +
   `tts-cli` binary; builds clean (cmake, vendored ggml pinned `58c38058` via
   `scripts/setup-ggml.sh`). API reality:
   - only `extern "C"` = `tts_cpp_cli_main(argc, argv)` (CLI dispatcher);
   - real capability = C++ `tts_cpp::chatterbox::Engine` (pimpl, warm-resident,
     cancel, streaming `StreamCallback`, seeded) — **not a C API**, explicitly
     unversioned ("a proper struct-based public API has not landed yet", v0.1.0);
   - ships its OWN GGUF key format. The cstr/chatterbox-turbo-GGUF HF mirror
     fails: `load_model_gguf: missing GGUF key: chatterbox.text_vocab_size`;
   - converter scripts require torch + `snapshot_download("ResembleAI/chatterbox-turbo")`
     (~1.5 GB) to produce ggufs in its format.
2. **`llama.cpp` `tools/tts`** (PR #26254, "multi-model TTS", 2026-11 era) —
   Chatterbox Turbo + Multilingual V3 supported natively over frozen
   `llama.h` + `mtmd.h` C libraries. ggml-org publishes prebuilt GGUFs.

**Decision: llama.cpp is the T010 vehicle.** Reasons: stable C libs, prebuilt
GGUFs, active upstream, relay-relevant (llama-server TTS endpoint exists for
turbo; MTL not yet server-capable). gianni-cor swap recorded as superseded;
its Engine shape (warm + streaming + cancel) remains the inspiration for our
shim surface.

## Artifacts

- Working llama.cpp build: `/tmp/llama.cpp` (shallow sparse checkout, commit
  `186eab1cb83899282bec44b837270959b56ee8f3`, 2026-11-05). Configure:
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_APP=OFF`.
  Sparse-checkout gotchas: needs `cmake`, `cmake/license.cmake`, `ggml`,
  `vendor`, `tools` (app/ + tools/results break configure otherwise).
- GGUF drop dir (outside HF cache layout, per t012 convention):
  `/teranux/hugging_face/t012-drop/chatterbox/`
  - `chatterbox-turbo-Q8_0.gguf` 481,791,712 B (backbone + voice + vocoder)
  - `mmproj-chatterbox-turbo-F16.gguf` 555,371,392 B (semantic audio tower +
    S3Gen/HiFT + builtin `default` voice + CAMPPlus/VE/S3Tok preprocessors)
  - cstr mirror pair (incompatible key format) also on disk — DO NOT use.
- Binary: `/tmp/llama.cpp/build/bin/llama-tts`.

## The recipe (from tools/tts/tts.cpp, 243 lines)

```
common_init_from_params(params) -> llama_init            # model + context + sampler
mtmd_init_from_file(mmproj_path, model, mtmd_params)     # mmproj context
mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE   # is TTS model
speaker: mtmd_helper_bitmap_init_from_file(mctx, wav)    # optional cloning
shipped voice: mtmd_gen_audio_get_voice / voice_name     # when n_voices > 0
mtmd_helper::gen_audio gen;                              # orchestrator
gen.init(lctx, mctx);
gen.set_input({seq_id, prompt, speaker_ref, lang, top_k, top_p, seed});
loop gen.step_prompt(n_batch)                            # prompt/semantic stage
per frame: sample_semantic_code() = common_sampler_sample/accept
           gen.step_gen(sampled, h_state, &h_next, &stop) # frame + vocoder stages
gen.get_output(&rate, &data, &data_len, &n_samples)       # 24 kHz WAV bytes
```

`gen_audio` is C++ (`mtmd_helper` namespace). Layer-B FFI = own C++ shim
file in the workspace (NOT an upstream patch) exposing `extern "C"`:
`init(model, mmproj, threads, opts) -> handle`, `synth(handle, prompt, seed,
out_sampler* callback-or-buffer, timings*)`, `unload(handle)`. Faithful-port
law: hyperparameters default exactly as tts.cpp (`top_k 0`-default? cli sets
temp 0.8, top_p 0.95, rep_penalty via common defaults: keep parity with the
README defaults: top-k 1000, top-p 0.95, temp 0.8, rep 1.2 was gianni's —
for llama.cpp use whatever tts.cpp/common defaults produce the reference
round-trips we verified), documented in the lane contract block.

## Measured baseline (4800H, Q8_0+F16 mmproj, seed 42, release)

| text | threads | audio | wall (prompt+gen+vocoder) | audio/wall | notes |
|---|---|---|---|---|---|
| 1-sentence | 4 | 4.08 s | 9.2–9.6 s | 0.43–0.46 | 42–46 frames/s peak |
| 1-sentence | 8 | 3.84 s | 8.12 s | 0.47 | shorter sample (different ODE path) |
| 1-sentence | 16 | — | — | — | worse (SMT contention) |
| multi-sentence | 4 | 10.08 s | 18.04 s | 0.56 | vocoder 10.5 s dominates |

- Cold load ≈ 0.6 s (weights 1.03 GB, mmap by gguf loader).
- Peak RSS 2,462,420 KB (2.4 GB). T014's ≤300 MB warm/lane budget will be
  exceeded by this lane — known deviation to record in the T014 contract.
- `llama_context: n_ctx_seq (8448) > n_ctx_train (8196)` warning benign.
- audio output: WAV (RIFF) 24 kHz mono bytes; frames → samples 50:1.

## Determinism law for this lane

- 3× byte-identical at fixed (binary, seed, threads) for default voice AND
  cloned voice (full WAV incl. header).
- Different thread counts → different bytes AND different sample length
  (reduction-order class; same law as flow lanes). t8 2× identical.
- Seed 43 ≠ seed 42 → seeded, not constant.
- Gate convention: identity per (lane, quant, dial) triple; dial = threads
  (4 = our lane default), plus any llama sampler knobs we expose.

## Voice cloning + provenance

- `--tts-speaker-file` with 18.99 s piper-lane wav (22050 Hz mono s16) worked
  first try; round-trip nemo WER ≈ 0. >5 s required; 10–15 s recommended.
- Pre-baked voice (five conditioning tensors) equivalent exists in llama.cpp
  only as `--tts-voice <name>` shipped voices from mmproj — not the gianni
  `--save-voice`/`--ref-dir` npy flow. Lane v0: VoiceRef::Path → temporary
  wav clone per run (cloning preproc ≈ 0.4 s, measured inside prompt stage);
  optimization (cache cond tensors) optional later.
- Provenance/licensing: Chatterbox MIT (Resemble AI + llama.cpp MIT). Bundled
  `default` voice = `conds.pt` shipped with the model (MIT, from chattersbox:
  "Boba's voice" upstream acknowledgment — record in targets.md when bundled).
  Cloned-voice wavs: user-supplied, consent-on-file per repo rule.

## Layer-B plan (next session)

1. Vendor/pin llama.cpp in workspace (git pin + build.rs or cmake build from
   workspace script; mirror T011's espeak-rs-sys style; static link).
2. `crates/tts-engine/src/lanes/ggml_ffi.rs` + `chatterbox_shim.cpp` +
   build.rs C++-compile glue (cc crate) — mirroring tts.cpp line-for-line.
3. Wire `make_lane` arm `"ggml-ffi"`, feature `ggml-ffi`.
4. Telemetry: prompt/gen/vocoder stage timings → ChunkMs slots (acoustic =
   semantic+frame-gen; vocoder = hift; g2p = 0); load_ms = model+mmproj init.
5. Model resolution: ModelRef::Path → main gguf; VoiceRef::Path → wav file;
   ModelRef options `--threads 4` default lane dial.
6. Gates: unit tests (shim ABI smoke), `piper_lane_probe`-style
   `ggml_ffi_lane_probe` example, byte-identity battery, hard-80 + hard-500
   via eval/run.sh, round-trip WER, scoreboard row.
7. Regression check: full-four-feature build + existing lanes' e2e (build.sh).
