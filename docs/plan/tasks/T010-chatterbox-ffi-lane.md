# T010 — chatterbox.cpp FFI lane (ggml-ffi)

- **Status:** `in-progress` (layer-A preflight DONE 2026-11-05; layer B = FFI crate + Rust lane next)
- **Milestone:** [M3](../index.md#milestone-m3--ggml-ffi-lane-cloning)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md)

## Goal

Voice cloning capability via the mature ggml port: `gianni-cor/chatterbox.cpp` (Turbo EN variant), behind our seam with FFI and the te-style off-disk GGUF mechanics.

## Approach

- Preflight (gate before code): verify the library exposes a stable C API (targets.md verify-list) — if C API is absent/unstable, subprocess wrapper is the fallback (te relay pattern) and this task records the swap.
- **RE-TARGETED at preflight (2026-11-05): the vehicle is `llama.cpp` `tools/tts` (Chatterbox Turbo), not `gianni-cor/chatterbox.cpp`.** gianni-cor exposes no stable C API (single `extern "C"` = `tts_cpp_cli_main`; Engine API is C++-only, unversioned, own GGUF format, torch converters required); llama.cpp upstream has chatterbox natively (PR #26254), frozen `llama.h`/`mtmd.h` C libs, and ggml-org prebuilt GGUFs. Full evidence table → targets.md verify-list + `docs/notes/chatterbox.md`.
- FFI wrapper crate `lanes/ggml-ffi/` (name kept): a workspace-side C++ shim (`chatterbox_shim.cpp`, `extern "C"`) wrapping `mtmd_helper::gen_audio` exactly as tools/tts/tts.cpp does (faithful-port law), linked against static libllama + libmtmd + ggml; llama.cpp pinned via git pin, T008-style.
- GGUF: ggml-org prebuilt `chatterbox-turbo-Q8_0.gguf` + `mmproj-chatterbox-turbo-F16.gguf` (their validated shipped pair; the task-file's q4_0/f16 recipe belonged to the gianni-cor converters → superseded; q4_0 possible later via llama.cpp's own requantizer if needed).
- Cloning path: `--tts-speaker-file`-equivalent (`mtmd_helper_bitmap_init_from_file`) inside the shim; consent/provenance note recorded per bundled voice (targets.md license asymmetries rule).
- Telemetry: FFI boundary timings (prompt/semantic vs frame-gen/vocoder) mapped into chunks.jsonl stages; chunked via T013; model + contexts live in one `load()` per lane instance (worker-loop=1; llama.cpp contexts are single-threaded per instance — mirroring te session-per-thread).
- Round-trip WER gate like every lane; their CER 0.00 CI gates are the port's own discipline — we add hard-sentences corpus to it.

## Done-conditions

- [ ] `tts synth --lane ggml-ffi --voice <cloned> ...` yields recognized speech from a reference-WAV clone (round-trip WER reported)
- [x] C-API (or subprocess) decision recorded with evidence in targets.md verify-list — **decision 2026-11-05: llama.cpp FFI via own C++ shim (no subprocess needed); gianni-cor swap recorded**
- [x] Cold-load wall + peak RSS logged (llama-tts baseline: cold load ≈ 0.6 s, peak RSS 2.4 GB); determinism verified at CLI level (3× byte-identical per dial for both default and cloned voice)

## Layer-A log (2026-11-05)

Built llama.cpp master (ggml 186eab1, cmake 4.4.4, release, `-DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF`, needs sparse-checkout dirs: cmake ggml vendor tools — `app/` and `tools/results` break otherwise) → `llama-tts` green.

GGUFs (ggml-org/chatterbox-turbo-GGUF, converted 2026-11-05, auto-converter): `chatterbox-turbo-Q8_0.gguf` 481,791,712 B + `mmproj-chatterbox-turbo-F16.gguf` 555,371,392 B → `/teranux/hugging_face/t012-drop/chatterbox/`. BF16 primary also available; f16 t3/s3gen q8 pair from cstr repo uses the gianni-cor GGUF key format (incompatible).

Measurements on 4800H (seed 42, release, no warmup):

| text | threads | audio | wall (prompt+gen+vocoder) | audio/wall | peak RSS |
|---|---|---|---|---|---|
| 1-sentence (4.08 s out) | 4 | 4.08 s | 9.2–9.6 s (1.5–1.8 + 2.3 + 5.2–5.5) | 0.43–0.46 | 2,462,420 KB |
| 1-sentence | 8 | 3.84 s | 8.12 s | 0.47 | — |
| 1-sentence | 16 | — | — | — | SLOWER (20.2 s for 10.6 s audio) |
| multi-sentence (10.08 s out) | 4 | 10.08 s | 18.04 s (1.80 + 5.70 + 10.54) | 0.56 | — |

Determinism: within fixed (binary, seed, threads) — 3×/2× byte-identical incl. cloned voice; across thread counts (t4 vs t8, t4 vs t16) bytes differ AND sample length differs (sampling divergence, same reduction-order class as our flow-lane law) → gate law: identity per (lane, quant, dial) triple, threads is a dial. Seed 43 differs from seed 42 → seeded not constant.

Voice cloning: `--tts-speaker-file` with an 18.99 s 22050 Hz mono s16 wav (piper lane output) works first try; 3.36 s audio / 8.02 s wall; 2× byte-identical; round-trip nemo-int8 transcribe = verbatim text. Default-voice round-trip also verbatim.

Shim notes (from tools/tts/tts.cpp): `common_init_from_params` → llama_init; `mtmd_init_from_file(mmproj, model)`; check `mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_NONE`; speaker ref via `mtmd_helper_bitmap_init_from_file`; optional pre-shipped voice via `mtmd_gen_audio_get_voice` (`n_voices`, `mtmd_gen_audio_voice_name`); `mtmd_helper::gen_audio` flow: `init(lctx, mctx)` → `set_input({seq_id, prompt, speaker_ref, lang, top_k, top_p, seed})` → `step_prompt(n_batch)` loop → per frame: `sample_semantic_code()` = `common_sampler_sample/accept`, then `gen.step_gen(sampled, h_state, &h_next, &stop)` → `get_output(&sample_rate, &data, &data_len, &n_samples)` (24 kHz, WAV bytes). Timings available: prompt/generation/vocoder splits = chunks.jsonl stages (acoustic=semantic+frame-gen, vocoder=hift).
