# T010 — chatterbox.cpp FFI lane (ggml-ffi)

- **Status:** `done` (2026-11-06: layer A + layer B complete; hard-80 **80/80 identity pass, verdict trusted**, WER med 60.00 / mean 67.54 — see `docs/notes/scoreboard.md` §2026-11-06; verdict **ADOPT as the board's voice-cloning lane, not a default-TTS quality contender**) — [M3 CLOSED](../index.md)
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

- [x] `tts synth --lane ggml-ffi --voice <cloned> ...` yields recognized speech from a reference-WAV clone (round-trip WER reported) — **met 2026-11-06**: `--voice /tmp/t011-long.wav` (18.99 s piper reference) → nemo-int8 transcript verbatim ("Hello from the Chatterbox native clone running inside the seam."), 2× byte-identical across invocations
- [x] C-API (or subprocess) decision recorded with evidence in targets.md verify-list — **decision 2026-11-05: llama.cpp FFI via own C++ shim (no subprocess needed); gianni-cor swap recorded**
- [x] Cold-load wall + peak RSS logged (llama-tts baseline: cold load ≈ 0.6 s, peak RSS 2.4 GB); determinism verified at CLI level (3× byte-identical per dial for both default and cloned voice)

### Hard-80 battery (2026-11-06, `eval/results/20261106-0033-ggml-ffi-hard-t010-cbx-t4/`)

`--corpus hard --limit 80`, t4, seed 42, judge nemo-int8 `te @38b3d623`:
**80/80 identity pass → trusted.** WER med 60.00 / mean 67.54 (raw med
80.0); TTFA med 8344 / p95 9725 ms; seam rtf (audio/wall) med 0.406; peak
RSS max 2,494,480 KB. Last place on the same-judge hard-80 board (ryan
22.50 ≪ amy 40.00 < candle 42.86 < ptts 46.43 ≪ 60.0) — expressive T3
sampling over whisper-derived semantic tokens; adoption case is cloning,
not quality. **Hard-500 not run**: llama-class fixed per-dial reduction
order + per-chunk KV resets keep grain-race risk low and the adoption case
(cloning) does not hinge on hard-500; if a cloning voice becomes a
production path, gate it at hard-500 then.

## Verdict (M3)

**ADOPT as the board's voice-cloning lane — the only lane that clones**;
not a default-TTS quality contender. Speed mid-pack
(audio/wall 0.406). RSS ≈ 2.4 GB = CLI-scale, not T014-resident-scale
(deviation recorded in the T014 contract). en-US default-lane question
unaffected: piper ryan remains the answer.

## Layer-B log (2026-11-06, session-4)

**Vehicle, final:** in-process FFI — own `extern "C"` shim (`crates/tts-engine/src/shim/chatterbox_shim.cpp`) built by `build.rs` (`cc`, C++17, `-DGGML_USE_CPU`, 7 include dirs) and linked against the durable pinned llama.cpp static archives at `/teranux/third_party/llama.cpp` (HEAD `186eab1c…`; `libmtmd`, `libllama-common`, `libllama-common-base` [build-info symbols live here — found by link failure], `libllama`, `libggml*`, `libvendor-hash`; system `gomp,numa,m,dl,pthread`). The shim binds `mtmd_helper_gen_audio_*` (already `extern "C"`) directly plus `common_init_from_params` (C++-mangled) — no re-wrap of the C++ orchestration.

**Shim surface** (`tts_cbx_*`): `init(threads,n_batch,n_predict_cap)` / `ok` / `set_speaker_wav` / `reset_speaker` / `sample_rate` / `n_voices` / `voice_name` / `synthesize(text,seed,top_k,top_p,min_p,temp,penalty_repeat → wav_buffer,samples,frames,prompt_s,gen_s,wav_s)` / `last_error` / `free`. Pins mirror llama-tts: `embedding=true`, top_k 40 / top_p 0.95 / min_p 0.05 / temp 0.80 / penalty_repeat 1.0, n_batch 2048, frames cap 512, `mtmd_params.device = nullptr` (ggml_backend_dev_t, NOT ""). Clocks prompt/gen/vocoder via `ggml_time_us()`.

**API drift fixed** (llama.cpp `186eab1`): `common_params` has NO `n_threads` — use `params.cpuparams.n_threads` + `cpuparams_batch.n_threads`; `mtmd_context_params.device` = `nullptr`.

**Lane** (`crates/tts-engine/src/lanes/ggml_ffi.rs`): mirror of piper.rs structure; `GgmlFfiLane` + `TtsCbxSession` wrapper (`CbxHandle` Send newtype, single worker thread); model-only `ModelRef::Path` (Hub/Pseudo rejected); `resolve_mmproj` twin next to the backbone or explicit `mmproj` lane option; `resolve_dials` returns `CbxDials{top_k,top_p,min_p,temp,penalty_repeat,max_frames}` (unknown dial / out-of-range = hard error — seam law); per-call voice resolution: `VoiceRef::Path` → `tts_cbx_set_speaker_wav`, `Default`/`Id` → `tts_cbx_reset_speaker` (sticky-voice law, verified); `restart_voice: false`; determinism `Seeded` with live seed; TTS_CBX_THREADS dial default 4. **6 unit tests** (dial law, overrides, hard errors, mmproj probe). Feature `ggml-ffi`; probe example `ggml_ffi_lane_probe` (3× shipped identity + optional clone 2× + post-clone shipped-restore check + RunMeta stability + stage/TTFA dump + rep0 wav for WER).

**Probe gates (GREEN)**: 3× byte-identical shipped at seed 42 (pcm_sha `507c9ea1…`, cross-process stable vs CLI run), cloned 2× (`bef82bf9…`), shipped voice restored after cloning (no sticky speaker), RunMeta stable across reps; stages acoustic 3.8–4.1 s + vocoder 5.2–5.3 s, TTFA ≈ 9.1–9.4 s (cold-ish), audio 4.24 s, rtf(audio/wall) ≈ 0.45–0.47 — matches llama-tts CLI baseline. Round-trip WER (probe rep0, te nemo-int8): **verbatim**.

**Seam CLI e2e (GREEN)**: `tts synth --lane ggml-ffi …` — telemetry rows complete (header `determinism:seeded`, chunk stages g2p 0 / acoustic / vocoder / io 0, peak_rss ≈ 2.43–2.45 ×10⁶ KB — the **RSS deviation for T014**: 2.4 GB ≫ 300 MB/lane warm budget); two same-seed CLI runs byte-identical; **thread dial diverges through the seam (t8 ≠ t4) as expected**; T013 long text → 4 chunks, 11.4 s audio, wall 30.4 s, TTFA 7.6 s, rtf 0.375, cross-run byte-identical; long-run round-trip transcript legible (pangram ASR noise only).

**build.sh/run.sh feature lists extended to FIVE features** (`graph-ort,flow-ptts,flow-candle,piper,ggml-ffi`); full workspace gate OK (fmt + clippy -D warnings + 45 tests + [smoke, kokoro, flow-ptts] e2e identities). Release binary 32.6 MB (llama static ≈ +14 MB over the four-lane 18.4 MB).

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
