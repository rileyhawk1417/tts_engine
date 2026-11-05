# T011 — Piper lane (streaming-graph)

- **Status:** `in-progress` (session-1: discovery + graph surgery + acceptance probe done; Rust lane not yet built)
- **Milestone:** [M4](../index.md#milestone-m4--streaming--measurement-lanes)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md), [T013](T013-longtext-chunking-module.md)

## Goal

The latency competition for the flow-latent lane: Piper/ONNX VITS as an en_US graph lane, TTFA-first, carrying the streaming/IPC shape that later feeds `tts serve` (see `docs/notes/resident-service.md`).

## Session-1 findings (2026-11-05, probed on the real artifacts)

Artifacts (T012 drop dir, `/teranux/hugging_face/t012-drop/piper/`):

| voice | piper version | symbols | size | graph out |
|---|---|---|---|---|
| `en_US-amy-medium` | 1.0.0 | 256 (map 154) | 63 MB | `(1,1,1,N)` f32, 22050 Hz |
| `en_US-ryan-high` | 0.2.0 (legacy) | 130 | 121 MB | `(1,1,1,N)` f32, 22050 Hz |

1. **Both are FUSED VITS graphs** — in: `input` (phoneme ids i64 `[b,n]`), `input_lengths` i64 `[b]`, `scales` f32 `[3]` (noise_scale, length_scale, noise_w); out: audio f32 directly. **No mel intermediate** → decoder-level mel streaming is NOT possible with these artifacts.
2. **True mel-level streaming** (PR #255, merged) requires **training-checkpoint re-export** (`piper_train.export_onnx_streaming` → encoder.onnx + decoder.onnx, flow moved to decoder, encoder RTF 0.01). Grounded constants from the streaming scripts (fetched `infer_onnx_streaming.py`): **chunk 45 mel frames** (hop 256 → 0.523 s @ 22050), **chunk_padding CLI default 5 frames/side** (class docstring says 10 — discrepancy preserved for porting care); mechanics are **pad-and-TRIM** (`audio[start_pad:-end_pad]`), NOT overlap-add/windowing; short inputs (≤ chunk+2·pad frames) skip streaming entirely. Noise in the streaming lineage is **frozen at export** (`torch.manual_seed(1234)`; `randn` traced into encoder as constants; decoder has no noise input) — deterministic but voice character locked; our noise-input surgery is strictly friendlier. Phase B also gated on **checkpoint availability** for a voice we care about — dropped unless phase A leaves headroom. Prior art for quality-on-boundaries: mush42/sonata (alive, 2024) and marty1885/paroli (similarity-search concat).
3. **Chunk-level streaming is what the daemon actually needs**: `tts serve` streams text chunks (T013) as `SynthStream` frames — exactly what relay's kokoroWorker does today. Phase A = fused graphs + chunk-level synth through the seam. Phase B (mel streaming) only improves single-long-sentence latency.
4. **Nondeterminism root-caused and fixed.** Same ids → different audio every run: the graphs embed **2× `RandomNormalLike` with no seed attr** (duration noise eps1, spectral noise eps2). Surgery (`eval/patch_piper_rng.py`): each `RandomNormalLike(X) -> O` becomes `O = Slice(noise_k, starts=0, ends=Shape(X), axes=[0,1,2], steps=1)` — caller-supplied seeded noise, scales wiring untouched. Ground-truth shapes (tapped from original graphs):
   - **eps1 = [1, 2, T_text]** (duration noise; lane-exact from phoneme count)
   - **eps2 = [1, 192, T_mel]** (192 bins on BOTH voices; T_mel is runtime-dynamic — durations depend on eps1 — hence the noise_1 budget=2048 + dynamic Slice)
5. **Acceptance (patched `-rng` graphs)**: 3× byte-identical runs on both voices; different noise → different audio (seed dial works); rtf unchanged (0.075 vs 0.078 pre-patch). Patched graphs live next to originals (`*-rng.onnx`).

Probe timings (python ORT 1.29, fused graph, mid-length chat line):

| voice | phonemes | wall med | audio s | rtf |
|---|---|---|---|---|
| amy medium | 100 | 233 ms | 3.11 | **0.075** |
| ryan high | 100 | 707 ms | 1.80 | 0.393 |
| amy medium | 30 | 93 ms | 1.15 | 0.081 |

 amy ≈ 10× faster than kokoro (0.80) and 4× faster than flow-candle (0.334). ryan still 2× faster than kokoro.

## Phase A approach (this lane, session-2)

- Lane name `piper`, gated `#[cfg(feature = "piper")]`, layout mirrors `graph-ort` (kokoro): ort 2.0.0-rc.15 api-21 session, `load_ms` = wall of `load()` only.
- **Phonemization**: `espeak-rs` 0.2 (vetted 2026-11-05: crate lives in `thewh1teagle/piper-rs`, NOT archived — standalone `thewh1teagle/espeak-rs` repo is the one that 404s; deps = `espeak-rs-sys` only (bindgen+cmake, no ort coupling); MIT wrapper). Vendored espeak-ng means a cmake C build in-tree + an `espeak-ng-data` runtime dir to provision; **GPLv3-linking caveat documented like kokoro's espeak question**. Fallback/experiment: `espeak-ng-rs` (eugenehp) pure-Rust port, fresh 2026-09 — fidelity + license unverified. **G2P perf measured 2026-11-05 (system espeak-ng 1.52.0, ctypes, warm)**: in-process G2P **0.070 ms med / 0.135 ms max** per sentence (54 kchar/s), init 15.4 ms one-time; CLI one-shot 10.0 ms/utterance (process+data load included) = worst-case architecture. G2P ≈ **0.03%** of amy's 233 ms VITS wall → **phonemizer is a build+licensing surface, NOT a perf surface**. Note: system lib exports only `espeak_TextToPhonemes` (3-arg) — `TextToPhonemesWithTerminator` NOT exported in shared build; verify in vendored build; T013 chunking covers clause splits at the text layer regardless. Map chars → ids via `phoneme_id_map` from the `.onnx.json`; port piper's id assembly (bos/pad/eos + separators) **verbatim from piper source** — two different assembly conventions exist for format 1.0.0 vs 0.2.0; keep per-format handling.
- **Noise rng**: lane owns a seeded standard-normal rng (rand_distr / candle-equivalent); honors seam `--seed` structurally (eps1 exact, eps2 budgeted). Determinism class: `deterministic (given seed)`.
- **Scales** from the voice json `inference` block (0.667/1.0/0.8 default).
- **Chunk streaming**: T013 chunking → per-chunk phonemize + single ORT run → chunk released on completion; TTFA = first chunk's wall (expected ~100–250 ms chat-line).
- **Byte-identity gate**: per (voice, scales, seed) triple through the eval harness, like every other lane.
- **Header**: `lane: piper`, `quant: f32`, `chunk_policy: "text-chunks(T013)"`, speed field per the resident-service draft.

## Phase B (optional, NOT this session)

- Requires a piper **training checkpoint** for re-export via `export_onnx_streaming`; decoder chunk constants + overlap/add windowing to be ported from `piper_train/export_onnx_streaming.py` + `infer_onnx_streaming.py` if ever pursued. Drop-dir voices are checkpoint-less → **blocked on artifact availability, not on engineering**.

## Done-conditions (phase A)

- [ ] `tts synth --lane piper --model <dir>` e2e: phonemize→run→wav with correct 22050 Hz rate
- [ ] Byte-identity gate PASS (lane, f32, chat-length + hard-80 rows)
- [ ] TTFA + RTF + WER scoreboard row vs kokoro (t005a ref) and flow-latent rows
- [ ] Adopt/shelve verdict (expected: adopt as the cheapest-CPU en_US lane given rtf 0.075)
