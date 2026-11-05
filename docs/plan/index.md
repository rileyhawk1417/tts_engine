# tts_engine — build plan

> Root index for the build plan. Read this first every session.
> Status board is here; **tracking checkpoints live in the project memory store**
> (`.pi/memory/projects/tts-engine/`). Task files carry design, never status churn.

## Scope

Model-agnostic, CPU-first TTS engine in Rust. One seam — `tts synth {text, lang, voice, lane} → WAV/PCM + chunks.jsonl` — over many lanes. Weights load off disk (mmap-first). Every run emits telemetry (per-stage ms, TTFA, RTF, peak RSS). Every lane is judged by round-trip WER via `transcription_engine`. Honest-numbers discipline inherited from `te`: measure first, adopt with proof, falsify and shelve otherwise.

**Build targets:** see [targets.md](targets.md). Six mapped engines, four active lanes (graph-ort, flow-latent, ggml-ffi, streaming-graph), one measurement-only lane, two parked engines.

## How to work

1. Read this index.
2. Pick the highest-priority not-started task whose dependencies are all complete; prefer the one unblocking the most downstream work.
3. Read that task's file, do the work, verify its done-conditions.
4. Update the status here, then save a session checkpoint to project memory (numbers, decisions, next step). One memory file per session; work-list stays in this index.

Status values: `not-started` · `in-progress` · `completed` · `blocked` · `shelved` · `superseded`.

## Milestone M0 — foundation

| ID | Task | Status | Depends on |
|---|---|---|---|
| T001 | Seam + telemetry spec | completed | — |
| T002 | Rust workspace scaffold + build gates | completed | T001 |
| T003 | Round-trip WER eval harness | completed | T002 |
| T013 | Long-text chunking module | completed (landed with T002) | T002 |

## Milestone M1 — default lane (graph-ort / Kokoro)

| ID | Task | Status | Depends on |
|---|---|---|---|
| T004 | Kokoro-ORT lane end-to-end | completed (2026-11-05) | T002, T003, T013 |
| T005 | Kokoro baseline scoreboard on Zen 2 | completed (2026-11-05) | T004 |
| T006 | Phonon ptts-lane probe | done 2026-11-05 | T005 |

## Milestone M2 — flow-latent lane (Phonon, then PocketTTS checkpoint)

| ID | Task | Status | Depends on |
|---|---|---|---|
| T006 | Phonon `ptts`-crate lane behind the seam | done 2026-11-05 | T002, T003 |
| T007 | Phonon evaluation + lane-vs-lane scoreboard | done 2026-11-05 (hard-80 in background; verdict in notes/phonon-eval.md) | T005, T006 |
| T008 | Candle from-scratch Phonon-31m port | done 2026-11-05 (`86c63df`; 48-step parity vs oracle) | T006 |
| T009 | Parity gate + q-law experiments | done 2026-11-05 (parity battery + hard-500 1t; notes in qlaws.md/scoreboard.md) | T007, T008 |

## Milestone M3 — ggml-ffi lane (cloning)

| ID | Task | Status | Depends on |
|---|---|---|---|
| T010 | chatterbox.cpp FFI lane | not-started | T002, T003 |

## Milestone M4 — streaming + measurement lanes

| ID | Task | Status | Depends on |
|---|---|---|---|
| T011 | Piper streaming lane | not-started | T002, T003, T013 |
| T012 | NeuTTS honest measurement experiment | not-started | T003 |

## Critical path

T001 → T002 → T003 → T004 → T005 → T007. T006 branches after T003 (can run while T004 is built). T008 gates on T006; T009 closes M2. Everything else (T010–T013) hangs off M0.

## Open questions

- G2P for the Kokoro lane: espeak-ng (GPL-linking caveat) vs misaki — spike or commit? (verify-list in targets.md)
- Streaming IPC shape for the seam: chunk callbacks in-process now (library users), named-pipe/WS later?
- Which `te` lane is the default round-trip judge: nemo int8 (fastest) vs parakeet q8_0 (tiebreak)?

## Sources

- Research memory: `projects/tts-engine/tts-engine-research-ground-truth-model-zoo-lanes-and-te-l.md`, `…/tts-zoo-expansion-verdict-pockettts-in-chatterbox-ffis-ne.md`, `…/phonon-gradium-is-the-flow-latent-lane-anchor-ptts-crate.md` (project memory store).
- Transferable `te` laws: global memory `stt/` collection (quantization, parallelism, gotchas, bench discipline).
- Phonon primary sources: gradium.ai/blog/release-phonon-open-weight, HF docs `gradium_phonon_tts`, repo `gradium-ai/xn-ptts`.
