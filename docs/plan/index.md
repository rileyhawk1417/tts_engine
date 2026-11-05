# tts_engine — build plan

> Root index for the build plan. Read this first every session.
> Status board is here; **tracking checkpoints live in the project memory store**
> (`.pi/memory/projects/tts-engine/`). Task files carry design, never status churn.

## Scope

Model-agnostic, CPU-first TTS engine in Rust. One seam — `tts synth {text, lang, voice, lane} → WAV/PCM + chunks.jsonl` — over many lanes. Weights load off disk (mmap-first). Every run emits telemetry (per-stage ms, TTFA, RTF, peak RSS). Every lane is judged by round-trip WER via `transcription_engine`. Honest-numbers discipline inherited from `te`: measure first, adopt with proof, falsify and shelve otherwise.

**Build targets:** see [targets.md](targets.md). Six mapped engines, four active shipped lanes (graph-ort, flow-latent ×2, piper, ggml-ffi) + one measurement-only lane (shelved), two parked engines; ggml-ffi adopted as the cloning lane only (M3 close, 2026-11-06).

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

## Milestone M3 — ggml-ffi lane (cloning)  **(CLOSED 2026-11-06: T010 adopted as the cloning lane** — llama.cpp `mtmd_helper::gen_audio` behind an own C++ shim; hard-80 80/80 identity trusted; WER med 60.0 = not a quality contender; ≈2.4 GB RSS = CLI-scale)**

| ID | Task | Status | Depends on |
|---|---|---|---|
| T010 | chatterbox.cpp FFI lane | **done** (M3 closed 2026-11-06 — llama.cpp FFI shim + lane, hard-80 80/80 identity trusted, cloned-voice done-condition met; verdict: ADOPT as cloning lane, not a quality contender; see notes/scoreboard.md §2026-11-06) | T002, T003 |

## Milestone M4 — streaming + measurement lanes  **(CLOSED 2026-11-05: T011 phase A adopted, T012 shelved as lane)**

| ID | Task | Status | Depends on |
|---|---|---|---|
| T011 | Piper streaming lane | done-patchA 2026-11-05 (lane + gates; hard-500 500/500 identity on amy + ryan; verdict ADOPT — notes/piper.md) | T002, T003, T013 |
| T012 | NeuTTS honest measurement experiment | done 2026-11-05 (backbone clears 50 Hz ≥2×; shelve as lane; see task file) | T003 |

## Milestone M5 — resident daemon

| ID | Task | Status | Depends on |
|---|---|---|---|
| T014 | `tts serve` resident daemon | not-started (design note shipped; unblocked by T011 phase A — no session-4-speed dependency) | T002, T003, T013 |

## Critical path

T001 → T002 → T003 → T004 → T005 → T007. T006 branches after T003 (can run while T004 is built). T008 gates on T006; T009 closes M2. T010/M3 and T014/M5 are independent tails off the foundation (T002, T003, T013);
both can run while anything else is in flight.

## Open questions

- **en_US default lane**: piper ryan (WER med 20.00 hard-500, wall/audio ≈ 0.268, same judge as flow rows) measured ≥ kokoro on every axis, but the kokoro comparison is cross-judge-sha (`cdb5b135` vs `38b3d623`) — re-run kokoro on the current judge before any default-LANE switch, or accept the approximate comparison and switch. Unresolved by design until there's a reason to adjudicate. **Chatterbox (T010) does not enter this question: 60.0 WER med hard-80, last place on the same-judge board; adopted as the cloning lane only.**
- ~~G2P for the Kokoro lane: espeak-ng (GPL-linking caveat) vs misaki~~ — **resolved 2026-11-05 (T004)**: espeak-ng subprocess (the misaki espeak-fallback materialization), phonemizer-style punctuation restore; caveat documented in the lane.
- ~~Streaming IPC shape for the seam~~ — **resolved 2026-11-05** in [notes/resident-service.md](../notes/resident-service.md): HTTP/1.1 over UDS + NDJSON chunk frames, worker-loop=1, one binary two modes. Landed as board task **T014 (M5)**; piper phase A removed the session-4-speed dependency.
- ~~Which `te` lane is the default round-trip judge~~ — **resolved 2026-11-05 (T005)**: nemo int8 default (literal-text transcriptions win on the hard set), parakeet q8_0 tiebreak; judge sha recorded per run.

## Sources

- Research memory: `projects/tts-engine/tts-engine-research-ground-truth-model-zoo-lanes-and-te-l.md`, `…/tts-zoo-expansion-verdict-pockettts-in-chatterbox-ffis-ne.md`, `…/phonon-gradium-is-the-flow-latent-lane-anchor-ptts-crate.md` (project memory store).
- Transferable `te` laws: global memory `stt/` collection (quantization, parallelism, gotchas, bench discipline).
- Phonon primary sources: gradium.ai/blog/release-phonon-open-weight, HF docs `gradium_phonon_tts`, repo `gradium-ai/xn-ptts`.
