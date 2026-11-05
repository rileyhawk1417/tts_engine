# T006 — Phonon `ptts`-crate lane behind the seam (flow-latent step a)

- **Status:** `done` (2026-11-05; gate + internal-corpus eval row trusted)
- **Milestone:** [M2](../index.md#milestone-m2--flow-latent-lane-phonon-then-pockettts-checkpoint)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md)

## Goal

Phonon-31m generating through our seam via Gradium's first-party `ptts` crate (MIT OR Apache-2.0): fast path to streaming + telemetry, and the correctness oracle for the from-scratch port (T008).

## Approach

- `lanes/flow-ptts/`: wrap `ptts::Checkpoint::resolve` (local path under `weights/`, hf only when explicitly allowed), `builder(Lang::X).device(Cpu)`; `tts.stream(text)` → map PCM chunks to seam chunks; `say_with(SpeechOptions)` for non-stream bench runs.
- Sessions: `session_default(&options)` / `session(&options, max_seq_len)` mapping = our warm-pool; telemetry captures session reuse (voice-primed KV) vs cold as TTFA delta rows.
- chunk callbacks: chunk boundary events → chunks.jsonl per-chunk rows (voice restart per chunk per Phonon policy).
- First clear the verify-list blockers (targets.md): cargo dep weight of `ptts` (ggml pulled?), actual LICENSE files, phonon-40m gate irrelevance (31m only).
- Weights off disk: q8 GGUF (`model.q8.gguf`, 64.5 MB) default; f32 variant noted for later comparison.

## Done-conditions

- [x] `tts synth --lane flow-ptts ...` streams intelligible audio, chunks.jsonl complete, per-chunk TTFA visible
  - probe WER 0% on 19 words through the nemo judge; internal corpus WER mean 4.71% (`eval/results/20261105-1043-flow-ptts-internal-t006`, verdict trusted)
  - per-chunk TTFA/rtf/RSS in chunks.jsonl; chunk stream ~35–88 chunks per run (ptts emits stream chunks, not sentence chunks)
- [x] Determinism gate passes (fixed seed) — two runs byte-identical
  - BUT: only at `TTS_XN_THREADS=1`. xn CPU FP kernels are order-nondeterministic under rayon parallelism (threads 4/8 diverge mid-stream; chunk counts can differ). Lane reports dial-aware class: `seeded` at 1 thread, `nondeterministic` + reason above. `build.sh`/`eval/run.sh` pin threads=1 for the lane.
- [x] Verify-list items resolved and recorded in targets.md (no dangling license unknowns for a shipped lane)
  - ptts dep tree: no ggml, no embedded runtimes (xn pure-Rust GEMM, tokenizers, safetensors/gguf parsers, rayon; hub feature disabled)
  - xn-ptts root LICENSE-MIT + LICENSE-APACHE present; crate headers agree

## Results and follow-ups (see docs/notes/phonon.md for the full record)

- Gate dial (threads=1, release): rtf 0.72–2.0 (long paragraphs amortize), TTFA ~3.1–3.5 s, RSS 190 MB max (kokoro: 457 MB). threads=8 measured rtf 3.44 / TTFA 0.9 s but is out of gate scope (nondet).
- T007 must add: Session primer TTFA-delta rows (voice conditioning paid per request vs once), internal+hard-80 matrix with both judges.
- T009 should chase: deterministic-parallel xn path (or an intentional gate-law relaxation) to publish the parallel speedup; f32-vs-q8 quality row.
