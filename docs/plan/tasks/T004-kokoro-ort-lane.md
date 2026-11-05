# T004 — Kokoro-ORT lane end-to-end (graph-ort)

- **Status:** `completed` (2026-11-05)
- **Milestone:** [M1](../index.md#milestone-m1--default-lane-graph-ort--kokoro)
- **Depends on:** [T002](T002-workspace-scaffold.md), [T003](T003-roundtrip-eval-harness.md), [T013](T013-longtext-chunking-module.md)

## Goal

First real lane: Kokoro-82M int8 ONNX through `ort`, fully behind the seam with telemetry — the proven cheapest path to a judged lane. Reference checkpoint: `hexgrad/Kokoro-82M` (targets.md).

## Approach (as executed)

- ONNX Runtime via `ort` rc.15 (`load-dynamic`, `api-21`), env per process at
  `TTS_ORT_DYLIB` (default `/teranux/ort-1.22.1/libonnxruntime.so`), same shape as te.
- Model `models/kokoro-82m-v1.0-onnx/model_quantized.onnx` (int8, 92 MB; big file
  `model.onnx` 325 MB also on disk for later), voice `voices/af_heart.bin`
  510×256 f32 → `Vec<f32>`, `style row = min(len(ps),510)-1`.
- Lane name **`kokoro`** (not the feature name), feature gate `graph-ort`
  (`ort` optional in the crate; workspace dep NOT optional — see gotcha below).
- G2P: espeak-ng CLI subprocess (`-q --ipa=3 --tie=^ -v <voice> --stdin`), one
  batched call per chunk window with trailing-newline law (see laws), per-word
  fallback on phword-count mismatch, punctuation re-inserted from core trails,
  vocab filter to 114-entry id table, `[0,*ids,0]` bounds.
- Stage split on the fused graph: `g2p` = espeak+restore+filter+batch,
  `acoustic` = all ORT runs (iSTFTNet is in-graph), `vocoder` = batch concat +
  pause insertion, `io` = f32→i16.

## Contracts honored

- Fixed 4 stage slots `{g2p,acoustic,vocoder,io}` in `StageTimings`.
- `determinism = "deterministic"`; two cold runs PCM bit-identity **verified**
  (also enforced in `scripts/build.sh` kokoro e2e block).
- Telemetry seam-side: `RunRecorder` only sees `StageTimings`; chunk rows carry
  sha256 `pcm_bytes`, rtf_running may dip (law), constant `t_synth_start_ms`.

## Results (first run, 16-core laptop — NOT the T005 scoreboard)

- "Hello from the Kokoro lane. Route 47 leaves at 3.45 p.m., isn't it?" →
  2 chunks, 7.950 s audio, te-judge transcription near-verbatim.
- Gate run (1 chunk, 2.625 s audio): TTFA ~3.1 s, RTF ~0.7–0.8, peak RSS ~227 MB
  (ORT init + session), wall ~3.1 s.
- **These numbers are provisional** — honest scoreboard lands in T005 on the
  Zen 2 box with judge nemo int8 + parakeet q8_0 tiebreak.

## Laws discovered (must not regress)

1. **espeak-ng trailing-newline law**: stdin WITHOUT a trailing newline clips
   the final phoneme of the final word (`Hello\nworld` → `wˈl...` "wˈɜːl"
   loses the 'd'). Always write `join("\n") + "\n"`; keep per-word fallback
   count check as the tripwire.
2. **atoms waterfall**: split sentences → clause → whitespace with EVERY
   sub-piece restarting from the top (level-recursion that falls straight to
   char-slice silently mis-shards long mark-less text; balanced-packing test
   catches it: 4×200-char input must produce 401+401 batches, not 510+293).
3. **split_cores rule**: a non-word char only closes a word when there IS a
   pending punct (`punct` non-empty) or whitespace; consecutive letters append
   to the open core (per-char flush was the original bug: "Hello" → H,e,l,l,o).
4. **Workspace optional-dependency gotcha**: `[workspace.dependencies]` cannot
   be `optional = true`; the member crate declares `ort = { workspace = true,
   optional = true }` + `graph-ort = ["dep:ort"]`.
5. **CLI registration**: `cargo run -p tts` WITHOUT `--features graph-ort`
   reports "unknown lane kokoro" — the gate runs with the feature everywhere.

## Debug hooks

- `TTS_DEBUG_G2P=1` traces per-chunk batches (`ps` string + token count) to
  stderr — left in place for T005 diagnosis.

## Done-conditions — all green via `./scripts/build.sh`

- [x] `tts synth --lane kokoro --model <onnx path>` produces judged-intelligible
      speech (te nemo: near-verbatim transcription)
- [x] chunks.jsonl credible stage breakdown + non-trivial TTFA
- [x] Two cold runs byte-identical (determinism gate) — enforced in build.sh
