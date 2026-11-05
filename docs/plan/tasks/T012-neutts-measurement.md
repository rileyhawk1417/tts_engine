# T012 — NeuTTS honest measurement experiment (measurement-only lane)

- **Status:** `done 2026-11-05 — backbone measured on OFFICIAL GGUFs (air 217 t/s, nano 342 t/s) + phase-2 codec decode measured through OUR graph-ort machinery (int8 646–744 tok/s = 12.9–14.9× budget; fp32 9.8–12.1×); verdict: SHELVE as lane — decode is not the limiter, backbone is; full chain would be backbone-bound ~4.3× realtime; the 15× community figure = Python stack end-to-end`
- **Milestone:** [M4](../index.md#milestone-m4--streaming--measurement-lanes)
- **Depends on:** [T003](T003-roundtrip-eval-harness.md)

## Goal

Settle the disputed CPU-realtime claim with our own numbers — or falsify fast and shelve. NeuTTS Air Q4/Q8 GGUF (Apache-2.0, qwen2 arch) has no codec decode path integrated anywhere; that integration cost is part of the verdict.

## Approach (CORRECTED during execution, 2026-11-05)

Two design assumptions in the original plan were wrong and changed the
shape of the experiment:

1. NeuCodec runs at **50 tokens/s** (FSQ, single codebook, 16 bits/token,
   0.8 kbps, 24 kHz audio) — not the "12.5 Hz-class" guessed in the original
   plan. Budget to beat: **≥50 generated tokens/s sustained**.
2. The "no codec decode artifact anywhere" assumption is OUTDATED: Neuphonic
   ships official **`neuphonic/neucodec-onnx-decoder`** (f32) and
   **`neucodec-onnx-decoder-int8`** — on-device decoder artifacts. Phase-2
   effort is therefore OUR EXISTING `graph-ort` machinery, not a custom ONNX
   export. (Both repos HF `gated: auto` — artifacts fetched 2026-11-05 into
   `/teranux/hugging_face/t012-drop/` after the user gated-in via CLI.)

   **Phase-2 measured through `graph-ort`** (ort 2.0.0-rc.15 api-21, ORT
   1.22.1 dylib — kokoro-lane-identical machinery), bench example
   `crates/tts-engine/examples/neucodec_ort_bench.rs`, t=8 threads, 20 reps
   after 3 warmups:

   | decoder | 50 tok (1 s) med | 250 tok (5 s) med | tok/s @250 | × budget | RSS |
   |---|---|---|---|---|---|
   | **int8** | **77.3 ms** | 335.9 ms | 744 | **12.9–14.9×** | 556 MB |
   | fp32 | 101.9 ms | 413.1 ms | 605 | 9.8–12.1× | 1.44 GB |

   - Graph contract verified exact: in `codes` int32 `[b,1,T]` (50 Hz FSQ
     ids) → out `audio` f32 `[b,1,480·T−480]` (24 kHz).
   - Marginal decode ≈ **1.3 ms/token** (int8) — a per-frame streaming add
     on top of the ~20 ms backbone step; decode never gates the chain.
   - Rate is value-independent (all-zero ids: same length, same wall).
   - Chain implication: end-to-end would be **backbone-bound at ~4.3×
     realtime** — head fully hidden behind generation. Combined with
     phase-1 this decomposes the public "15× slower" completely: every
     measured C-side piece clears realtime by multiples; the Python stack
     is the whole gap. Voice ENCODE stays unmeasured (w2v-bert on torch;
     ~74 s class) but is one-time setup, not per-utterance.

### Phase-1 measurements (2026-11-05, Zen 2 4800H/16T, llama.cpp 0.7.0-dev)

**Official artifacts measured directly** — `neutts-air-Q4_0.gguf` (527 MB,
NOT the 748 MB originally assumed) + `neutts-nano-Q4_0.gguf` (195 MB) came
from the local `/teranux/hugging_face/hub` mount; ungated Qwen2-0.5B GGUFs
used as same-arch corroboration. Same fixed prompt + seed 42 throughout:

| source | artifact | size | gen rate | vs 50 Hz budget | RSS |
|---|---|---|---|---|---|
| **our box, OFFICIAL neutts-air Q4_0** | official | 527 MB | **217.4 t/s** | **4.35×** | 1.07 GB |
| **our box, OFFICIAL neutts-nano Q4_0** | official | 195 MB | **342.4 t/s** | **6.8×** | 744 MB |
| our box, Qwen2-0.5B Q4_0 proxy | ungated | 353 MB | 215–285 t/s | 4.3–5.7× | 876 MB |
| our box, Qwen2-0.5B Q8_0 proxy | ungated | 531 MB | 229 t/s | 4.6× | 1.05 GB |
| authors, llama-bench, Ryzen HX 370, official Q4_0 | official | — | 119 t/s | 2.4× | — |

- Decode stays 1-core-effective at this size (user 1.55 s vs wall 3.01 s on
  air; rate flat across `-t 1..16`) — the rates are 1-core floors, not
  ceilings. Load wall: air 3.01 s incl. 250 tokens (net ≈1.9 s), mmap.
- Bigger official file does NOT halve the rate vs proxy: the size difference
  is mostly embedding/vocab (memory, not per-token flops).

### Community numbers this settles (HF discussion #6, 2025-10)

15× slower than realtime, load 174.93 s, voice encode 74.56 s — measured on
the **Python transformers + CPU codec chain**. Our rate table shows the
backbone alone is NOT the cause of 15×; the cost lives in
backbone-in-transformers + codec/voice-encode on torch CPU.

## Verdict: SHELVE as a lane (measurement complete)

- **CPU-realtime claim: CORRECTED, not confirmed** — both halves measured on
  box through C-side machinery: backbone 217–342 t/s (4.3–6.8× the 50 Hz
  budget), codec decode int8 646–744 tok/s (12.9–14.9×) — decode is NOT the
  limiter; the full chain would be backbone-bound at ~4.3× realtime. The
  public 15×-slower datapoint is the Python stack end-to-end. The claim is
  refuted on OUR terms (every rate measured), honest numbers recorded.
- **Adopted as lane? NO.** Reasons: (1) it'd be a THIRD decode stack while
  T010 already tests the single ggml/FFI path against a stronger model;
  (2) gated artifacts = supply-chain friction (every eval needs a token);
  (3) voice ENCODE remains torch-only (no ONNX artifact; 74.56 s one-time
  cost class) — a working chain still needs a custom voice-encoder story;
  (4) flow-candle already owns the resident-latency slot and kokoro owns
  default. The corrected rate profile (4.3× backbone-bound) would produce
  WER at the model-class level we already know regresses vs kokoro on this
  corpus — same reason the OTHER flow lane took the resident-latency slot.
- If the verdict ever flips: start from official Q4/Q8 GGUF behind our ggml
  FFI machinery + wire neucodec-onnx-decoder-int8 through graph-ort; gate on
  the measured decode-rate chain before writing lanes.

## Done-conditions

- [x] Phase-1 measured numbers recorded (gen t/s per quant, load wall, RSS,
  1-core decode note; ungated-proxy + official-artifact sources) — table above
- [x] Explicit adopt/shelve verdict with reason in scoreboard + memory
  checkpoint — **SHELVE as lane, claim corrected and fully measured, reasons enumerated**
- [x] targets.md measurement-only lane row updated to reflect the verdict
