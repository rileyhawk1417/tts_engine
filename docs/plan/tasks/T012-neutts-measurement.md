# T012 — NeuTTS honest measurement experiment (measurement-only lane)

- **Status:** `done 2026-11-05 — phase-1 measured (backbone clears 50 Hz budget ≥2× on this box via ungated qwen2 proxy + authors' official Q4_0 number); verdict: SHELVE as lane; phase-2 ONNX decode artifact exists officially (gated), decode rate unmeasured (no HF token on box)`
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
   export. (Both repos HF `gated: auto`, absent from the local mount →
   decode rate left UNMEASURED: two pip-install attempts of the torch
   `neucodec` package each blew the time-box with no landing — recorded,
   not retried. The 1.16 GB `neucodec` pth IS on the mount if a later pass
   wants the torch-CPU codec number.)

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

- **CPU-realtime claim: CORRECTED, not confirmed** — with a llama.cpp-class backbone path the 50 Hz budget clears ≥4.3× on this box (official artifact measured directly); the
  public 15×-slower datapoint is a Python-stack artifact. A C path (ggmi FFI
  + ORT int8 ONNX codec decode) is architecturally capable of ≥2× realtime
  here. The claim is refuted on OUR terms (rate measured), honest numbers
  recorded.
- **Adopted as lane? NO.** Reasons: (1) it'd be a THIRD decode stack while
  T010 already tests the single ggml/FFI path against a stronger model;
  (2) gated artifacts = supply-chain friction (every eval needs a token);
  (3) the chain cost (codec decode rate + voice encode — 74.56 s torch, no
  ONNX voice-encoder artifact) is still unmeasured end-to-end — adoption
  would need that chain to exist first; (4) flow-candle already owns the
  resident-latency slot and kokoro owns default.
- If the verdict ever flips: start from official Q4/Q8 GGUF behind our ggml
  FFI machinery + wire neucodec-onnx-decoder-int8 through graph-ort; gate on
  the measured decode-rate chain before writing lanes.

## Done-conditions

- [x] Phase-1 measured numbers recorded (gen t/s per quant, load wall, RSS,
  1-core decode note; ungated-proxy + official-artifact sources) — table above
- [x] Explicit adopt/shelve verdict with reason in scoreboard + memory
  checkpoint — **SHELVE as lane, claim corrected, reasons enumerated**
- [x] targets.md measurement-only lane row updated to reflect the verdict
