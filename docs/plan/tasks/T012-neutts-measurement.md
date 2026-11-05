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
   now ships official **`neuphonic/neucodec-onnx-decoder`** (f32) and
   **`neucodec-onnx-decoder-int8`** — an on-device decoder artifact. Phase-2
   effort is therefore OUR EXISTING `graph-ort` machinery, not a custom ONNX
   export. (Both repos HF `gated: auto` — need a token with conditions
   accepted; none on this box → decode rate left UNMEASURED.)

### Phase-1 measurements (2026-11-05, Zen 2 4800H/16T, llama.cpp 0.7.0-dev)

Official NeuTTS GGUFs are HF-gated (401 without token) → measured the
identical compute path with the ungated Qwen2-0.5B-Instruct GGUF (same
qwen2 arch the official artifact declares), plus the authors' own number on
the official file:

| source | artifact | gen rate | vs 50 Hz budget | RSS |
|---|---|---|---|---|
| our box, Qwen2-0.5B Q4_0 proxy, llama.cpp | ungated | **215–285 t/s** (prompt-dependent ±15%) | **4.3–5.7×** | 876 MB |
| our box, Qwen2-0.5B Q8_0 proxy | ungated | **229 t/s** | **4.6×** | 1.05 GB |
| authors, official neutts-air Q4_0, llama-bench, Ryzen HX 370 | official | **119 t/s** (500p + 250g) | **2.4×** | — |

- Effective decode was **single-core** in this build (user≈0.93 s vs wall
  2.15 s; rate did not change across `-t 1..16`) — 0.5B q4_0 is too small to
  fan out per-token; the rate is a 1-core floor, not a ceiling.
- Load wall ~1.3 s (mmap, 353 MB file) vs community torch load **174.93 s**
  — the 3-minute community load is the Python stack, not a GGUF property.
- Scale caveat: official Q4_0 file is 748 MB ≈ 2.1× our 353 MB proxy (bigger
  vocab/embeddings or heavier backbone) → scale our rate down ≈2× → ≥2×
  headroom, consistent with the authors' 119 t/s on the real file.

### Community numbers this settles (HF discussion #6, 2025-10)

15× slower than realtime, load 174.93 s, voice encode 74.56 s — measured on
the **Python transformers + CPU codec chain**. Our rate table shows the
backbone alone is NOT the cause of 15×; the cost lives in
backbone-in-transformers + codec/voice-encode on torch CPU.

## Verdict: SHELVE as a lane (measurement complete)

- **CPU-realtime claim: CORRECTED, not confirmed** — with a llama.cpp-class
  backbone path the 50 Hz budget clears ≥2× on this hardware class; the
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
