# T007 — Phonon evaluation: measurement-class correction + verdict

Date: 2026-11-05, Zen 2 4800H (8C/16T), release, PTEXT = "The engine loads its
weights straight from disk." unless noted. Probe tools (kept in repo):
- `examples/ptts_session_probe.rs` — raw ptts dials (cold vs primed session).
- `examples/flow_ptts_lane_probe.rs` — the actual lane path, phase-bisected.

## The T006 numbers were measured in the wrong class — three corrections

1. **Double normalization (real bug, fixed):** the lane pre-normalized text
   with ptts `Normalize`, then `stream_with` ran the synth's own normalizer
   inside `plan_chunks` again. Same sentence+seed: pre-normalized 103,680
   samples vs raw 65,280 — pre-normalizing STRETCHES generation 1.6×. ptts's
   per-language normalize IS the g2p slot; it is fused into `stream_with`
   (lane passes RAW text now; g2p stage reports the fusion, `Duration::ZERO`,
   same contract as the fused Mimi decode). For PTEXT this was a no-op
   (idempotent on that sentence) — the "The river is 3,000 km" sentence
   exposed it.
2. **Default quant is F32 (`Quant::F32`), not q8.** The first probe ran f32
   weights; the lane pins `Quant::Q80`. f32 is **2.4× faster per frame** on
   Zen 2 (f32 ~15 ms/frame vs q8 ~33 ms/frame), and produces different audio
   (65,280 vs 67,200 samples — EOS lands one frame earlier). All ptts-vs-lane
   comparisons must pin `ResolveOptions{quant: Q80}`. NOT a bug in the lane —
   but the single biggest T009 lever (see q-law below).
3. **"xn 1 thread = 3 s TTFA" was first-stream init cost, not streaming cost.**
   At q8/1t the first stream pays ~2.66 s of LM/state init (scales ~linearly
   with threads: ~0.4 s at 8 threads). After warmup the SAME dial streams at
   TTFA ~392 ms. Cold per-process CLI ≈ 3.0 s TTFA; a resident lane pays it
   once.

## Measured matrix (q8 = `model.q8.gguf`, deterministic 1t unless noted)

| config | TTFA | rtf (audio/wall) | determinism |
|---|---|---|---|
| q8 1t cold (CLI/gate, single-shot) | ~3050 ms | 0.72 | seeded, byte-identity PASS |
| q8 1t warm (resident lane) | 392 ms | ~1.74 | seeded (rep-identical) |
| q8 8t cold (info only) | ~890 ms | ~3.1 | nondeterministic (xn rayon) |
| f32 1t (T009, experimental) | 53 ms probe / 252 ms lane cold | 3.36 | NOT deterministic at 1t — ungated |
| kokoro int8 ORT (T005 gate) | ~3118 ms | 0.84 | deterministic |

Session primer: `session_default` primes voice in ~2-17 ms; primed TTFA 391 vs
cold 393 ms — **1.01×, a measured NULL result**. Voice conditioning is cheap on
31m; the primed-session machinery is NOT worth wiring into the lane (their
primed path also emits one extra frame — Session would break byte-parity with
the gate for zero gain). The lane stays on `stream_with` (per-call ephemeral).

## Claims vs measured (Gradium README perf table + research memory)

| Their claim | Ours (same class as their claim) | Delta verdict |
|---|---|---|
| M5 1-core: 26× · TTFA 5 ms (KleidiAI) / 13.6 ms vanilla | Zen 2 f32 1-thread: ~5× · TTFA ~53 ms | same class, slower machine; TTFB metric matches once init is excluded |
| RPi5 1-core: 2.3× · 68 ms | Zen 2 q8 1-thread warm: 1.74× · 392 ms; f32: 5× · 53 ms | q8 kernel is the bottleneck (see q-law), not the theory |
| "ms-class" TTFA streaming frame-by-frame | 35-53 ms f32 / 392 ms q8 warm | REPRODUCED (after class correction) |
| hard-set WER 31m 1.70% vs Kokoro 3.50% (LLM-judged) | ours: te round-trip WER (their `tts-hard-sentences-202610` corpus, hard-80 stride subset) — hard-80 running, see eval/results | methodology gap noted (LLM-judge vs ASR round-trip); not comparable 1:1 |
| Kokoro (their PyTorch ref): 7.3× · 776 ms TTFB on M5 | our kokoro ORT int8: 0.84× rtf · 3.1 s TTFA | different runtime class (PyTorch vs our ORT int8), both measured |

## q-law datapoint (T009 seed)

xn CPU: **q8 is 2.4× SLOWER than f32 per frame** on Zen 2 x86 (the inverse of
the STT transducer law q8_0 ≥ f16). int8-not-faster holds on this stack;
audio length even differs by a frame (EOS timing). T009 must re-measure q81 /
q8k variants before any quant default; f32 is the latency-king candidate with
~124 MB weights.

## Verdict (M2 default-lane question)

flow-latent (phonon-31m) does NOT replace graph-ort kokoro as the default with
current q8 gate numbers: cold single-shot TTFA 3.0 s ≈ kokoro's 3.1 s, and
hard-80 WER is pending. A **resident** flow-ptts lane wins decisively: warm
TTFA 392 ms (kokoro compute-bound — pooling does not help it, T005) and warm
rtf ~1.74 vs kokoro 0.84 — 2× kokoro's throughput at 1 thread, at a
deterministic dial, 190 MB RSS vs 457 MB. With f32 it is 53 ms TTFA / 5× rtf.
Decision: kokoro stays default lane for now; flow-latent is the resident-latency
lane (the M2 M5-class targets are met with the measurement corrections
applied); revisit default on T009 after the f32/q8 row and identity gate
certification on the f32 path.

## Open items (next session)

- hard-80 flow-ptts nemo row: running in background (1t, identity-gated);
  board row lands next session (eval/results/20261105-*-flow-ptts-hard-t007).
- Voice matrix (3 spread voices × fixed subset) — mechanics check pending.
- f32 lane variant: `ModelRef`-suffixed quant dial (`--quant f32`), identity-
  gate the f32 path, then re-run the internal matrix.
- Re-run internal n=8 corpus with the raw-text fix (numbers above are PTEXT/
  probe-class; row refresh cheap).
