# q-laws ledger — TTS class

`te`-style: one entry per hypothesis, adopted/shelved by measurement only.
Carried forward from the STT repo's laws so they can be confirmed or refuted
for TTS. Zen 2 4800H, release, PTEXT unless noted.

## L1 — q8_0 ≥ f16/f32 weight-precision (STT transducer law)

**Confirmed INVERTED-for-speed, REFUTED-for-law: f32 is faster than q8 on xn/Zen 2.**
- Same audio (65,280 samples / 2.800 s), same dial threads=1:
  | weights | TTFA | post-first drain | RSS (lane) |
  |---|---|---|---|
  | q8 `model.q8.gguf` (64 MB) | 393 ms | 821 ms | 180 MB |
  | f32 `model.safetensors` (99 MB) | 54 ms (probe) / 252 ms (lane cold) | ~510 ms | 280–300 MB |
- TTFA **7× faster**, drain **~1.6×** faster on f32. int8 weight-only matmul
  does NOT win on this stack — x86 q8 kernel path is slower than native f32.
- **Adopt f32 for latency IF a determinism fix lands (see L4); until then
  f32 stays behind `TTS_PTTS_QUANT=f32` (experimental, ungated).**

## L2 — KV preflight / session reuse beats per-request conditioning

**SHELVED — measured NULL.** `Synth::session(opts,·)` primes voice once
(~2–17 ms); primed TTFA 391 vs cold 393 ms at q8/1t — 1.01×. Conditioning is
~ms-class on phonon-31m; session machinery also shifts emission shape by one
frame (67,200 vs 65,280) → would break byte-parity with the gate for no gain.
Not wired into the lane.

## L3 — more threads = free speed on CPU

**SHELVED for published rows.** xn parallel kernels are order-nondeterministic
above 1 thread (chunk-level divergence, T006). Speed IS there (q8 8t: ~3.4×
rtf, 890 ms TTFA cold) but rows can't pass byte-identity → nondet lanes are
info-only until a fixed-partition/ordered-reduce knob exists in xn.

**AMENDMENT (T009 close-out):** thread-count is not the only nondeterminism
source — see L4 below (decode-batch grain race fires at ANY thread count
under load). Threads=1 is necessary but not sufficient for long-text
byte-identity.

## L4 — determinism scope (NEW LAW for this engine class)

ptts determinism is **dial- and shape-local**: identical runs in the same
process shape are bit-exact at threads=1 (q8 gate passes), but f32
cross-process runs diverge even at threads=1 (chunk COUNT 7 vs 5, byte
divergence at 73 KB) — worker-thread emission timing feeds the bounded-channel
drain. Consequence: **identity gate must be re-run per (lane, quant, dial)
triple**; it caught f32 being ungated-fast on first contact, which is the gate
doing its job.

**AMENDMENT (T009 close-out — grain race):** hard-500 FAILs at threads=1
(493/7 in run 1540, random FAIL set across runs) are NOT xn FP order — they
are the ptts decode-batch **grain race**: the drain timing shifts chunk
boundaries → per-chunk voice restart changes bytes (row 129: 57 vs 58 chunks;
4 standalone reps byte-identical on an idle box; WER + total duration
unaffected — mean 56.94 vs 56.98 across the two hard-500 runs). Long multi-
batch texts can FAIL identity at ANY thread count under load; short (≤1
decode batch) texts are stable. The candle lane decodes inline (no channel)
and passed identity everywhere (8/8 + 80/80 + all rep pairs). L4 gate law
intensifies: the triple alone isn't enough for ptts-class long texts under
load — verdicts on ptts rows should be stated with the FAIL budget measured
(no FAILs = hard-80; ~1.4 % = hard-500/1t; ~2.2 % = hard-500/threads>1).

## L5 — first-stream init cost is amortizable (resident pool)

**ADOPTED (lane-level contract).** First stream per process pays LM/state
init: q8 ~2.66 s @1t (~0.4 s @8t), f32 ~0.4–0.6 s. Warm requests then stream
at 392 ms (q8) / ~250 ms (f32, cold wall 833 ms incl. build). Any service
shape should hold a warm lane resident, not pay init per process — this is
the TTFA story in the scoreboard, not a bug.

## Ledger status

- [x] ≥4 hypotheses with verdicts (L1–L5)
- [x] Parity gate vs candle oracle — **PROVEN (T009 close-out)**: probe
      48-step latent parity (max |d| ≤ 1e-4) + lane WER battery (candle
      hard-80 med 42.86 vs ptts 46.43, internal 7/8 within 0.5 pt with the
      address outlier root-caused as seed drift + judge normalization) +
      candle byte-identity 100 % across corpora/dials + cross-quant PCM
      bound documented (rms 0.9–2.9 % q8-vs-f32, med 0).
- [x] Default-lane decision — **re-confirmed at M2 close**: kokoro remains
      default (throughput lane, rtf 0.80 / TTFA 4.6 s); flow-candle becomes
      the resident-latency lane (TTFA 352 ms — 13× faster — but wall-RTF
      0.334 ≈ 2.4× slower; session-4 target = decode batching + gemm to
      close it). f32 flip still pending L4-class determinism inside candle
      (not needed: candle is deterministic as-is).
