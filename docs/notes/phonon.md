# Phonon lane notes (flow-ptts, T006)

First-party `ptts` 0.6 crate (dual MIT OR Apache-2.0, repo `/tmp/xn-ptts`) running
`gradium/phonon-31m-en` on CPU behind our seam. These are the measured facts and
lane laws that aren't obvious from the code.

## Checkpoint on disk (gitignored, fetch once)

`models/phonon-31m-en/`: `config.json`, `tokenizer.json`, `model.q8.gguf`
(64.5 MB, Quant::Q80), `default-voice.safetensors`. Built without the `hub`
feature on purpose — no network in the load path; `ModelRef::Hub` is rejected
by the lane. Lane accepts the checkpoint dir or any file inside it.

## Architecture map (what runs where)

- ptts = flow-LM (31M causal LM, SentencePiece 4001 raw-text input) + one-step
  flow head + small Mimi decoder fused into the stream pull. Mimi IS the
  vocoder → seam slot `vocoder` is fusion, reported zero (kokoro fused-graph
  rule extends here).
- Lane-side g2p slot = `ptts::preprocess::Normalize` built-in per-language
  normalizer (espeak NOT needed on this lane — phonon reads raw text tokens).
  Measured in the g2p slot on chunk 0.
- 24 kHz mono, SentencePiece 12.5 Hz latents, EOS-headed — ptts decides chunk
  emission internally (stream chunks ≠ sentence chunks; a 31 s paragraph came
  out as 88 stream chunks).

## Determinism law (the big one)

**xn CPU FP kernels are order-nondeterministic under rayon parallelism.**
Same binary, same seed, `TTS_XN_THREADS=4`/`8` twice → bytes diverge mid-stream
(chunk 0/1 sha256 often identical; later chunks drift; stream SHAPE can even
differ — chunk counts have differed across runs). `TTS_XN_THREADS=1` is
bit-identical. There is no fixed-partition/deterministic-parallel knob in xn.

So the lane reports dial-aware determinism (seam §7):
- `TTS_XN_THREADS=1` → `Seeded` — gate-passable. This is the eval/gate dial
  (`eval/run.sh` and `build.sh` set it for the flow-ptts lane).
- threads > 1 → `NonDeterministic` with the reason in telemetry, e.g.
  `"xn cpu parallel FP reductions are order-nondeterministic at 8 threads;
  TTS_XN_THREADS=1 restores bit-identity at seed"`.

The `xn::set_num_threads` dial sizes ONE process-global rayon pool and must
happen before the first tensor op; later re-dials are silently ignored by
rayon, so the lane guards + warns, and `xn_thread_dial()` fixes the value
process-wide at first touch.

## Measured on the Zen 2 box (8C/16T, release builds, 20-word sentence)

| dial | wall | rtf (audio/wall) | TTFA | RSS |
|---|---|---|---|---|
| xn threads=1 | 5.7 s | 1.23 | 3.6 s | 187 MB |
| xn threads=8 | 2.0 s | 3.44 | 0.9 s | 335 MB |
| kokoro @10t intra (same text, reference) | 11.8 s | 0.77 | 11.8 s | 407 MB |

- Parallel xn is where the speed is; **at the byte-identity dial (threads=1)
  flow-ptts is only ~1.6× kokoro wall-relative and TTFA lands ~3.0–3.6 s**
  (1-thread LM per-chunk cost). Internal-corpus gate dial row: rtf 0.77–2.0
  (long paragraphs amortize), TTFA ~3.1–3.5 s, RSS **190 MB max vs kokoro 457 MB**
  — memory is the lane's real gate-dial win, plus the long-paragraph rtf.
- The 3–4× knobs (threads=8) can't publish as seed-byte-identical until T009
  finds an order-stable parallel path (or we relax the gate law intentionally
  — not before T009 weighs in).
- ptts CLI's advertised 110 ms first-chunk excludes voice conditioning; our
  TTFA = seam TTFA includes conditioning + first LM pass. Session priming
  (`Session::session(opts, max_seq_len)`, conditions once) is the T007 dial
  that removes conditioning from per-request TTFA.

## Quality probe (internal corpus, nemo judge, gate dial)

WER mean **4.71%** over 8 internal sentences (quoted-dialogue 10%,
address 16.7%, numbers-dates 7.7%, three sentences at 0%). Identity: pass.
Full matrix (hard-80 × nemo/parakeet) lands in T007.

## Voices

- `default-voice.safetensors` registers as voice `default`; generation without
  any voice is NOT supported (LM needs conditioning) → the lane fails load if
  the checkpoint ships no voice.
- `VoiceRef::Path` → registered at Synth build (`builder.add_voice`) because
  ptts bakes voices at build time; rebuilds happen on lang change or when a
  path voice arrives.
- Per-voice LICENSE recording still pending (`gradium/phonon-voices` ~100
  voices — targets.md open item; only matters when we bundle voices).

## Port-relevant facts for T008 (candle oracle)

- Weights: q8 gguf (GGUF Q80 = q8_0-style) for the lane; bf16 safetensors for
  the port. q80 chosen over f32: 64.5 MB, faster loads, and phonon's audio
  quality tolerance is high — measured WER difference unknown for f32 (T009
  q-law row).
- LM quant path showed thread-stable hashes in chunks 0/1 (integer
  accumulation); divergence enters via FP paths (flow head / Mimi decode) —
  relevant constraint for the port's parallel plan.
- `SpeechOptions` (voice/temperature/seed/cfg_coef/max_tokens_per_chunk/
  post_process) all settable per request; our lane wires seed + voice, leaves
  temperature/cfg at ptts defaults (record defaults once T007 sweeps).
