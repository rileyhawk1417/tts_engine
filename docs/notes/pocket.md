# Pocket-tts — second flow-candle checkpoint family (T015)

Status: **done (2026-11-06)**. `kyutai/pocket-tts-without-voice-cloning`
(english, `alba` fallback voice) rides the existing `flow-candle` lane as a
second checkpoint family — `--model models/pocket-tts-en`. Same transformer /
flow head / Mimi decoder code as phonon-31m; only family-specific wiring was
added. Phonon-31m stays byte-equal (regression gate below).

## Checkpoint provenance

- Repo: `kyutai/pocket-tts-without-voice-cloning` (token-accessible). NOT the restricted `kyutai/pocket-tts` (401/403).
- Derived from the voice-cloning weights: `mimi.encoder.*` zeroed (43 tensors, `remove_voice_cloning_and_push.py`) then F32→BF16 (`switch_to_bf16.py`). Encoder keys ride along zeroed and are dropped by the standard remap (decode-path only).
- Weights license CC-BY-4.0 (gated, prohibited-use clause); **pocket-tts code MIT** → vendorable if ever needed.
- Staged at `models/pocket-tts-en/` (367 MB, gitignored): `model.safetensors` blob `be9c6b4876d3f307…` (214 BF16 tensors: 127 flow_lm + 87 mimi), `tokenizer.json` sha256 `f498428e1eafee50…` (HF Unigram, 245,020 B), `embeddings/` 27 English voices.

## What landed (code)

| File | Delta |
|---|---|
| `candle_ptts/loader.rs` | (a) remap: `flow_lm.bos_before_voice` → `…speaker_wavs.sink` (pocket stages the bos row as a prefix sink — `insert_bos_before_voice: true` in the YAML). (b) `LoadedVoice::{Emb, Kv}` + `load_voice()` single entry point; `…self_attn/cache` keys route to `voice_cache_from_map` (per-layer `[2,1,F,H,D]` planes + I64 offset, layer offsets must agree). `load_voice_emb` delegates and rejects KV voices with an explicit error. |
| `candle_ptts/cfg.rs` | `Family { Phonon, Pocket }` on `TTSConfig` (`"pocket"` in config.json, absent = phonon); helpers `min_frames_before_eos()` (0 / 6), `fallback_voice()` (None / `alba`); `TTSConfig::is_phonon`. |
| `candle_ptts/model.rs` | `import_voice_kv()`: writes each layer's pre-baked `(k, v)` plane at offset 0 into the state cache, zero-pads to budget, sets `current_end = offset`. No backbone pass — the file IS the prefix. `TTSModel.min_frames_before_eos` from config. |
| `candle_ptts/synth.rs` | `Voice { name, payload }` with `frames()`; `prompt_base` dispatches Emb→`prompt_audio` / Kv→`import_voice_kv` / None→`prompt_no_voice`; EOS gate becomes `frames_done > min_frames_before_eos && should_stop` (phonon 0 → unchanged). |
| `lanes/flow_candle.rs` | Voice loading via `load_voice`; family law (emb voice on pocket config or KV voice on phonon config → `LaneError::Load` at startup); `VoiceRef::Default` resolves via `cfg.fallback_voice()` with a registered-voice check. |

## Voice = KV-cache law

Pocket voices are 18-tensor safetensors: 6 ×
`transformer.layers.{0..5}.self_attn/cache` `[2,1,F,16,64]` F32 (plane 0 keys,
plane 1 values, rope already baked at positions `0..F`) + 6 × `…/offset` I64
(all equal F). Import is a pure cache write: positions, mask, and rope all key
off `current_end` in `attention.rs`, so a text prompt continues at absolute
position F exactly as pocket python would. `complete_kv` appends after the
imported prefix; the causal mask keeps voice frames visible to all queries.

## Config (authored — `models/pocket-tts-en/config.json`)

(Checkpoint dir `models/` is gitignored — the config is not tracked; the
complete lock is transcribed below.)

Every value sourced from `english.yaml` of the pocket-tts CLI: flow_lm
d_model 1024 / heads 16 / layers 6 / ffn 4096 / max_period 10000, n_bins 4000
(LUT → `conditioner.embed.weight [4001,1024]`, padding row baked in — same
loader contract as phonon), lut_dim 1024, flow {lsd, depth 6, dim 512},
ldim 32, gelu tanh, time_rms default (RmsVar); mimi sr 24000 / frame_rate 12.5
/ seanet 512 / ratios [6,5,4] / transformer 512-8-2-2048 context 250
layer_scale 0.01 / quantizer 32→512; temperature 0.3 (informational — the
driver pins 0.3), lsd_decode_steps 1, eos_threshold −4.0; `model_id: null`
(no HF oracle → gelu default Tanh); tokenizer `tokenizer.json`;
`cfg_null_audio_empty: true` + `prefix_sinks: {speaker_wavs}` (load-guard
law). `time_rms_norm` omitted → enum default **RmsVar** — sourced, not
inferred: `ptts-0.6.0/src/mlp.rs` documents `RmsVar` as "the variance centred
and unbiased, as pocket-tts computes it (`torch.var`). The default, so
pocket-tts checkpoints load unchanged" (phonon = uncentred `Rms` from
audiocraft). Both variants live in `mlp.rs` since T008; pocket's
TimestepEmbedder alpha tensors load through the same path.
`emb_mean`/`emb_std` are the latent-side normalization buffers (both
families), not time-embedder state.

## Deviations (labeled, T008 law)

1. BF16 checkpoint — per-get cast to F32 in candle-nn's map backend; zero loader changes.
2. KV-cache voices — `LoadedVoice::Kv` + `import_voice_kv` (prompt length from the recorded offset).
3. `bos→sink` remap row added to `remap_key`.
4. Config authored by us; the pocket repo ships YAML, not our config.json schema.
5. Fork caveat: pocket binary builds xn-ptts 0.2.11 (their fork); our lane pins ptts 0.6.0 / xn 0.2.13. Parity target = pocket **python semantics**, verified structurally.
6. No ptts-0.6.0 oracle for pocket → parity gate = self-identity (3× byte-identity) + structural equivalence (probe inventory, config audit), not 48-step ReplayRng trace.
7. `TTSConfig.family` — new optional field, default phonon.

## Gates + numbers (4800H, release, seed 42, nemo-int8 judge `te @38b3d623`)

- Structural probe (`flow_candle_probe models/pocket-tts-en`): 214 tensors, 214/214 visible through VarBuilder, 170 used by remap (43 zeroed encoder tensors unused), `TTSModel` built in 103 ms.
- Round-trip: "The quick brown fox…"/"…attention caches…" verbatim under nemo; long-text (4 sentences) near-verbatim with judge-artifact WER.
- Determinism: pocket 3× byte-identical (`43976644dbf857ea…`), `cmp` green; phonon 3× unique-sha=1 and byte-equal to pre-T015 outputs (regression unchanged).
- Unknown voice → error listing all 27 names; fallback (`VoiceRef::Default`) → `alba`.
- **Hard-80 battery** `20261106-0141-flow-candle-hard-t015-pocket-alba`: **80/80 identity pass → trusted**; WER mean 37.06 / median 38.75; TTFA med 503 / p95 521 ms; rtf med 0.220 (wall/audio ≈ 4.5×); peak RSS 666,376 KB.

Same-judge hard-80 table (WER mean / med lower is better):

| row | WER mean | med | rtf med | TTFA med | RSS max |
|---|---|---|---|---|---|
| kokoro int8 | 42.03 | 25.00 | 0.80 | 4596 ms | ~457 MB |
| piper ryan-high | 33.01 (500) | 20.00 (500) | 3.73 | 663 ms | 343 MB |
| flow-candle phonon-31m | 60.28 | 42.86 | 0.334 | 352 ms | 219 MB |
| flow-candle pocket-tts-en / alba | **37.06** | 38.75 | **0.220** | 503 ms | 666 MB |

Pocket-alba WER mean beats every flow-latent row and lands between piper ryan
and kokoro on the mean; the median is worse than kokoro/ryan but the mean
gain is the notable one (the adversarial-corpus outliers are smaller). RSS ≈
650 MB (BF16→F32 materialization + KV voices) is above the phonon lane's 219
MB but well under the ≤2 GB resident-service headroom; speed is the fastest
flow-class row measured (inline candle decode, 1 step).

## Verify list (all checked 2026-11-06)

- [x] flow_lm width law: every flow_lm tensor ≥ 1024-wide on the transformer axis (probe top sizes + 214/214 present).
- [x] `cond_embed.weight [512,1024]`, `input_linear [1024,32]`, `out_eos [1,1024]`, `out_norm [1024]` exact (loader shape-checked at build).
- [x] mimi decode-path only: 43 `mimi.encoder.*` zeroed tensors dropped by remap (unused-by-model sweep).
- [x] Voice KV import: 6 layers × `[2,1,F,16,64]` F32 + I64 offset for all 27 voices (family law validates every voice at load); unknown voice errors.
- [x] Text prompt: Unigram ids through `tok.rs` (dtype-agnostic), LUT n_bins 4000 + baked padding row (embed `[4001,1024]`).
- [x] EOS −4.0 + min-frames-before-EOS 6 (probe prints `eos_threshold=-4`; long-text terminates on real EOS).
- [x] `bos_before_voice [1,1,1024]` lands in `speaker_wavs.sink` via PrefixSink (TTSModel::load maps the conditioner prefix; sink required at load).
- [x] Phonon-31m regression unchanged (byte-equal to prior session outputs).
