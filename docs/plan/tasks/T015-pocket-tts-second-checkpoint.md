# T015 — Pocket-tts as a second flow-candle checkpoint

- **Status:** not-started (audited 2026-11-06; ready to implement)
- **Milestone:** M2-flow expansion (second checkpoint in the existing `flow-candle` lane; **not** a new lane)
- **Deps:** T008 (candle port), T009 (parity gate); builds on T013 chunking via the lane
- **North stars:** `docs/notes/phonon.md` (candle port laws), `docs/plan/targets.md` (pocket row), upstream `kyutai-labs/pocket-tts` (MIT)
- **Decision source:** board verdict 2026-11-06 — pocket-tts IN, phonon-40m stays eval-only-NC

## Goal

Add `kyutai/pocket-tts-without-voice-cloning` (english) as a **second checkpoint family** loadable by the
`flow-candle` lane (`--model pocket-tts-en`), passing the same gates as phonon-31m: 48-step cross-impl parity
probe shape (fork: no ptts-0.6.0 oracle for pocket — parity = self-identity + structural equivalence), byte-identity
gate, and a hard-80/500 scoreboard row.

## Audit (COMPLETE 2026-11-06 — measured facts)

### Checkpoint + access
- Model repo `kyutai/pocket-tts-without-voice-cloning`: token-accessible (401 anonymous, 200 with token).
- **Tokenizer blocker RESOLVED**: repo ships `tokenizer.json` (245,020 B, HF **Unigram** model type) at repo root AND
  `languages/english/tokenizer.json`; the old `kyutai/pocket-tts` gate is irrelevant. Staged locally.
- Local model dir staged: `models/pocket-tts-en/` (gitignored):
  - `model.safetensors` = blob `be9c6b4876d3…cabb0a0` (219,029,196 B, **214 tensors, all BF16**) sha16 `be9c6b4876d3f307`
  - `tokenizer.json` sha16 `f498428e1eafee50` (Unigram)
  - `embeddings/*.safetensors` = **27 English voices** (~6.2–8.0 MB each)
- **Provenance scripts (first-party, repo root)**: `remove_voice_cloning_and_push.py` zeroes ALL `mimi.encoder.*`
  tensors (names kept); `switch_to_bf16.py` casts every F32 tensor → BF16. So blob = voice-cloning weights,
  encoder zeroed, BF16.
- Raw blob census: 127 `flow_lm.*` + 87 `mimi.*`; **zero `backbone.`/`condition_provider` names** — pocket ships
  ptts-layout keys already. bos/sink: `flow_lm.bos_before_voice [1,1,1024]`, `flow_lm.bos_emb [32]`. Speaker:
  `flow_lm.speaker_proj_weight [1024,32]`.

### Voice files = pre-baked per-layer KV caches (NOT wavs, NOT emb vectors)
- Per voice: `transformer.layers.{0..5}.self_attn/cache` **F32 `[2,1,F,16,64]`** + `self_attn/offset` **I64 `[1]`**;
  alba F=127 with offset=127 on every layer (all frames real); 11–12 tensors total; no `emb` key.
- Law: `k/v = cache[0]/cache[1]` planes `[1,F,16,64]`, `current_end = offset` — matches our `MhaState` exactly
  (`complete_kv` appends at `current_end`, identical offsets across batch required).
- num heads 16, head_dim 64 = d_model 1024/16 ⇒ `num_kv_heads` default (= num_heads) and `attn_dim` default
  (= d_model) are already correct for pocket via our `FlowLMTransformerConfig::num_kv_heads()/head_dim()` defaults.

### Config (`languages/english.yaml`, full dump banked in session memory)
- flow: `{depth: 6, dim: 512}` → FlowType **Lsd** (2 time conds, our default)
- transformer: d_model **1024**, num_heads **16**, num_layers **6**, hidden_scale 4 (ffn 4096), max_period 10000
- lookup_table: dim 1024, **n_bins 4000**, tokenizer "tokenizers"
- mimi: sample_rate 24000, inner 32 / outer 512 / frame_rate 12.5, seanet ratios [6,5,4] k7 rk3 lk3, transformer
  `{d_model 512, num_heads 8, num_layers 2, layer_scale 0.01, context 250, dim_feedforward 2048}`, quantizer dim 32
- `default_temperature 0.3`; `insert_bos_before_voice: true` → maps to `speaker_wavs` prefix-sink (xn-ptts fork)
- Rust defaults: `DEFAULT_SAMPLER_DECODE_STEPS=1`, `DEFAULT_EOS_THRESHOLD=-4.0`, `MAX_TOKEN_PER_CHUNK=50`,
  `DEFAULT_LANGUAGE="english"`, fallback voice `"alba"`, `MIN_FRAMES_BEFORE_EOS=6` (rust `model.rs`)
- EOS comparator is our law verbatim: `eos_from_logit = v > eos_threshold` with pocket threshold **-4.0** (phonon
  = 4.0). Same code, config value differs.

### Local capability check (candle_ptts/*)
- `cfg.rs`: `Gelu/TimeRmsNorm/FlowType/LutConditionerConfig` cover pocket; `mimi_with_gelu` falls mimi gelu to
  `default_gelu()` = **Tanh for model_id=null** (pocket) — pocket flow_lm and mimi both want tanh ✓.
- `model.rs`: `PrefixSink` + `SpeakerMimi`+`load_speaker_proj [(1024,32)]` + FlowType dispatch (Lsd/Drifting/
  Pointwise) all present; `voice_prompt_len`, `prompt_text`, `prompt_audio`, TTSState pub fields ready.
- `attention.rs`: `MhaState { pub k_cache, pub v_cache, pub current_end }`, `LayerState::FlowLm(MhaState)` enum,
  `StreamingTransformerState { pub layer_states: Vec<LayerState> }` — KV import is plain field write.
- `loader.rs`: `remap_key` = oracle table verbatim; `Weights::from_safetensors` keeps stored dtype;
  `var_builder()` = `from_tensors(map, F32)`; **candle-nn `SimpleBackend for HashMap` does
  `to_dtype(dtype)` per get** ⇒ **BF16→F32 cast needs ZERO loader changes**.
- `check_model_ext` accepts voices with no `model_ext` header (pocket ships `__metadata__: None`) ✓.
- `VOICE_DIRS = ["voices", "embeddings"]` and `checkpoint_voice_files` already scan `embeddings/*` ✓.
- `tok.rs`: dtype-agnostic HF tokenizers wrapper (Unigram fine) ✓.
- `plan.rs` chunk law: MAX_TOKENS_PER_CHUNK=50 == pocket MAX_TOKEN_PER_CHUNK ✓.

### Named deviations (T008 law — each labeled in code + note)
1. **BF16 checkpoint**: weights land BF16, cast per-get to F32 (materialization deviation of T008 s5 unchanged).
2. **KV-cache voices**: `Voice` gains a payload variant (`Emb(tensor)` | `Kv { planes, current_end }`); prompt
   length source branches (T vs T frames), base-state build imports per-layer K/V before prompt frames.
3. **bos→sink remap row**: `flow_lm.bos_before_voice` → `flow_lm.condition_provider.conditioners.speaker_wavs.sink`
   (single new row in `remap_key`; phonon keys unaffected). `bos_emb [32]` unused by the flow path (harmless).
4. **Config authored by us**: `models/pocket-tts-en/config.json` written from `english.yaml` + pocket defaults;
   every value sourced (see Config-lock table below).
5. **Fork caveat (inherited)**: pocket's own Rust binary builds `ptts` from the gabrieldemarmiesse/xn-ptts fork
   rev `7efa269…` (xn 0.2.11); our lane pins ptts 0.6.0/xn 0.2.13 upstream — parity target is pocket's PYTHON
   semantics (flow_lm.py/attention.py/tts_model.py), not its Rust binary.
6. **No ptts-0.6.0 oracle for pocket**: identity + structural equivalence replace cross-impl parity (probe shape
   adapted; phonon parity harness untouched).
7. **Family field**: TTSConfig gains explicit optional `family: Family` (`Phonon` default | `Pocket`) to gate
   voice-kind, min-frames-before-eos (6), and the `-4.0` EOS threshold semantics read from config.

## Config lock (`models/pocket-tts-en/config.json` values to write)

| field | value | source |
|---|---|---|
| flow_lm.d_model / num_heads / num_layers | 1024 / 16 / 6 | english.yaml transformer |
| flow_lm.hidden_scale → ffn | 4 → 4096 | english.yaml (hidden_scale 4) |
| flow_lm.max_period | 10000 | english.yaml |
| flow_lm.flow.kind / depth / dim | lsd / 6 / 512 | english.yaml flow (default FlowType) |
| flow_lm.ldim (z_dim) | 32 | mimi.quantizer.dimension |
| flow_lm.gelu | tanh | pocket `_ff_block` approximate="tanh" |
| flow_lm.time_rms_norm | RmsVar (default) | xn-ptts fork flow config precedent (phonon parity) |
| lsd_decode_steps | 1 | pocket DEFAULT_SAMPLER_DECODE_STEPS=1 |
| eos_threshold | -4.0 | pocket DEFAULT_EOS_THRESHOLD |
| lookup_table.n_bins / dim / tokenizer | 4000 / 1024 / tokenizers | english.yaml lookup_table |
| mimi.* | outer 512, inner(channels) per quantizer output_dimension 512, seanet ratios [6,5,4], k 7, rk 3, lk 3, dilation_base 2, pad constant, compress 2, frame_rate 12.5, sr 24000, transformer {d_model 512, heads 8, layers 2, layer_scale 0.01, context 250, ffn 2048} | english.yaml mimi |
| mimi.gelu | tanh (via default_gelu on model_id null) | pocket mimi transformer gelu |
| prefix_sinks.speaker_wavs | sink config per phonon precedent | insert_bos_before_voice: true |
| model_id | null | no hub fallback allowed |
| family | pocket | new optional field |

## Done conditions

1. [ ] `models/pocket-tts-en/config.json` authored + `loader.rs` bos→sink remap row; phonon-31m lane still passes
       smoke + parity byte-identity (regression gate).
2. [ ] `Family::Pocket` branching: voice files load as KV caches (`load_voice_kv`), `Voice` payload enum wired
       through `synth.rs` (`base_state` import path), min-frames-before-eos 6 for pocket.
3. [ ] Lane e2e: `./target/release/tts synth --lane flow-candle --model pocket-tts-en --voice alba --text "…"` →
       valid WAV (24 kHz) + telemetry row with `load_ms`, stages, `determinism`.
4. [ ] Byte-identity gate: 3× byte-identical at fixed (dir, seed, threads=1) — pocket rows; plus 3× phonon-31m
       regression unchanged.
5. [ ] hard-80 scoreboard row (nemo judge, seed 42, threads dial pinned in the run dir), verdict stated with the
       ptts-class FAIL budget law.
6. [ ] `docs/notes/` updated (new note `docs/notes/pocket.md` or a section in `phonon.md`), `targets.md` row,
       `index.md` status, scoreboard section, memory checkpoint.

## Verify list

- [ ] pocket flow_lm width law: every `flow_lm.*` tensor ≥1024 wide or explicitly checked vs config
- [ ] `cond_embed.weight [512,1024]` + `input_linear [1024,32]` + `out_eos [1,1024]` + `out_norm [1024]` shapes exact
- [ ] mimi decode path tensors only (43 zeroed encoder tensors dropped by remap (`mimi.encoder*` unused))
- [ ] voice KV import: 6 layers × [2,1,F,16,64] F32 + offset I64 accepted for all 27 voices; unknown voice → error
- [ ] text prompt: Unigram tokenizer ids + LUT n_bins 4000 (id<4000) + bos semantics (embed row 4000)
- [ ] EOS: threshold -4.0, min-frames-before-eos 6
- [ ] `bos_before_voice [1,1,1024]` lands in `speaker_wavs.sink` (loaded via PrefixSink path)
- [ ] regression: phonon-31m smoke + parity byte-identity unchanged
