# T013 — Long-text chunking module

- **Status:** `completed` (landed with T002 scaffold, 2026-11-05)
- **Milestone:** [M0](../index.md#milestone-m0--foundation)
- **Depends on:** [T002](T002-workspace-scaffold.md)
- **Deliverable:** `crates/tts-engine/src/chunking.rs` (14 unit tests) + seam appendix

## Goal

Shared, unit-tested text chunking (pause/sentence-aware) that all lanes consume — the seam-side analogue of learned chunking policies, so lanes only override where their model demands it.

## Approach

- `crates/tts-engine/src/chunking.rs`: sentence split → clause split → hard-split (long word/hash runs), configurable target tokens-per-chunk (default ~50, Phonon law) and max-chunk.
- Port rules: Phonon ~50-token sentence chunks, >200-token split at clause marks, single word cut between characters; te law: chunk at pauses, smaller chunks faster on CPU. Piper lane later adds overlap-add concerns at decoder level only.
- Language-aware sentence boundaries (en, fr, de, es, pt at minimum — targets table languages).
- Lanes receive `Vec<Chunk>` with per-chunk voice/prompt restart semantics declared by the lane; seam logs per-chunk stage timings under the existing chunks.jsonl schema.
- Property tests: idempotence on already-chunked input, no empty chunks, total text preserved (modulo whitespace policy).

## Done-conditions

- [x] Module merged with 14 tests green (sentences/abbreviations/paragraphs/clauses/char-cut/idempotence/byte-ranges/join); CLI chunks multi-sentence input seam-side before lane hand-off
- [x] Chunk policy documented in `docs/seam.md` (appendix) including per-lane override hooks

## Notes from build

- Token approximation is word-count based (seam level); exact subword counts are lane-local and may re-split WITHIN seam spans (T006 Phonon).
- One word > `max_tokens` chars is char-cut (last resort) only when the packing path runs; a lone unpunctuated monster word is ONE seam chunk (pause-boundary law) — `single_monster_word_is_one_seam_chunk` pins this.
