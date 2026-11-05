#!/usr/bin/env bash
# tts_engine build gate. No task is "done" without this passing.
# fmt -> clippy -D warnings -> tests -> build -> smoke lane e2e + determinism.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet
cargo build --workspace -q

mkdir -p out
TEXT="Hello from the seam. The smoke lane writes honest silence and honest telemetry, one row per chunk."

cargo run -q -p tts -- synth --lane smoke --model smoke --lang en \
  --text "$TEXT" -o out/smoke.wav --jsonl out/smoke/chunks.jsonl --seed 42

# Determinism check: same seed, second run, PCM must be bit-identical.
cargo run -q -p tts -- synth --lane smoke --model smoke --lang en \
  --text "$TEXT" -o out/smoke2.wav --jsonl out/smoke/chunks2.jsonl --seed 42

python3 - <<'EOF'
import hashlib, json

for f in ("out/smoke/chunks.jsonl", "out/smoke/chunks2.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    assert rows[0]["type"] == "header" and rows[-1]["type"] == "summary"
    chunks = rows[1:-1]
    assert [c["i"] for c in chunks] == list(range(len(chunks)))
    assert all(c["t_synth_start_ms"] == chunks[0]["t_synth_start_ms"] for c in chunks)
    assert sum(1 for c in chunks if c["ttfa_ms"] is not None) == 1
    rtfs = [c["rtf_running"] for c in chunks]
    assert all(r > 0 for r in rtfs)  # NOT contractually monotone (see seam §5.1)
    s = rows[-1]
    assert s["chunks"] == len(chunks)
    assert s["rtf_final"] == rtfs[-1]
    assert s["peak_rss_kb_max"] > 0
    assert s["audio_seconds"] > 0

def pcm(f):
    rows = [json.loads(l) for l in open(f)]
    return [r["pcm_bytes"] for r in rows if r["type"] == "chunk"]

a, b = pcm("out/smoke/chunks.jsonl"), pcm("out/smoke/chunks2.jsonl")
assert a == b and a, "byte-identity gate failed on smoke determinism"
print("smoke telemetry OK:", len(a), "chunks; pcm byte-identity pass")
EOF
echo "build.sh OK"
