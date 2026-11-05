#!/usr/bin/env bash
# tts_engine build gate. No task is "done" without this passing.
# fmt -> clippy -D warnings -> tests -> build -> smoke lane e2e + determinism
#      (+ kokoro lane e2e + determinism when model artifacts are present).
set -euo pipefail
cd "$(dirname "$0")/.."

# The kokoro (ORT), flow-ptts (Phonon/xn), flow-candle (T008 port), and
# piper (T011 fused VITS) lanes live behind cargo features; run every cargo
# step with them so gate = shipped reality.
cargo fmt --all
cargo clippy --workspace --all-targets --features graph-ort,flow-ptts,flow-candle,piper -- -D warnings
cargo test --workspace --features graph-ort,flow-ptts,flow-candle,piper --quiet
cargo build --workspace --features graph-ort,flow-ptts,flow-candle,piper -q

# e2e runs the release binary — the shipped artifact (xn debug build is ~8x slower)
cargo build --workspace --release --features graph-ort,flow-ptts,flow-candle,piper -q

mkdir -p out
TEXT="Hello from the seam. The smoke lane writes honest silence and honest telemetry, one row per chunk."

cargo run -q -p tts -- synth --lane smoke --model smoke --lang en \
  --text "$TEXT" -o out/smoke.wav --jsonl out/smoke/chunks.jsonl --seed 42

# Determinism check: same seed, second run, PCM must be bit-identical.
cargo run -q -p tts -- synth --lane smoke --model smoke --lang en \
  --text "$TEXT" -o out/smoke2.wav --jsonl out/smoke/chunks2.jsonl --seed 42

python3 - <<'EOF'
import json

def check(fa, fb):
    for f in (fa, fb):
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
        rows.pop()  # summary

    def pcm(f):
        rows = [json.loads(l) for l in open(f)]
        return [r["pcm_bytes"] for r in rows if r["type"] == "chunk"]

    a, b = pcm(fa), pcm(fb)
    assert a == b and a, "byte-identity gate failed"
    return len(a)

n = check("out/smoke/chunks.jsonl", "out/smoke/chunks2.jsonl")
print("smoke telemetry OK:", n, "chunks; pcm byte-identity pass")

try:
    n = check("out/kokoro/chunks.jsonl", "out/kokoro/chunks2.jsonl")
    print("kokoro telemetry OK:", n, "chunks; pcm byte-identity pass")
except FileNotFoundError:
    print("kokoro e2e skipped: no out/kokoro telemetry (run it or see below)")
EOF

# Kokoro (graph-ort) e2e — skipped on machines without the artifacts.
if [[ -f models/kokoro-82m-v1.0-onnx/model_quantized.onnx ]] && command -v espeak-ng >/dev/null; then
  KTEXT="Hello from the Kokoro lane."
  target/release/tts synth --lane kokoro \
    --model models/kokoro-82m-v1.0-onnx/model_quantized.onnx --lang en \
    --text "$KTEXT" -o out/kokoro.wav --jsonl out/kokoro/chunks.jsonl --seed 42
  target/release/tts synth --lane kokoro \
    --model models/kokoro-82m-v1.0-onnx/model_quantized.onnx --lang en \
    --text "$KTEXT" -o out/kokoro2.wav --jsonl out/kokoro/chunks2.jsonl --seed 42
  python3 - <<'EOF'
import json

for f in ("out/kokoro/chunks.jsonl", "out/kokoro/chunks2.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    assert rows[0]["lane"] == "kokoro", rows[0]["lane"]
    assert rows[0]["sample_rate"] == 24000
    assert rows[0]["determinism"] == "deterministic"
def pcm(f):
    rows = [json.loads(l) for l in open(f)]
    return [r["pcm_bytes"] for r in rows if r["type"] == "chunk"]

a, b = pcm("out/kokoro/chunks.jsonl"), pcm("out/kokoro/chunks2.jsonl")
assert a == b and a, "byte-identity gate failed on kokoro determinism"
# audio sanity from the WAV itself (telemetry only carries sha256, by contract)
import wave, audioop
with wave.open("out/kokoro.wav", "rb") as w:
    assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    n, frames = w.getnframes(), w.readframes(w.getnframes())
assert n >= 12000, f"audio shorter than 0.5 s: {n} frames"
peak = max(abs(int.from_bytes(frames[i:i+2], 'little', signed=True)) for i in range(0, len(frames) - 1, 4096))
assert peak > 1000, f"audio is near-silence (peak {peak})"
print("kokoro e2e OK:", len(a), "chunks; pcm byte-identity pass; audio", round(n/24000, 3), "s peak", peak)
EOF
else
  echo "kokoro e2e skipped (missing model artifacts or espeak-ng)"
fi

# Phonon (flow-ptts) e2e — xn CPU FP kernels are order-nondeterministic under
# rayon parallelism, so the byte-identity gate runs at TTS_XN_THREADS=1 (the
# lane reports `seeded` there; above 1 it reports nondet with that reason).
if [[ -f models/phonon-31m-en/model.q8.gguf ]]; then
  PTEXT="The engine loads its weights straight from disk."
  TTS_XN_THREADS=1 target/release/tts synth --lane flow-ptts \
    --model models/phonon-31m-en --lang en --seed 42 \
    --text "$PTEXT" -o out/flow-ptts.wav --jsonl out/flow-ptts/chunks.jsonl
  TTS_XN_THREADS=1 target/release/tts synth --lane flow-ptts \
    --model models/phonon-31m-en --lang en --seed 42 \
    --text "$PTEXT" -o out/flow-ptts2.wav --jsonl out/flow-ptts/chunks2.jsonl
  python3 - <<'EOF2'
import json

for f in ("out/flow-ptts/chunks.jsonl", "out/flow-ptts/chunks2.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    assert rows[0]["lane"] == "flow-ptts", rows[0]["lane"]
    assert rows[0]["sample_rate"] == 24000
    assert rows[0]["determinism"] == "seeded", rows[0]["determinism"]
def pcm(f):
    rows = [json.loads(l) for l in open(f)]
    return [r["pcm_bytes"] for r in rows if r["type"] == "chunk"]

a, b = pcm("out/flow-ptts/chunks.jsonl"), pcm("out/flow-ptts/chunks2.jsonl")
assert a == b and a, "byte-identity gate failed on flow-ptts determinism"
# audio sanity from the WAV itself (telemetry only carries sha256, by contract)
import wave
with wave.open("out/flow-ptts.wav", "rb") as w:
    assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    n, frames = w.getnframes(), w.readframes(w.getnframes())
assert n >= 12000, f"audio shorter than 0.5 s: {n} frames"
peak = max(abs(int.from_bytes(frames[i:i+2], 'little', signed=True)) for i in range(0, len(frames) - 1, 4096))
assert peak > 1000, f"audio is near-silence (peak {peak})"
print("flow-ptts e2e OK:", len(a), "chunks; pcm byte-identity pass (TTS_XN_THREADS=1); audio", round(n/24000, 3), "s peak", peak)
EOF2
else
  echo "flow-ptts e2e skipped (missing models/phonon-31m-en artifacts)"
fi

echo "build.sh OK"
