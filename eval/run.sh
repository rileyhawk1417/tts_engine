#!/usr/bin/env bash
# Round-trip WER eval harness (T003) — the adoption gate for any lane.
#
# Per corpus sentence: synth twice (same seed) → wav byte-identity gate → judge
# (te, default nemo int8) transcribes synthA.wav → te-repo WER (eval/wer.py,
# byte-identical copy) vs the source text → metrics from chunksA.jsonl.
#
# Usage:
#   eval/run.sh --lane smoke --model smoke                       # mechanics
#   eval/run.sh --lane <lane> --model <path-or-id> [--corpus internal|hard|<file.toml>]
#              [--limit N] [--judge nemo|parakeet] [--te PATH] [--seed S] [--tag TAG]
#
# Outputs (gitignored): eval/results/<stamp>[-<tag>]/ (rows.tsv, results.json,
# per-sentence workdirs) + one appended table row in
# eval/results/scoreboard.md.
#
# Laws kept here: identity gate must pass before WER is trusted (verdict
# "untrusted" rows are kept but flagged); TTFA/RTF/RSS come from OUR
# synthesizer's telemetry only — judge latency is te's number, never ours.
set -euo pipefail
cd "$(dirname "$0")/.."

# --out DIR resumes a prior run dir: rows with transcript.txt are skipped,
# rows.tsv is kept, and end-of-run summaries still regenerate.
LANE="" MODEL="" CORPUS="internal" LIMIT=0 JUDGE="nemo" SEED=42 TAG="" RESUME=0
TE=../transcription_engine/target/release/te
while [ $# -gt 0 ]; do
    case "$1" in
        --lane) LANE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --judge) JUDGE="$2"; shift 2 ;;
        --te) TE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --out) OUT="$2"; RESUME=1; shift 2 ;;
        *) echo "run.sh: unknown arg $1" >&2; exit 2 ;;
    esac
done
[ -n "$LANE" ] && [ -n "$MODEL" ] || { echo "run.sh: --lane and --model required" >&2; exit 2; }

TTS=target/release/tts
# lane features live behind cargo features: graph-ort (kokoro), flow-ptts
# (Phonon), flow-candle (T008 candle port), piper (T011 fused VITS). ALL
# FOUR on every rebuild — a feature-stripped relink silently drops lanes
# (the binary keeps its size, mtime lies after the hardlink relink, so
# size/mtime are NOT staleness signals; only the feature list is truth).
# Run flow-ptts rows with TTS_XN_THREADS=1 so byte-identity gates
# pass (xn CPU FP kernels are order-nondeterministic under parallelism).
[ -x "$TTS" ] || cargo build --release --features graph-ort,flow-ptts,flow-candle,piper -q
cargo build --release --features graph-ort,flow-ptts,flow-candle,piper -q
if [ "$LANE" = flow-ptts ]; then
    export TTS_XN_THREADS="${TTS_XN_THREADS:-1}"
fi
[ -x "$TE" ] || { echo "run.sh: judge binary missing: $TE (build te with TE_FEATURES=nemo)" >&2; exit 3; }
TE_SHA16=$(sha256sum "$TE" | cut -c1-16)
case "$JUDGE" in
    nemo) TE_FLAG="--nemo"; JUDGE_ID="nemo-int8 (te @${TE_SHA16})" ;;
    parakeet)
        "$TE" transcribe --help 2>/dev/null | grep -q -- --gguf \
            || { echo "run.sh: judge parakeet needs te built with TE_FEATURES=parakeet" >&2; exit 3; }
        TE_FLAG="--gguf tdt-0.6b-v2"  # q8_0 default quant per te
        export TE_PARAKEET_LIB="${TE_PARAKEET_LIB:-/teranux/parakeet.cpp/build/libparakeet.so}"
        JUDGE_ID="parakeet-q8_0 (te @${TE_SHA16})" ;;
    *) echo "run.sh: unknown judge '$JUDGE'" >&2; exit 2 ;;
esac

# corpus dir-name: bare stem for file corpora (no slashes in the stamp)
CORPUS_NAME=$(basename "$CORPUS" .toml)
if [ "$RESUME" != 1 ]; then
    STAMP=$(date +%Y%m%d-%H%M)-${LANE}-${CORPUS_NAME}${TAG:+-$TAG}
    OUT=eval/results/$STAMP
fi
mkdir -p "$OUT"
# a usable resume target must already carry a rows.tsv
if [ "$RESUME" = 1 ] && [ ! -f "$OUT/rows.tsv" ]; then
    echo "run.sh: --out '$OUT' has no rows.tsv — starting fresh" >&2
    RESUME=0
fi

# --- corpus loader: prints id<TAB>lang<TAB>text -------------------------------
if [ "$CORPUS" = hard ]; then
    [ -f eval/data/hard-sentences-en.jsonl ] || python3 eval/import_hard.py
    SENTENCES=$(python3 - <<'PY'
import json
with open("eval/data/hard-sentences-en.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print(f'{r["item_id"]}\tEN\t{r["sentence"]}')
PY
)
else
    CORPUS_FILE="eval/corpus.toml"
    [ "$CORPUS" = internal ] || CORPUS_FILE="$CORPUS"
    SENTENCES=$(python3 - "$CORPUS_FILE" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    doc = tomllib.load(f)
for s in doc["sentence"]:
    print(f'{s["id"]}\t{doc["meta"]["lang"]}\t{s["text"]}')
PY
)
fi

SENT_N=$(printf '%s\n' "$SENTENCES" | wc -l)
TAKE=$SENT_N
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$SENT_N" ] && TAKE=$LIMIT

if [ "$RESUME" != 1 ]; then
    : > "$OUT/rows.tsv"
    printf 'id\tlang\twer\traw_wer\tttfa_ms\trtf_final\tpeak_rss_kb_max\taudio_s\twall_ms\tidentity\n' >> "$OUT/rows.tsv"
fi

# metrics: "ttfa_ms rtf_final peak_rss_kb_max audio_s wall_ms" from summary row
summary_metrics() {  # $1 = chunks.jsonl
    python3 - "$1" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    s = json.loads(f.read().splitlines()[-1])
assert s["type"] == "summary", "last row must be summary"
print(s["ttfa_ms"], s["rtf_final"], s["peak_rss_kb_max"], s["audio_seconds"], s["wall_ms"])
PY
}

printf '%s\n' "$SENTENCES" | head -n "$TAKE" | while IFS=$'\t' read -r ID LANG_ TEXT; do
    # resume: a prior row dir for this ID with a transcript is done - skip it
    EXIST=$(find "$OUT" -maxdepth 1 -name "*-$ID" -print -quit 2>/dev/null || true)
    if [ -n "$EXIST" ] && [ -f "$EXIST/transcript.txt" ]; then
        echo "[$(wc -l < "$OUT/rows.tsv")] $ID skip (resume)"
        continue
    fi

    n=$(( $(wc -l < "$OUT/rows.tsv") ))
    WD="$OUT/$(printf '%02d' $n)-$ID"
    mkdir -p "$WD"
    printf '%s\n' "$TEXT" > "$WD/ref.txt"
    echo "[$n] $ID"

    "$TTS" synth --lane "$LANE" --model "$MODEL" --lang "$LANG_" --text "$TEXT" \
        --seed "$SEED" -o "$WD/synthA.wav" --jsonl "$WD/chunksA.jsonl" 2> "$WD/synth.log"
    "$TTS" synth --lane "$LANE" --model "$MODEL" --lang "$LANG_" --text "$TEXT" \
        --seed "$SEED" -o "$WD/synthB.wav" --jsonl "$WD/chunksB.jsonl" 2>> "$WD/synth.log"

    if cmp -s "$WD/synthA.wav" "$WD/synthB.wav"; then IDENTITY=pass; else IDENTITY=FAIL; fi

    "$TE" transcribe $TE_FLAG --out-dir "$WD" "$WD/synthA.wav" > "$WD/te.log" 2>&1 \
        || echo "judge failed (see $WD/te.log)" >&2
    WER_VAL=nan RAW=nan
    if [ -f "$WD/transcript.txt" ]; then
        LINE=$(python3 eval/wer.py "$WD/ref.txt" "$WD/transcript.txt" \
            | sed -nE 's/^ *([0-9.]+)% WER.*\(raw ([0-9.]+)%\).*/\1 \2/p' || true)
        [ -n "$LINE" ] && { WER_VAL=${LINE%% *}; RAW=${LINE##* }; }
    else
        echo "no transcript produced" >&2
    fi

    read -r TTFA RTF RSS AUDIO WALL < <(summary_metrics "$WD/chunksA.jsonl")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ID" "$LANG_" "$WER_VAL" "$RAW" "$TTFA" "$RTF" "$RSS" "$AUDIO" "$WALL" "$IDENTITY" >> "$OUT/rows.tsv"
    echo "    WER=$WER_VAL identity=$IDENTITY"
done

N_PROC=$(($(wc -l < "$OUT/rows.tsv") - 1))

# --- aggregate: results.json + scoreboard.md append ---------------------------
python3 - "$OUT" "$LANE" "$MODEL" "$CORPUS_NAME" "$JUDGE_ID" "$SEED" <<'PY'
import json, math, os, sys
out_dir, lane, model, corpus, judge, seed = sys.argv[1:7]
rows = []
with open(f"{out_dir}/rows.tsv") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        vals = line.rstrip("\n").split("\t")
        r = dict(zip(header, vals))
        for k in ("wer", "raw_wer", "ttfa_ms", "rtf_final", "peak_rss_kb_max", "audio_s", "wall_ms"):
            r[k] = float(r[k]) if k in r and r[k] else None
        rows.append(r)

def stat(vals, q=None):
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    return sum(xs) / len(xs) if q is None else xs[math.ceil(q * len(xs)) - 1]

identity = bool(rows) and all(r["identity"] == "pass" for r in rows)
wers = [r["wer"] for r in rows]
result = {
    "lane": lane, "model": model, "corpus": corpus, "judge": judge, "seed": int(seed),
    "sentences": len(rows), "WER_mean": stat(wers), "WER_max": max(wers, default=None),
    "ttfa_ms_mean": stat([r["ttfa_ms"] for r in rows]),
    "ttfa_ms_p95": stat([r["ttfa_ms"] for r in rows], q=0.95),
    "rtf_mean": stat([r["rtf_final"] for r in rows]),
    "peak_rss_kb_max": max((r["peak_rss_kb_max"] for r in rows), default=None),
    "byte_identity": "pass" if identity else "FAIL",
    "verdict": "trusted" if identity else "untrusted (identity gate failed)",
    "rows": rows,
}
with open(f"{out_dir}/results.json", "w") as f:
    json.dump(result, f, indent=2)

board = "eval/results/scoreboard.md"
if not os.path.exists(board):
    with open(board, "w") as f:
        f.write("| date | lane | model | corpus | n | judge | WER mean % | TTFA mean/p95 ms | RTF mean | RSS max KB | identity |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
fmt = lambda v, spec: "-" if v is None else format(v, spec)
with open(board, "a") as f:
    f.write("| {} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} | {} |\n".format(
        out_dir.split("/")[-1], lane, model, corpus, len(rows), judge,
        fmt(result["WER_mean"], ".2f"), fmt(result["ttfa_ms_mean"], ".0f"),
        fmt(result["ttfa_ms_p95"], ".0f"), fmt(result["rtf_mean"], ".3f"),
        fmt(result["peak_rss_kb_max"], ".0f"), result["byte_identity"]))
print(f"[eval] WER mean {fmt(result['WER_mean'], '.2f')}% | verdict: {result['verdict']}")
print(f"[eval] results in {out_dir}; scoreboard row appended")
PY
