#!/usr/bin/env bash
# Thread-dial sweep (chunk-1 speed session) — ORT lanes.
# Per setting: fresh process, TTS_ORT_INTRA=T, seed 42, --repeat 3.
#   - 3 "synth ok" rows -> wall/rtf/ttfa/rss per rep (rep0 = coldish w/ FS caches)
#   - sha256 per rep wav: in-process 3x identity + cross-setting byte compare
# Usage: sweep_ort.sh <lane> <model-path> [voice]
set -euo pipefail
cd "$(dirname "$0")/.."
LANE="$1"; MODEL="$2"; VOICE="${3:-}"
STAMP=$(date +%m%d-%H%M%S)
OUTROOT="/tmp/threads-sweep/$STAMP-$LANE"
mkdir -p "$OUTROOT"
BIN=target/release/tts
# Fixed sweep text: 3 internal-corpus sentences (greeting/contraction/numbers-dates).
TEXT="Hello, this is a smoke test of the speech engine. Isn't it strange how quickly the weather changed this morning? Route 47 leaves at 3:45 pm and arrives around 8:10 in the evening."
VOICE_ARG=(); [ -n "$VOICE" ] && VOICE_ARG=(--voice "$VOICE")
echo "sweep: lane=$LANE out=$OUTROOT load=$(cut -d' ' -f1-3 /proc/loadavg)"
for T in 1 2 4 8 16; do
    echo "=== TTS_ORT_INTRA=$T ==="
    TTS_ORT_INTRA=$T "$BIN" synth --lane "$LANE" --model "$MODEL" "${VOICE_ARG[@]+"${VOICE_ARG[@]}"}" \
        --text "$TEXT" -o "$OUTROOT/t$T.wav" --seed 42 --repeat 3 >/dev/null 2> "$OUTROOT/t$T.log"
    grep "synth ok" "$OUTROOT/t$T.log" | sed -E 's/.*chunks=([0-9]+) audio=([0-9.]+)s wall=([0-9]+)ms rtf=([0-9.]+) ttfa=([0-9]+)ms peak_rss=([0-9]+)KB.*/  chunks=\1 audio=\2 wall=\3ms rtf=\4 ttfa=\5ms rss=\6KB/'
    for f in "$OUTROOT/t$T".wav.rep*; do
        echo "  sha: $(sha256sum "$f" | cut -c1-16)  $(basename "$f")"
    done
    u=$(sha256sum "$OUTROOT/t$T".wav.rep* | awk '{print $1}' | sort -u | wc -l)
    echo "  unique shas across reps: $u (1 = identity PASS)"
done
