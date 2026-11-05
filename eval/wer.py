#!/usr/bin/env python3
"""WER of engine transcripts vs a reference, per the session recipe.

Usage: scripts/wer.py REF.txt HYP.txt [HYP.txt ...]

REF has no stamps (e.g. outputs/reference/fw_ref_full.txt). HYP lines may carry
the engine's stamp format "[   1.12 ->    4.40]" (space-padded, 2-dp floats,
no 's' suffix); stamps are stripped before comparison.

Normalization = lowercase, unicode quotes -> ASCII, edge-strip punctuation
(contractions survive). Two scores:

  WER (repo convention)  — difflib opcodes; each non-equal block costs
    max(len(ref part), len(hyp part)). This reproduces every historical
    number in docs/benchmarks.md (te_greedy_f32_full 17.63 ≈ official 17.56,
    gguf q8_0 15.x vs raw-word 23.x which double-counts garble blocks).
  WER (raw word)         — naive estimate (ref+hyp-2*matched)/ref; reported
    for reference only, not comparable to history.
"""
import difflib
import re
import sys

STAMP = re.compile(r"^\[\s*[0-9.]+\s*->\s*[0-9.]+\]\s*(.*)$")
# Session rule: lowercase + strip punctuation. Edge-strip so contractions
# survive (didn't stays a word); unicode quotes first so ’/‘ match ASCII.
_PUNCT = "\"'()[]{}<>.,!?;:…«»–—"


def _norm_word(w: str) -> str:
    w = w.lower().translate(str.maketrans({"’": "'", "‘": "'",
                                          "“": '"', "”": '"'}))
    return w.strip(_PUNCT)


def words(path: str, strip_stamps: bool) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if strip_stamps:
                m = STAMP.match(line)
                line = m.group(1) if m else line
            out += [w for w in (_norm_word(x) for x in line.split()) if w]
    return out


def scores(ref: list[str], hyp: list[str]) -> tuple[float, float]:
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    if not ref:
        return float("nan"), float("nan")
    conv_edits = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes()
                     if t != "equal")
    matched = sum(b.size for b in sm.get_matching_blocks())
    raw = (len(ref) + len(hyp) - 2 * matched) / len(ref)
    return conv_edits / len(ref), raw


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref = words(sys.argv[1], strip_stamps=False)
    print(f"ref: {len(ref)} words ({sys.argv[1]})")
    for hyp_path in sys.argv[2:]:
        hyp = words(hyp_path, strip_stamps=True)
        conv, raw = scores(ref, hyp)
        print(f"{conv:7.2%} WER   (raw {raw:5.2%})   {len(hyp):4d} words   {hyp_path}")


if __name__ == "__main__":
    main()