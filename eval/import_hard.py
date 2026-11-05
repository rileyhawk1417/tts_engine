#!/usr/bin/env python3
"""Import the gradium/tts-hard-sentences-202610 dataset (T003 import step).

Downloads the dataset rows via the HF datasets-server API and writes the
English subset to eval/data/hard-sentences-en.jsonl (gitignored). Text-only,
CC-BY-4.0 — no weight downloads involved.

Usage: eval/import_hard.py [--lang EN] [--all]

Card/source: https://huggingface.co/datasets/gradium/tts-hard-sentences-202610
Columns kept: item_id, category, language, sentence, english.
The counts are printed and recorded in the output sidecar for the
"counts match dataset card" done-condition.
"""
import argparse
import json
import os
import sys
import urllib.request

DATASET = "gradium/tts-hard-sentences-202610"
API = f"https://datasets-server.huggingface.co"
KEEP = ["item_id", "category", "language", "sentence", "english"]


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "tts-engine-eval/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--all", action="store_true", help="keep every language")
    args = ap.parse_args()

    size = get_json(f"{API}/size?dataset={DATASET}")
    num_rows_total = size["size"]["splits"][0]["num_rows"]

    out_rows, offset = [], 0
    while True:
        page = get_json(
            f"{API}/rows?dataset={DATASET}&config=default&split=test"
            f"&offset={offset}&length=100"
        )
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            if args.all or row.get("language", "").upper() == args.lang.upper():
                out_rows.append({k: row[k] for k in KEEP})
        if len(rows) < 100:
            break
        offset += len(rows)

    os.makedirs("eval/data", exist_ok=True)
    out_path = "eval/data/hard-sentences-en.jsonl" if not args.all else \
        "eval/data/hard-sentences-all.jsonl"
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    report = {
        "dataset": DATASET,
        "dataset_card_num_rows": num_rows_total,
        "rows_kept": len(out_rows),
        "filter": {"language": None if args.all else args.lang.upper()},
        "out": out_path,
    }
    with open(out_path + ".meta.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"dataset card rows: {num_rows_total}")
    print(f"rows kept ({report['filter']['language'] or 'all'}): {len(out_rows)}")
    print(f"wrote {out_path}")
    if num_rows_total < 1000:
        print("WARNING: card reports < 1K rows — recount before trusting", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
