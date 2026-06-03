"""
Normalize kanun_short field in the full scraped corpus.

The scraper assigns generic IDs ("T1_5237", "T1_4721") to kanun_short for all
documents, because the active list from mevzuat.gov.tr only provides full names
(mevAdi), not short codes.

For the 16 priority kanunlar (defined in mevzuat_targets.PRIORITY_KANUNLAR) we
have curated short names ("TCK", "TMK", ...). This script rewrites their
kanun_short + doc_id to use the readable short form so:
  - sentetik QA / gold test set citation matching ([TCK m.299]) works
  - eval harness retrieves with consistent doc_id

Non-priority kanunlar keep their generic ID (no acronym collision risk).

Output: data/corpus/mevzuat_full_normalized.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from src.ingest.mevzuat_targets import PRIORITY_KANUNLAR

log = logging.getLogger("normalize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus/mevzuat_full.jsonl")
    ap.add_argument("--output", default="data/corpus/mevzuat_full_normalized.jsonl")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    in_path = repo_root / args.input
    out_path = repo_root / args.output

    # mevzuat_no -> short_name lookup (16 priority kanunlar)
    short_by_no = {entry[0]: entry[3] for entry in PRIORITY_KANUNLAR}
    log.info("Priority mapping loaded: %d kanunlar", len(short_by_no))

    total = 0
    normalized = 0
    short_counts: Counter = Counter()
    missing_kanun_no = 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            no = rec.get("kanun_no")
            if no is None:
                missing_kanun_no += 1
            elif no in short_by_no:
                new_short = short_by_no[no]
                old_short = rec.get("kanun_short", "")
                rec["kanun_short"] = new_short
                # Update doc_id prefix (only first occurrence — madde suffix preserved)
                if old_short and rec.get("doc_id", "").startswith(old_short + "_"):
                    rec["doc_id"] = new_short + rec["doc_id"][len(old_short):]
                normalized += 1
            short_counts[rec.get("kanun_short", "")] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("=== STATS ===")
    log.info("Toplam satır: %d", total)
    log.info("Normalize edilen (priority): %d", normalized)
    log.info("kanun_no eksik kayıt: %d", missing_kanun_no)
    log.info("Generic ID'de kalan satır: %d", total - normalized - missing_kanun_no)

    priority_shorts = [entry[3] for entry in PRIORITY_KANUNLAR]
    log.info("\nPriority kanun madde sayıları:")
    for p in priority_shorts:
        log.info("  %-8s %d", p, short_counts.get(p, 0))

    log.info("\nOutput: %s", out_path)


if __name__ == "__main__":
    main()
