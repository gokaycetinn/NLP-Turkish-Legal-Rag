"""
Merge per-author dev JSONLs into a single gold set with sanity checks.

Reads `dev_*.jsonl` files from data/test_set/, validates schema, checks for
duplicates and broken JSON, verifies gold_doc_ids against the corpus, and
produces a stratified 80/20 dev/test split.

Outputs:
  - dev_full.jsonl     — training-time eval (~80%)
  - test_full.jsonl    — held-out final test (~20%)
  - merge_report.json  — author/difficulty/category breakdown + warnings

Usage:
    python -m src.eval.merge_gold_set
    python -m src.eval.merge_gold_set --in-dir data/test_set \\
        --out-dev data/test_set/dev_full.jsonl \\
        --out-test data/test_set/test_full.jsonl \\
        --meta data/index_full/meta.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger("merge_gold_set")

REQUIRED_FIELDS = ["id", "question", "answer", "gold_doc_ids",
                   "gold_citations", "difficulty", "category", "author"]
VALID_DIFFICULTIES = {"lookup", "reasoning", "edge", "multi_hop", "no_answer"}


def load_corpus_doc_ids(meta_path: Path) -> set[str]:
    ids = set()
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def validate_row(row: dict, line_no: int, source: str) -> tuple[bool, list[str]]:
    errs = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            errs.append(f"missing field '{field}'")
    if not errs:
        is_no_answer = row.get("difficulty") == "no_answer"
        if not isinstance(row["gold_doc_ids"], list):
            errs.append("gold_doc_ids must be list")
        elif not row["gold_doc_ids"] and not is_no_answer:
            errs.append("gold_doc_ids must be non-empty (unless difficulty=no_answer)")
        if not isinstance(row["gold_citations"], list):
            errs.append("gold_citations must be list")
        if row.get("difficulty") not in VALID_DIFFICULTIES:
            errs.append(f"invalid difficulty: {row.get('difficulty')!r}")
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            errs.append("id must be non-empty string")
        if not isinstance(row.get("question"), str) or len(row["question"]) < 5:
            errs.append("question must be string of len ≥ 5")
        if not isinstance(row.get("answer"), str) or len(row["answer"]) < 5:
            errs.append("answer must be string of len ≥ 5")
    return (len(errs) == 0, errs)


def read_dev_file(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, errors). Each error: {line, source, content, errs}."""
    rows, errors = [], []
    with path.open(encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                errors.append({
                    "source": path.name, "line": ln, "type": "json_parse",
                    "content": stripped[:100], "errs": [str(e)],
                })
                continue
            ok, errs = validate_row(row, ln, path.name)
            if not ok:
                errors.append({
                    "source": path.name, "line": ln, "type": "schema",
                    "id": row.get("id"), "errs": errs,
                })
                continue
            row["_source_file"] = path.name
            rows.append(row)
    return rows, errors


def stratified_split(rows: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split stratified by difficulty."""
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    rng = random.Random(seed)
    test, dev = [], []
    for d, group in by_diff.items():
        shuf = group.copy()
        rng.shuffle(shuf)
        n_test = max(1, int(len(shuf) * test_ratio)) if len(shuf) >= 5 else 0
        test.extend(shuf[:n_test])
        dev.extend(shuf[n_test:])
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/test_set")
    ap.add_argument("--pattern", default="dev_*.jsonl",
                    help="Glob pattern; excludes 'dev_full.jsonl' itself.")
    ap.add_argument("--out-dev", default="data/test_set/dev_full.jsonl")
    ap.add_argument("--out-test", default="data/test_set/test_full.jsonl")
    ap.add_argument("--out-report", default="data/test_set/merge_report.json")
    ap.add_argument("--meta", default="data/index_full/meta.jsonl",
                    help="Corpus meta.jsonl for gold_doc_ids validation.")
    ap.add_argument("--test-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = repo_root / args.in_dir
    out_dev = repo_root / args.out_dev
    out_test = repo_root / args.out_test
    out_report = repo_root / args.out_report
    meta_path = repo_root / args.meta

    # Load corpus doc_ids
    log.info("Loading corpus doc_ids from %s", meta_path)
    corpus_ids = load_corpus_doc_ids(meta_path)
    log.info("Corpus: %d unique doc_ids", len(corpus_ids))

    # Read all dev files (exclude any pre-existing full/test files)
    files = sorted(p for p in in_dir.glob(args.pattern)
                   if p.name not in {"dev_full.jsonl", "test_full.jsonl"})
    log.info("Found %d dev files: %s", len(files), [p.name for p in files])

    all_rows, all_errors = [], []
    per_file_stats = {}
    for p in files:
        rows, errors = read_dev_file(p)
        all_rows.extend(rows)
        all_errors.extend(errors)
        per_file_stats[p.name] = {"valid_rows": len(rows), "errors": len(errors)}
        log.info("  %s: %d valid, %d error(s)", p.name, len(rows), len(errors))

    # Duplicates
    id_counts = Counter(r["id"] for r in all_rows)
    dup_ids = {i for i, c in id_counts.items() if c > 1}
    question_counts = Counter(r["question"].strip().lower() for r in all_rows)
    dup_questions = {q for q, c in question_counts.items() if c > 1}

    if dup_ids:
        log.warning("Duplicate id'ler bulundu: %d (örnek: %s)",
                    len(dup_ids), list(dup_ids)[:5])
    if dup_questions:
        log.warning("Duplicate question'lar: %d", len(dup_questions))

    # Deduplicate: keep first occurrence of each id
    seen_ids: set[str] = set()
    unique_rows = []
    for r in all_rows:
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        unique_rows.append(r)
    log.info("Unique row count (by id): %d", len(unique_rows))

    # gold_doc_ids validity
    missing_doc_rows = []
    for r in unique_rows:
        missing = [d for d in r["gold_doc_ids"] if d not in corpus_ids]
        if missing:
            missing_doc_rows.append({"id": r["id"], "missing": missing,
                                      "source": r.get("_source_file")})
    if missing_doc_rows:
        log.warning("%d rows have gold_doc_ids not in corpus", len(missing_doc_rows))

    # Stratified split: keep rows with at-least-one-valid gold_doc_id OR no_answer rows
    splittable = [r for r in unique_rows
                  if r["difficulty"] == "no_answer"
                  or any(d in corpus_ids for d in r["gold_doc_ids"])]
    log.info("Splittable rows: %d (incl no_answer, dropping only rows with 0 valid gold docs)",
             len(splittable))

    dev_rows, test_rows = stratified_split(splittable, args.test_ratio, args.seed)
    log.info("Split → dev %d / test %d", len(dev_rows), len(test_rows))

    # Strip internal marker before writing
    def clean(r):
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        return out

    out_dev.parent.mkdir(parents=True, exist_ok=True)
    with out_dev.open("w", encoding="utf-8") as f:
        for r in dev_rows:
            f.write(json.dumps(clean(r), ensure_ascii=False) + "\n")
    with out_test.open("w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(clean(r), ensure_ascii=False) + "\n")
    log.info("Wrote %s, %s", out_dev, out_test)

    # Stats
    author_counts = Counter(r["author"] for r in unique_rows)
    diff_counts = Counter(r["difficulty"] for r in unique_rows)
    cat_counts = Counter(r["category"] for r in unique_rows)

    report = {
        "summary": {
            "total_raw": sum(s["valid_rows"] for s in per_file_stats.values()),
            "total_errors": len(all_errors),
            "unique_by_id": len(unique_rows),
            "duplicate_ids": len(dup_ids),
            "duplicate_questions": len(dup_questions),
            "rows_with_missing_doc_ids": len(missing_doc_rows),
            "splittable": len(splittable),
            "dev_size": len(dev_rows),
            "test_size": len(test_rows),
        },
        "per_file": per_file_stats,
        "by_author": dict(author_counts.most_common()),
        "by_difficulty": dict(diff_counts.most_common()),
        "by_category": dict(cat_counts.most_common()),
        "errors": all_errors[:50],  # cap report size
        "duplicate_id_samples": list(dup_ids)[:20],
        "missing_doc_samples": missing_doc_rows[:20],
    }
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=== RAPOR ===")
    log.info("Yazara göre: %s", dict(author_counts.most_common()))
    log.info("Difficulty: %s", dict(diff_counts.most_common()))
    log.info("Category: %s", dict(cat_counts.most_common()))
    log.info("Toplam hata: %d (parse + schema)", len(all_errors))
    log.info("Detaylı rapor: %s", out_report)


if __name__ == "__main__":
    main()
