"""
Aggregate 8-cell ablation results into report-ready tables.

Reads:
  results/{A1,A2,A3,A4,A5,A5a,A5b,A5c}/per_example.jsonl   (from run_eval)
  results/<ablation>/with_judge.jsonl                       (from judge_run, optional)

Writes:
  results/ablation_table.csv      Main results table (rapor §7.1, slide 12)
  results/per_category.csv        Per-category breakdown (rapor §7.4, slide B5)
  results/isolations.csv          A5 − A5a/b/c, A4 − A3 deltas (rapor §7.2, slide 13)
  results/REPORT_TABLES.md        Markdown copy-paste blocks for REPORT.md
  results/SLIDES_TABLES.md        Plain rows for SLIDES.pptx hand-edit

Usage:
    python -m src.eval.aggregate_results --results results/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

log = logging.getLogger("aggregate")

ABLATIONS = ["A1", "A2", "A3", "A4", "A5", "A5a", "A5b", "A5c"]

METRICS = [
    ("recall@5",   "R@5"),
    ("recall@10",  "R@10"),
    ("mrr@10",     "MRR@10"),
    ("ndcg@10",    "nDCG@10"),
    ("f1",         "F1"),
    ("rouge_l",    "ROUGE-L"),
    ("bleu",       "BLEU"),
    ("em",         "EM"),
    ("cit_p",      "Cit-P"),
    ("cit_r",      "Cit-R"),
    ("cit_f1",     "Cit-F1"),
    ("faithfulness", "Faith"),
]

# (compare A, compare B, label)
ISOLATIONS = [
    ("A5", "A5a", "LLM SFT"),
    ("A5", "A5b", "Embedding FT"),
    ("A5", "A5c", "Reranker FT"),
    ("A4", "A3",  "Reranker FT (2nd, vanilla LLM)"),
]


def _load_per_example(ablation_dir: Path) -> list[dict]:
    """Load with_judge.jsonl if present (has faithfulness), else per_example.jsonl."""
    wj = ablation_dir / "with_judge.jsonl"
    pe = ablation_dir / "per_example.jsonl"
    if wj.exists():
        path = wj
    elif pe.exists():
        path = pe
    else:
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_mean(values):
    nums = [v for v in values if isinstance(v, (int, float)) and v == v]  # filter None and NaN
    return mean(nums) if nums else None


def _macro_metrics(rows: list[dict]) -> dict:
    out = {}
    for key, _ in METRICS:
        out[key] = _safe_mean([r.get(key) for r in rows])
    out["n"] = len(rows)
    return out


def aggregate_main(results_root: Path) -> dict[str, dict]:
    """ablation → {metric: value}"""
    table = {}
    for ab in ABLATIONS:
        rows = _load_per_example(results_root / ab)
        if not rows:
            log.warning("Missing: %s", ab)
            table[ab] = {"n": 0}
            continue
        table[ab] = _macro_metrics(rows)
        log.info("%s: n=%d  R@10=%.3f  Cit-F1=%.3f  Faith=%s",
                 ab, table[ab]["n"],
                 table[ab].get("recall@10") or 0,
                 table[ab].get("cit_f1") or 0,
                 f"{table[ab]['faithfulness']:.3f}" if table[ab].get("faithfulness") is not None else "—")
    return table


def aggregate_per_category(results_root: Path) -> dict[tuple[str, str], dict]:
    """(ablation, category) → metrics"""
    out = {}
    for ab in ABLATIONS:
        rows = _load_per_example(results_root / ab)
        if not rows:
            continue
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            cat = r.get("category") or "unknown"
            groups[cat].append(r)
        for cat, grp in groups.items():
            out[(ab, cat)] = _macro_metrics(grp)
    return out


def write_csv(path: Path, headers: list[str], rows: list[list]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    log.info("Wrote %s", path)


def fmt_num(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Root results directory")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(args.results)
    if not root.exists():
        raise SystemExit(f"Results dir not found: {root}")

    # === Main table ===
    main_table = aggregate_main(root)
    headers = ["ablation", "n"] + [k for k, _ in METRICS]
    rows = []
    for ab in ABLATIONS:
        m = main_table.get(ab, {})
        rows.append([ab, m.get("n", 0)] + [fmt_num(m.get(k)) for k, _ in METRICS])
    write_csv(root / "ablation_table.csv", headers, rows)

    # === Isolations ===
    iso_rows = []
    for a, b, label in ISOLATIONS:
        ma, mb = main_table.get(a, {}), main_table.get(b, {})
        deltas = {}
        for k, _ in METRICS:
            va, vb = ma.get(k), mb.get(k)
            if va is None or vb is None:
                deltas[k] = None
            else:
                deltas[k] = va - vb
        iso_rows.append([f"{a} − {b}", label] + [fmt_num(deltas.get(k)) for k, _ in METRICS])
    write_csv(root / "isolations.csv",
              ["comparison", "isolated component"] + [pretty for _, pretty in METRICS],
              iso_rows)

    # === Per-category ===
    pc = aggregate_per_category(root)
    cats = sorted({cat for _, cat in pc.keys()})
    pc_rows = []
    for ab in ABLATIONS:
        for cat in cats:
            m = pc.get((ab, cat), {})
            if not m:
                continue
            pc_rows.append([ab, cat, m.get("n", 0)] + [fmt_num(m.get(k)) for k, _ in METRICS])
    write_csv(root / "per_category.csv",
              ["ablation", "category", "n"] + [k for k, _ in METRICS],
              pc_rows)

    # === Markdown tables for REPORT.md ===
    md = []
    md.append("# Generated tables for REPORT.md (copy-paste into §7)\n")

    # §7.1 main A1 vs A5
    md.append("## §7.1 Main comparison (A1 Base vs A5 Full FT)\n")
    md.append("| Metric | A1 Base | A5 Full FT | Δ |\n|---|---|---|---|")
    a1, a5 = main_table.get("A1", {}), main_table.get("A5", {})
    for k, pretty in METRICS:
        va, vb = a1.get(k), a5.get(k)
        delta = (vb - va) if (va is not None and vb is not None) else None
        md.append(f"| {pretty} | {fmt_num(va)} | {fmt_num(vb)} | {fmt_num(delta)} |")
    md.append("")

    # §7.2 Isolations
    md.append("\n## §7.2 Ablation isolations\n")
    md.append("| Comparison | Isolated component | ΔR@10 | ΔCit-F1 | ΔFaith |")
    md.append("|---|---|---|---|---|")
    for a, b, label in ISOLATIONS:
        ma, mb = main_table.get(a, {}), main_table.get(b, {})
        def d(k):
            va, vb = ma.get(k), mb.get(k)
            if va is None or vb is None:
                return "—"
            return fmt_num(va - vb)
        md.append(f"| {a} − {b} | {label} | {d('recall@10')} | {d('cit_f1')} | {d('faithfulness')} |")
    md.append("")

    # §7.3 Full ablation table
    md.append("\n## §7.3 All 8 ablations × all metrics\n")
    md.append("| Ablation | " + " | ".join(p for _, p in METRICS) + " |")
    md.append("|" + "---|" * (len(METRICS) + 1))
    for ab in ABLATIONS:
        m = main_table.get(ab, {})
        md.append(f"| **{ab}** | " + " | ".join(fmt_num(m.get(k)) for k, _ in METRICS) + " |")
    md.append("")

    # §7.4 Per-category Citation F1
    md.append("\n## §7.4 Per-category Citation F1\n")
    md.append("| Category | " + " | ".join(ABLATIONS) + " |")
    md.append("|" + "---|" * (len(ABLATIONS) + 1))
    for cat in cats:
        row = [cat]
        for ab in ABLATIONS:
            v = pc.get((ab, cat), {}).get("cit_f1")
            row.append(fmt_num(v))
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    (root / "REPORT_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Wrote %s", root / "REPORT_TABLES.md")

    # === Slide-ready hand-edit rows ===
    slides = []
    slides.append("# SLIDES.pptx — slide 12 (A1 vs A5) values\n")
    for k, pretty in METRICS:
        va, vb = a1.get(k), a5.get(k)
        delta = (vb - va) if (va is not None and vb is not None) else None
        slides.append(f"{pretty:>10s}   A1: {fmt_num(va):>7s}   A5: {fmt_num(vb):>7s}   Δ: {fmt_num(delta):>7s}")
    slides.append("\n# SLIDES.pptx — slide 13 (isolations) values\n")
    for a, b, label in ISOLATIONS:
        ma, mb = main_table.get(a, {}), main_table.get(b, {})
        slides.append(f"{a:>4s} − {b:<4s}  {label:<24s}  "
                      + "  ".join(f"Δ{p}: {fmt_num((ma.get(k) or 0) - (mb.get(k) or 0)) if (ma.get(k) is not None and mb.get(k) is not None) else '—'}"
                                  for k, p in [("recall@10", "R@10"), ("cit_f1", "Cit-F1"), ("faithfulness", "Faith")]))
    (root / "SLIDES_TABLES.md").write_text("\n".join(slides), encoding="utf-8")
    log.info("Wrote %s", root / "SLIDES_TABLES.md")

    print("\nDONE. Generated:")
    for f in ["ablation_table.csv", "isolations.csv", "per_category.csv",
              "REPORT_TABLES.md", "SLIDES_TABLES.md"]:
        print(f"  {root / f}")


if __name__ == "__main__":
    main()
