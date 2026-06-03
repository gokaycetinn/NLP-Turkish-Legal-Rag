"""
Standalone faithfulness judge CLI.

Takes a per_example.jsonl produced by run_eval.py (which already contains
question, answer, sources/retrieved context), runs the LLM-as-judge on each
row, and writes a new JSONL with `faithfulness` + `judge_claims` fields added.

Usage:
    # Single ablation
    python -m src.eval.judge_run \
        --in results/A5/per_example.jsonl \
        --out results/A5/with_judge.jsonl \
        --corpus data/corpus/mevzuat_full_normalized.jsonl

    # Batch — all ablations under results/
    python -m src.eval.judge_run \
        --batch outputs/results_v2/ \
        --corpus data/corpus/mevzuat_full_normalized.jsonl

The script is **resumable**: if --out already exists, rows whose id is already
present and have a non-null faithfulness are skipped. Useful when API quotas
or rate-limits cut a run short.

Requires ANTHROPIC_API_KEY in env or .env.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean

from src.eval.judge import FaithfulnessJudge

log = logging.getLogger("judge_run")


def _load_corpus(corpus_path: str | None) -> dict[str, dict]:
    """Load corpus JSONL into a dict keyed by doc_id for fast lookup."""
    if not corpus_path:
        return {}
    p = Path(corpus_path)
    if not p.exists():
        log.warning("Corpus file not found: %s — context will be empty", corpus_path)
        return {}
    log.info("Loading corpus from %s …", p)
    corpus = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                doc_id = doc.get("doc_id")
                if doc_id:
                    corpus[doc_id] = doc
            except json.JSONDecodeError:
                continue
    log.info("Corpus loaded: %d documents", len(corpus))
    return corpus


def _format_context(sources: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved source dicts into a plain-text context block."""
    parts, used = [], 0
    for s in sources or []:
        head = f"[{s.get('kanun_short', '?')} m.{s.get('madde_no', '?')}]"
        baslik = s.get("baslik")
        if baslik:
            head += f" ({baslik})"
        block = head + "\n" + (s.get("text") or "")
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _context_from_doc_ids(doc_ids: list[str], corpus: dict[str, dict], max_chars: int = 6000) -> str:
    """Build context string from retrieved_doc_ids using corpus lookup."""
    sources = []
    for doc_id in (doc_ids or []):
        doc = corpus.get(doc_id)
        if doc:
            sources.append(doc)
    return _format_context(sources, max_chars)


def _load_done(out_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                rid = r.get("id") or str(r.get("idx", ""))
                if rid and r.get("faithfulness") is not None:
                    done[rid] = r
            except json.JSONDecodeError:
                continue
    return done


def judge_file(in_path: Path, out_path: Path, judge: FaithfulnessJudge,
               corpus: dict[str, dict] | None = None,
               sleep: float = 0.3, limit: int | None = None) -> dict:
    rows = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    done = _load_done(out_path)
    log.info("File %s: %d rows, %d already done", in_path.name, len(rows), len(done))

    # Append-mode: write back done rows + new
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    written = 0
    scored: list[float] = []
    failed = 0

    with tmp.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            rid = row.get("id") or str(row.get("idx", f"row{i}"))
            if rid in done:
                merged = done[rid]
            else:
                q = row.get("question", "")
                a = row.get("answer", "")
                # Prefer inline sources; fall back to corpus doc_id lookup
                sources = row.get("sources") or row.get("retrieved") or []
                if sources:
                    context = _format_context(sources)
                else:
                    doc_ids = row.get("retrieved_doc_ids") or []
                    context = _context_from_doc_ids(doc_ids, corpus or {})
                if not a or not context:
                    result = {"faithfulness": None, "claims": [], "error": "missing_answer_or_context"}
                else:
                    try:
                        result = judge.score(q, a, context, sleep=sleep)
                    except Exception as e:
                        log.warning("[%s] judge error: %s", rid, e)
                        result = {"faithfulness": None, "claims": [], "error": str(e)}
                        failed += 1
                merged = dict(row)
                merged["faithfulness"] = result.get("faithfulness")
                merged["judge_claims"] = result.get("claims", [])
                if result.get("error"):
                    merged["judge_error"] = result["error"]

            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if merged.get("faithfulness") is not None:
                scored.append(merged["faithfulness"])

            if (i + 1) % 10 == 0:
                avg = mean(scored) if scored else float("nan")
                log.info("  [%d/%d] avg faithfulness so far: %.3f", i + 1, len(rows), avg)

    tmp.replace(out_path)
    summary = {
        "n_rows": written,
        "n_scored": len(scored),
        "n_failed": failed,
        "mean_faithfulness": mean(scored) if scored else None,
    }
    log.info("DONE %s | scored %d/%d | mean=%s",
             out_path.name, len(scored), written, summary["mean_faithfulness"])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="Input per_example.jsonl (single mode)")
    ap.add_argument("--out", dest="out_path", help="Output with_judge.jsonl (single mode)")
    ap.add_argument("--batch", help="Process */per_example.jsonl for every ablation under given dir")
    ap.add_argument("--corpus", default=None, help="Path to mevzuat_full_normalized.jsonl for doc_id lookup")
    ap.add_argument("--sleep", type=float, default=0.5, help="Pause between API calls (rate-limit cushion)")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N rows (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    judge = FaithfulnessJudge()  # raises if GROQ_API_KEY missing
    corpus = _load_corpus(args.corpus)

    if args.batch:
        root = Path(args.batch)
        targets = sorted(root.glob("*/per_example.jsonl"))
        if not targets:
            log.error("No per_example.jsonl found under %s", root)
            sys.exit(1)
        log.info("Batch mode: %d ablations to judge", len(targets))
        all_summaries = {}
        for in_path in targets:
            ablation = in_path.parent.name
            out_path = in_path.parent / "with_judge.jsonl"
            log.info("=== %s ===", ablation)
            all_summaries[ablation] = judge_file(in_path, out_path, judge, corpus, args.sleep, args.limit)
        # Write batch summary
        out_summary = root / "judge_summary.json"
        out_summary.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
        log.info("Batch summary: %s", out_summary)
    elif args.in_path and args.out_path:
        in_path = Path(args.in_path)
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        judge_file(in_path, out_path, judge, corpus, args.sleep, args.limit)
    else:
        ap.error("Either --in/--out or --batch required")


if __name__ == "__main__":
    main()
