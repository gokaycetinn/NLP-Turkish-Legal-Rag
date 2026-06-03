"""
End-to-end evaluation harness for the Turkish Legal RAG pipeline.

Reads a test set JSONL with fields:
  - question        (str, required)
  - answer          (str, gold reference answer — optional for retrieval-only eval)
  - gold_doc_ids    (list[str], required for retrieval metrics)
  - gold_citations  (list[[kanun_short, madde_no]], optional — defaults to gold_doc_ids' meta)

For each example: runs retrieval, optionally the full RAG (LLM) pipeline, and computes:
  - Retrieval: Recall@{1,3,5,10}, MRR@10, nDCG@10
  - Answer: EM, token-F1, BLEU, ROUGE-L
  - Citation: precision / recall / F1
  - Faithfulness: LLM-as-judge over retrieved context (optional, --judge)

Writes:
  - <out>/per_example.jsonl    one row per question with all metrics + answer
  - <out>/summary.json         macro averages

Usage:
  # retrieval-only (fast, no API)
  python -m src.eval.run_eval --test data/test_set/dev.jsonl --mode hybrid --no-llm

  # full RAG with Gemini + faithfulness judge
  python -m src.eval.run_eval --test data/test_set/dev.jsonl --llm-backend gemini \\
      --llm-model gemini-2.5-flash --judge --out results/eval_gemini_flash
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean

from src.eval.metrics import (
    bleu, citation_scores, exact_match, macro_avg,
    mrr_at_k, ndcg_at_k, recall_at_k, rouge_l, token_f1,
)
from src.pipeline.rag import RAGPipeline, format_context
from src.retrieval.retriever import Retriever

log = logging.getLogger("eval")


def load_test_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        if "gold_doc_ids" in r and isinstance(r["gold_doc_ids"], str):
            r["gold_doc_ids"] = [r["gold_doc_ids"]]
        # if gold_citations missing but corpus meta available, harness will fill in later
        r.setdefault("gold_doc_ids", [])
        r.setdefault("gold_citations", [])
    return rows


def derive_gold_citations(row: dict, meta_by_id: dict) -> set[tuple[str, str]]:
    if row.get("gold_citations"):
        return set((s.lower(), str(n)) for s, n in row["gold_citations"])
    out = set()
    for did in row.get("gold_doc_ids", []):
        m = meta_by_id.get(did)
        if m:
            out.add((m.get("kanun_short", "").lower(), str(m.get("madde_no", ""))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Test set JSONL")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-llm", action="store_true", help="Retrieval-only (skip LLM)")
    ap.add_argument("--llm-backend", choices=["hf", "gemini", "dummy"], default="dummy")
    ap.add_argument("--llm-model", default="gemini-2.5-flash")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--reranker", choices=["none", "v1", "v2", "pretrained"], default="none",
                    help="Cross-encoder reranker over first-stage candidates.")
    ap.add_argument("--reranker-dir", default="data/models/bge-reranker-tr-legal-v2",
                    help="For v2/v1: local dir; for pretrained: HF id.")
    ap.add_argument("--candidate-k", type=int, default=50,
                    help="First-stage top-k retrieved before reranking.")
    ap.add_argument("--judge", action="store_true", help="Run faithfulness judge")
    ap.add_argument("--out", default="results/eval_run")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    test_path = repo_root / args.test
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_test_set(test_path)
    if args.limit:
        rows = rows[: args.limit]
    log.info("Loaded %d test examples", len(rows))

    # Optional reranker
    reranker = None
    if args.reranker != "none":
        from src.reranker import Reranker
        rr_path = (repo_root / args.reranker_dir) if args.reranker != "pretrained" else args.reranker_dir
        if args.reranker == "v2":
            reranker = Reranker.load_v2(rr_path)
        elif args.reranker == "v1":
            reranker = Reranker.load_v1(rr_path)
        else:
            reranker = Reranker.load_pretrained(args.reranker_dir)
        log.info("Reranker loaded: %s", reranker.name)

    # Build retriever (and pipeline if needed)
    if args.no_llm:
        retriever = Retriever.load(
            repo_root / args.index_dir, embed_model=args.embed_model,
            load_bm25=(args.mode != "dense"), load_dense=(args.mode != "bm25"),
        )
        rag = None
    else:
        rag = RAGPipeline.build(
            index_dir=str(repo_root / args.index_dir),
            embed_model=args.embed_model,
            llm_backend=args.llm_backend, llm_model=args.llm_model,
            top_k=args.top_k, retrieval_mode=args.mode,
            adapter_path=args.adapter_path,
            reranker=reranker, candidate_k=args.candidate_k,
        )
        retriever = rag.retriever

    meta_by_id = {m["doc_id"]: m for m in retriever.meta}

    judge = None
    if args.judge:
        from src.eval.judge import FaithfulnessJudge
        judge = FaithfulnessJudge()

    per_example_path = out_dir / "per_example.jsonl"
    summary_path = out_dir / "summary.json"

    ks = [1, 3, 5, 10]
    retr_metrics = {f"recall@{k}": [] for k in ks}
    retr_metrics["mrr@10"] = []
    retr_metrics["ndcg@10"] = []
    ans_metrics = {"em": [], "f1": [], "bleu": [], "rouge_l": []}
    cit_metrics = {"cit_p": [], "cit_r": [], "cit_f1": []}
    faith_scores: list[float] = []

    with per_example_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            q = row["question"]
            gold_ids = set(row.get("gold_doc_ids", []))
            gold_cits = derive_gold_citations(row, meta_by_id)

            # First-stage retrieval. If reranker is used, fetch candidate_k then rerank.
            first_k = args.candidate_k if reranker else max(args.top_k, 10)
            hits = retriever.search(q, top_k=first_k, mode=args.mode)
            if reranker:
                hits = reranker.rerank(q, hits, top_k=max(args.top_k, 10))
            retrieved_ids = [h.doc_id for h in hits]

            r_row = {}
            for k in ks:
                v = recall_at_k(retrieved_ids, gold_ids, k)
                retr_metrics[f"recall@{k}"].append(v)
                r_row[f"recall@{k}"] = v
            mrr = mrr_at_k(retrieved_ids, gold_ids, 10)
            ndcg = ndcg_at_k(retrieved_ids, gold_ids, 10)
            retr_metrics["mrr@10"].append(mrr)
            retr_metrics["ndcg@10"].append(ndcg)
            r_row["mrr@10"] = mrr
            r_row["ndcg@10"] = ndcg

            answer = None
            faith = None
            if rag is not None:
                topk_hits = hits[: args.top_k]
                context = format_context(topk_hits)
                from src.pipeline.rag import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
                user_prompt = USER_PROMPT_TEMPLATE.format(question=q, context=context)
                answer = rag.llm.generate(SYSTEM_PROMPT, user_prompt)

                gold_a = row.get("answer", "") or ""
                if gold_a:
                    ans_metrics["em"].append(exact_match(answer, gold_a))
                    ans_metrics["f1"].append(token_f1(answer, gold_a))
                    ans_metrics["bleu"].append(bleu(answer, gold_a))
                    ans_metrics["rouge_l"].append(rouge_l(answer, gold_a))

                cs = citation_scores(answer, gold_cits)
                cit_metrics["cit_p"].append(cs["p"])
                cit_metrics["cit_r"].append(cs["r"])
                cit_metrics["cit_f1"].append(cs["f1"])
                r_row["citation"] = cs

                if judge is not None:
                    faith_out = judge.score(q, answer, context)
                    faith = faith_out.get("faithfulness")
                    if faith is not None:
                        faith_scores.append(faith)
                    r_row["faithfulness"] = faith_out

            fout.write(json.dumps({
                "idx": i, "question": q,
                "gold_doc_ids": list(gold_ids),
                "retrieved_doc_ids": retrieved_ids[:10],
                "answer": answer, "gold_answer": row.get("answer"),
                **r_row,
            }, ensure_ascii=False) + "\n")

            print(f"[eval] {i + 1}/{len(rows)} done", flush=True)

    def _avg(d):
        return {k: macro_avg(v) for k, v in d.items() if v}

    summary = {
        "n_examples": len(rows),
        "retrieval_mode": args.mode,
        "top_k": args.top_k,
        "retrieval": _avg(retr_metrics),
        "answer": _avg(ans_metrics) if rag else {},
        "citation": _avg(cit_metrics) if rag else {},
        "faithfulness_mean": mean(faith_scores) if faith_scores else None,
        "faithfulness_n": len(faith_scores),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("=== SUMMARY ===")
    log.info(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("Wrote: %s", summary_path)


if __name__ == "__main__":
    main()
