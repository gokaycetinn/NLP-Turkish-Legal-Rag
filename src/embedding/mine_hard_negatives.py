"""
Mine hard negatives from BM25 for each synthetic (question, madde) pair.

For each row in synthetic_qa.jsonl we:
  1. Run BM25 with the question
  2. Take top-K hits, exclude the positive doc_id
  3. Keep the first --num-negatives as hard negatives
  4. If fewer than --num-negatives remain, pad with random in-domain (same kanun) maddeler

Output: data/finetune/embed_triplets.jsonl, ready for sentence-transformers
contrastive training (TripletLoss or MultipleNegativesRankingLoss).

Usage:
    python -m src.embedding.mine_hard_negatives \\
        --qa data/finetune/synthetic_qa.jsonl \\
        --index-dir data/index \\
        --output data/finetune/embed_triplets.jsonl \\
        --top-k 20 --num-negatives 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.retrieval.retriever import Retriever

log = logging.getLogger("mine_hard_negatives")


def load_qa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_kanun_index(meta: list[dict]) -> dict[str, list[int]]:
    """Map kanun_short -> list of meta indices for in-domain random fallback."""
    by_kanun: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_kanun[m.get("kanun_short", "")].append(i)
    return by_kanun


def mine_for_question(retriever: Retriever, question: str, positive_doc_id: str,
                      top_k: int, num_negatives: int,
                      meta_by_id: dict[str, dict], by_kanun: dict[str, list[int]],
                      rng: random.Random) -> dict:
    """Run BM25, return (hard_negatives, positive_rank_in_top_k_or_None)."""
    hits = retriever.search(question, top_k=top_k, mode="bm25")

    positive_rank = None
    negatives = []
    for i, h in enumerate(hits):
        if h.doc_id == positive_doc_id:
            positive_rank = i + 1
            continue
        if len(negatives) < num_negatives:
            negatives.append(h)

    # Fallback: pad with random in-domain maddeler if BM25 didn't give enough
    if len(negatives) < num_negatives:
        kanun = meta_by_id.get(positive_doc_id, {}).get("kanun_short", "")
        used_ids = {n.doc_id for n in negatives} | {positive_doc_id}
        candidates = by_kanun.get(kanun, []) or list(range(len(retriever.meta)))
        rng.shuffle(candidates)
        for idx in candidates:
            if len(negatives) >= num_negatives:
                break
            m = retriever.meta[idx]
            if m["doc_id"] in used_ids:
                continue
            # Build a Hit-like object via the retriever internals
            negatives.append(retriever._build_hit(idx, score=0.0, rank=99))
            used_ids.add(m["doc_id"])

    return {
        "negatives": negatives[:num_negatives],
        "positive_rank": positive_rank,
        "positive_in_top_k": positive_rank is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/finetune/synthetic_qa.jsonl")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--output", default="data/finetune/embed_triplets.jsonl")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--num-negatives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    qa_path = repo_root / args.qa
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qa_rows = load_qa(qa_path)
    log.info("Loaded %d QA records", len(qa_rows))

    retriever = Retriever.load(
        repo_root / args.index_dir, load_bm25=True, load_dense=False
    )
    meta_by_id = {m["doc_id"]: m for m in retriever.meta}
    by_kanun = build_kanun_index(retriever.meta)
    rng = random.Random(args.seed)

    stats = {
        "total": 0,
        "positive_in_top_k": 0,
        "positive_ranks": [],
        "padded_with_random": 0,
        "missing_positive_doc": 0,
    }

    with out_path.open("w", encoding="utf-8") as fout:
        for qa in tqdm(qa_rows, desc="mining"):
            positive_doc_id = qa["source_doc_id"]
            positive_meta = meta_by_id.get(positive_doc_id)
            if not positive_meta:
                stats["missing_positive_doc"] += 1
                continue

            result = mine_for_question(
                retriever=retriever,
                question=qa["question"],
                positive_doc_id=positive_doc_id,
                top_k=args.top_k,
                num_negatives=args.num_negatives,
                meta_by_id=meta_by_id,
                by_kanun=by_kanun,
                rng=rng,
            )

            negs = result["negatives"]
            stats["total"] += 1
            if result["positive_in_top_k"]:
                stats["positive_in_top_k"] += 1
                stats["positive_ranks"].append(result["positive_rank"])
            bm25_neg_count = sum(1 for n in negs if n.rank != 99)
            if bm25_neg_count < args.num_negatives:
                stats["padded_with_random"] += 1

            out_row = {
                "anchor": qa["question"],
                "positive": positive_meta.get("text", ""),
                "positive_doc_id": positive_doc_id,
                "negatives": [n.text for n in negs],
                "negative_doc_ids": [n.doc_id for n in negs],
                "difficulty": qa.get("difficulty"),
                "kanun_short": qa.get("kanun_short"),
                "positive_in_top20": result["positive_in_top_k"],
                "positive_rank": result["positive_rank"],
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    # Report
    log.info("=== STATS ===")
    log.info("Toplam triplet: %d", stats["total"])
    if stats["missing_positive_doc"]:
        log.info("Eksik positive doc (corpus'ta yok): %d", stats["missing_positive_doc"])
    in_topk = stats["positive_in_top_k"]
    if stats["total"]:
        log.info("Positive BM25 top-%d'de: %d/%d (%.1f%%)",
                 args.top_k, in_topk, stats["total"], 100 * in_topk / stats["total"])
    if stats["positive_ranks"]:
        ranks = stats["positive_ranks"]
        log.info("Positive rank ortalama: %.2f (min=%d max=%d)",
                 sum(ranks) / len(ranks), min(ranks), max(ranks))
    log.info("Random ile dolduruldu: %d satır", stats["padded_with_random"])
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
