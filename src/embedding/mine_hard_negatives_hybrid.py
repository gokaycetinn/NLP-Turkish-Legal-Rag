"""
Hybrid hard negative mining: BM25 ∪ dense (e5-large base).

For each synthetic QA row:
  1. Get BM25 top-K and dense top-K candidates (K=50)
  2. Union the candidate pools (excluding the positive)
  3. Score each candidate with combined normalized BM25 + dense score
  4. Boost candidates from the same kanun (+20%) — harder confusables
  5. Pick top --num-negatives as hard negatives
  6. Fallback: pad with random same-kanun maddeler if too few

Why hybrid: BM25-only mining produces negatives the dense model already
distinguishes easily. Adding the base e5 retrieval picks up dense-confusable
negatives, giving the cross-encoder reranker harder training signal.

Usage:
    python -m src.embedding.mine_hard_negatives_hybrid \\
        --qa data/finetune/synthetic_qa.jsonl \\
        --index-dir data/index_full \\
        --output data/finetune/embed_triplets_v2.jsonl \\
        --top-k 50 --num-negatives 7

For a smoke run on 100 rows:
    python -m src.embedding.mine_hard_negatives_hybrid --limit 100
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

log = logging.getLogger("mine_hard_negatives_hybrid")

DEFAULT_SAME_KANUN_BOOST = 1.20


def load_qa(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_kanun_index(meta: list[dict]) -> dict[str, list[int]]:
    by_kanun: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        by_kanun[m.get("kanun_short", "")].append(i)
    return by_kanun


def _minmax_normalize(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vs = list(scores.values())
    lo, hi = min(vs), max(vs)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def mine_for_question(
    retriever: Retriever,
    question: str,
    positive_doc_id: str,
    top_k: int,
    num_negatives: int,
    meta_by_id: dict[str, dict],
    by_kanun: dict[str, list[int]],
    rng: random.Random,
    same_kanun_boost: float = DEFAULT_SAME_KANUN_BOOST,
) -> dict:
    """Hybrid BM25 ∪ dense candidate pool, score-combined hard-neg selection."""
    bm = retriever._bm25_search(question, top_k)
    de = retriever._dense_search(question, top_k)

    bm_scores = {idx: s for idx, s in bm}
    de_scores = {idx: s for idx, s in de}

    # Track positive rank (use best of two lists for reporting)
    positive_rank_bm = next(
        (i + 1 for i, (idx, _) in enumerate(bm)
         if retriever.meta[idx]["doc_id"] == positive_doc_id),
        None,
    )
    positive_rank_de = next(
        (i + 1 for i, (idx, _) in enumerate(de)
         if retriever.meta[idx]["doc_id"] == positive_doc_id),
        None,
    )
    positive_rank = (
        min(r for r in (positive_rank_bm, positive_rank_de) if r is not None)
        if (positive_rank_bm or positive_rank_de) else None
    )

    # Normalize scores within each list, then combine
    bm_norm = _minmax_normalize(bm_scores)
    de_norm = _minmax_normalize(de_scores)
    pos_kanun = meta_by_id.get(positive_doc_id, {}).get("kanun_short", "")

    candidate_scores: dict[int, float] = {}
    for idx in set(bm_scores) | set(de_scores):
        m = retriever.meta[idx]
        if m["doc_id"] == positive_doc_id:
            continue
        combined = 0.5 * bm_norm.get(idx, 0.0) + 0.5 * de_norm.get(idx, 0.0)
        if pos_kanun and m.get("kanun_short", "") == pos_kanun:
            combined *= same_kanun_boost
        candidate_scores[idx] = combined

    ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
    negatives = [retriever._build_hit(idx, score, rank=r)
                 for r, (idx, score) in enumerate(ranked[:num_negatives])]

    # Fallback: pad with random in-kanun maddeler
    padded = False
    if len(negatives) < num_negatives:
        padded = True
        used_ids = {n.doc_id for n in negatives} | {positive_doc_id}
        candidates = list(by_kanun.get(pos_kanun, [])) or list(range(len(retriever.meta)))
        rng.shuffle(candidates)
        for idx in candidates:
            if len(negatives) >= num_negatives:
                break
            m = retriever.meta[idx]
            if m["doc_id"] in used_ids:
                continue
            negatives.append(retriever._build_hit(idx, score=0.0, rank=99))
            used_ids.add(m["doc_id"])

    return {
        "negatives": negatives[:num_negatives],
        "positive_rank": positive_rank,
        "positive_rank_bm25": positive_rank_bm,
        "positive_rank_dense": positive_rank_de,
        "positive_in_top_k": positive_rank is not None,
        "padded": padded,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/finetune/synthetic_qa.jsonl")
    ap.add_argument("--index-dir", default="data/index_full")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--output", default="data/finetune/embed_triplets_v2.jsonl")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-negatives", type=int, default=7)
    ap.add_argument("--same-kanun-boost", type=float, default=DEFAULT_SAME_KANUN_BOOST,
                    help="Multiplier for hard negatives from the same kanun (default: 1.20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, mine only first N rows (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    qa_path = repo_root / args.qa
    out_path = repo_root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qa_rows = load_qa(qa_path)
    if args.limit:
        qa_rows = qa_rows[: args.limit]
        log.info("SMOKE TEST: limiting to %d rows", args.limit)
    log.info("Loaded %d QA records", len(qa_rows))

    retriever = Retriever.load(
        repo_root / args.index_dir,
        embed_model=args.embed_model,
        load_bm25=True,
        load_dense=True,
    )
    meta_by_id = {m["doc_id"]: m for m in retriever.meta}
    by_kanun = build_kanun_index(retriever.meta)
    rng = random.Random(args.seed)

    stats = {
        "total": 0,
        "positive_in_top_k": 0,
        "positive_in_bm25": 0,
        "positive_in_dense": 0,
        "positive_ranks": [],
        "padded": 0,
        "same_kanun_neg_count": 0,
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
                same_kanun_boost=args.same_kanun_boost,
            )

            negs = result["negatives"]
            stats["total"] += 1
            if result["positive_in_top_k"]:
                stats["positive_in_top_k"] += 1
                stats["positive_ranks"].append(result["positive_rank"])
            if result["positive_rank_bm25"]:
                stats["positive_in_bm25"] += 1
            if result["positive_rank_dense"]:
                stats["positive_in_dense"] += 1
            if result["padded"]:
                stats["padded"] += 1
            pos_kanun = positive_meta.get("kanun_short", "")
            stats["same_kanun_neg_count"] += sum(
                1 for n in negs if n.kanun_short == pos_kanun
            )

            out_row = {
                "anchor": qa["question"],
                "positive": positive_meta.get("text", ""),
                "positive_doc_id": positive_doc_id,
                "negatives": [n.text for n in negs],
                "negative_doc_ids": [n.doc_id for n in negs],
                "negative_kanun_shorts": [n.kanun_short for n in negs],
                "difficulty": qa.get("difficulty"),
                "kanun_short": qa.get("kanun_short"),
                "positive_in_top50": result["positive_in_top_k"],
                "positive_rank": result["positive_rank"],
                "positive_rank_bm25": result["positive_rank_bm25"],
                "positive_rank_dense": result["positive_rank_dense"],
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    log.info("=== STATS ===")
    log.info("Toplam triplet: %d", stats["total"])
    if stats["missing_positive_doc"]:
        log.info("Eksik positive doc: %d", stats["missing_positive_doc"])
    if stats["total"]:
        n = stats["total"]
        log.info("Positive hybrid top-%d'de: %d/%d (%.1f%%)",
                 args.top_k, stats["positive_in_top_k"], n,
                 100 * stats["positive_in_top_k"] / n)
        log.info("  BM25 yakaladı: %d (%.1f%%)",
                 stats["positive_in_bm25"], 100 * stats["positive_in_bm25"] / n)
        log.info("  Dense yakaladı: %d (%.1f%%)",
                 stats["positive_in_dense"], 100 * stats["positive_in_dense"] / n)
    if stats["positive_ranks"]:
        ranks = stats["positive_ranks"]
        log.info("Positive rank ortalama: %.2f (min=%d max=%d)",
                 sum(ranks) / len(ranks), min(ranks), max(ranks))
    if stats["total"]:
        avg_same = stats["same_kanun_neg_count"] / stats["total"]
        log.info("Aynı kanundan ortalama negatif: %.2f / %d", avg_same, args.num_negatives)
    log.info("Random ile dolduruldu: %d satır", stats["padded"])
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()
