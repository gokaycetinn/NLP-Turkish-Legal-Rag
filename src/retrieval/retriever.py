"""
Unified retriever for the Turkish Legal RAG system.

Supports:
  - BM25 sparse retrieval
  - Dense (FAISS) retrieval with any sentence-transformers model
  - Hybrid (BM25 + dense) via Reciprocal Rank Fusion (RRF)

Usage:
    retriever = Retriever.load("data/index", embed_model="intfloat/multilingual-e5-large")
    hits = retriever.search("Cumhurbaşkanına hakaret suçunun cezası nedir?", top_k=5, mode="hybrid")
    for h in hits:
        print(h["score"], h["kanun_short"], h["madde_no"], h["text"][:100])
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .build_index import tokenize, doc_text_for_indexing

log = logging.getLogger("retriever")

RetrievalMode = Literal["bm25", "dense", "hybrid"]


@dataclass
class Hit:
    doc_id: str
    score: float
    kanun_short: str
    kanun_full: str
    madde_no: str
    baslik: str | None
    text: str
    url: str
    rank: int = 0

    def to_dict(self):
        return {
            "doc_id": self.doc_id, "score": self.score,
            "kanun_short": self.kanun_short, "kanun_full": self.kanun_full,
            "madde_no": self.madde_no, "baslik": self.baslik,
            "text": self.text, "url": self.url, "rank": self.rank,
        }


class Retriever:
    """Hybrid retriever over a corpus indexed by build_index."""

    def __init__(self, index_dir: Path, embed_model: str | None = None):
        self.index_dir = Path(index_dir)
        self.embed_model_name = embed_model
        self.meta: list[dict] = []
        self.bm25 = None
        self.faiss_index = None
        self.encoder = None
        self._needs_e5_prefix = False

    @classmethod
    def load(cls, index_dir: str | Path, embed_model: str = "intfloat/multilingual-e5-large",
             load_bm25: bool = True, load_dense: bool = True) -> "Retriever":
        r = cls(Path(index_dir), embed_model=embed_model)
        r._load_meta()
        if load_bm25:
            r._load_bm25()
        if load_dense:
            r._load_dense(embed_model)
        return r

    # ---- loading ------------------------------------------------------------

    def _load_meta(self):
        path = self.index_dir / "meta.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Metadata not found: {path}")
        with path.open(encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f]
        log.info("Loaded %d meta records", len(self.meta))

    def _load_bm25(self):
        path = self.index_dir / "bm25.pkl"
        if not path.exists():
            raise FileNotFoundError(f"BM25 not found: {path}")
        with path.open("rb") as f:
            obj = pickle.load(f)
        self.bm25 = obj["bm25"]
        log.info("Loaded BM25 (n=%d)", self.bm25.corpus_size)

    def _load_dense(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            raise SystemExit("Install: pip install sentence-transformers faiss-cpu")

        safe = model_name.replace("/", "_")
        faiss_path = self.index_dir / f"faiss_{safe}.index"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

        self.faiss_index = faiss.read_index(str(faiss_path))
        log.info("Loaded FAISS index (ntotal=%d)", self.faiss_index.ntotal)

        log.info("Loading encoder: %s", model_name)
        self.encoder = SentenceTransformer(model_name)
        self._needs_e5_prefix = "e5" in model_name.lower()

    # ---- search -------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        scores = self.bm25.get_scores(toks)
        top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx]

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q = "query: " + query if self._needs_e5_prefix else query
        emb = self.encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, ids = self.faiss_index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    @staticmethod
    def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion of multiple ranked id lists."""
        fused: dict[int, float] = {}
        for ranks in rank_lists:
            for r, idx in enumerate(ranks):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + r + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def search(self, query: str, top_k: int = 10, mode: RetrievalMode = "hybrid",
               candidate_k: int = 50) -> list[Hit]:
        """Search the corpus and return top_k hits."""
        if mode == "bm25":
            results = self._bm25_search(query, top_k)
        elif mode == "dense":
            results = self._dense_search(query, top_k)
        elif mode == "hybrid":
            bm = self._bm25_search(query, candidate_k)
            de = self._dense_search(query, candidate_k)
            bm_ids = [i for i, _ in bm]
            de_ids = [i for i, _ in de]
            fused = self._rrf([bm_ids, de_ids])[:top_k]
            results = fused
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return [self._build_hit(idx, score, rank) for rank, (idx, score) in enumerate(results)]

    def _build_hit(self, idx: int, score: float, rank: int) -> Hit:
        m = self.meta[idx]
        return Hit(
            doc_id=m["doc_id"], score=score,
            kanun_short=m.get("kanun_short", ""), kanun_full=m.get("kanun_full", ""),
            madde_no=m.get("madde_no", ""), baslik=m.get("baslik"),
            text=m.get("text", ""), url=m.get("url", ""), rank=rank,
        )


# ---- CLI for quick sanity check --------------------------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("query", help="Query string in Turkish")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    r = Retriever.load(
        repo_root / args.index_dir,
        embed_model=args.embed_model,
        load_dense=(args.mode != "bm25"),
        load_bm25=(args.mode != "dense"),
    )
    hits = r.search(args.query, top_k=args.top_k, mode=args.mode)
    for h in hits:
        print(f"\n[#{h.rank+1}] score={h.score:.4f}  {h.kanun_short} m.{h.madde_no}  ({h.kanun_full})")
        print(f"     {h.text[:200]}...")


if __name__ == "__main__":
    _cli()
