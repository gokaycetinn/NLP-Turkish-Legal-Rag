"""
Build BM25 (sparse) and FAISS (dense) indexes over the mevzuat corpus.

Inputs:
  data/corpus/mevzuat_*.jsonl  (one madde per line, fields: doc_id, kanun_short, madde_no, text, ...)

Outputs:
  data/index/bm25.pkl                    — pickled BM25Okapi + tokenized corpus
  data/index/faiss_{model_safe}.index    — FAISS index
  data/index/meta.parquet                — doc_id -> kanun/madde metadata mapping
  data/index/embeddings_{model}.npy      — raw embedding matrix (for re-ranking experiments)

Run:
    python -m src.retrieval.build_index --corpus data/corpus/mevzuat_priority.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_DIR = DATA_DIR / "index"

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE = 32

log = logging.getLogger("build_index")


# ---- Turkish-friendly tokenization -----------------------------------------

# Simple lowercase + remove punctuation; preserves Turkish characters.
# For better results, a Zemberek-NLP tokenizer can be plugged here later.
TOKEN_PATTERN = re.compile(r"[a-zçğıöşüâîû0-9]+", re.IGNORECASE)


def tr_lower(text: str) -> str:
    """Turkish-aware lowercase (handles I/İ correctly)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(tr_lower(text))


# ---- Corpus loading --------------------------------------------------------

def load_corpus(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    log.info("Loaded %d maddeler from %s", len(records), jsonl_path)
    return records


def doc_text_for_indexing(record: dict) -> str:
    """Combine kanun name + heading + text for a richer searchable doc."""
    parts = [
        record.get("kanun_full", ""),
        f"Madde {record.get('madde_no', '')}",
        record.get("baslik") or "",
        record.get("text", ""),
    ]
    return " ".join(p for p in parts if p).strip()


# ---- BM25 ------------------------------------------------------------------

def build_bm25(records: list[dict], output_dir: Path):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit("Install: pip install rank-bm25")

    log.info("Tokenizing corpus for BM25 ...")
    tokenized = [tokenize(doc_text_for_indexing(r)) for r in records]

    log.info("Fitting BM25Okapi ...")
    bm25 = BM25Okapi(tokenized)

    output_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = output_dir / "bm25.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    log.info("BM25 saved: %s", bm25_path)


# ---- Dense embeddings + FAISS ----------------------------------------------

def build_faiss(records: list[dict], output_dir: Path, model_name: str = DEFAULT_EMBED_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        raise SystemExit("Install: pip install sentence-transformers faiss-cpu")

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    # E5 expects "passage: " prefix for documents
    needs_prefix = "e5" in model_name.lower()
    texts = [
        ("passage: " + doc_text_for_indexing(r) if needs_prefix else doc_text_for_indexing(r))
        for r in records
    ]

    log.info("Encoding %d documents (batch=%d) ...", len(texts), EMBED_BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    log.info("Embeddings shape: %s", embeddings.shape)

    # Cosine similarity via inner product on normalized vectors
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    safe_name = model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = output_dir / f"faiss_{safe_name}.index"
    emb_path = output_dir / f"embeddings_{safe_name}.npy"
    faiss.write_index(index, str(faiss_path))
    np.save(emb_path, embeddings)
    log.info("FAISS saved: %s", faiss_path)
    log.info("Embeddings saved: %s", emb_path)


# ---- Metadata mapping ------------------------------------------------------

def save_metadata(records: list[dict], output_dir: Path):
    """Save the doc metadata in row order so index positions map back."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for r in records:
            # Strip the big "fikralar" list to keep meta small
            slim = {k: r[k] for k in ("doc_id", "kanun_short", "kanun_full",
                                     "kanun_no", "madde_no", "madde_type",
                                     "baslik", "url") if k in r}
            slim["text"] = r["text"]
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    log.info("Metadata saved: %s", meta_path)


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus/mevzuat_priority.jsonl")
    parser.add_argument("--output-dir", default="data/index")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--skip-dense", action="store_true", help="BM25 only (faster, no GPU needed)")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / args.corpus
    output_dir = repo_root / args.output_dir

    records = load_corpus(corpus_path)
    save_metadata(records, output_dir)

    if not args.skip_bm25:
        build_bm25(records, output_dir)

    if not args.skip_dense:
        build_faiss(records, output_dir, model_name=args.embed_model)

    log.info("Done. Index dir: %s", output_dir)


if __name__ == "__main__":
    main()
