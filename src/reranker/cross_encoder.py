"""
Cross-encoder reranker wrapper.

Supports three model flavors via a single API:
  1. **v2 (LoRA)** — BGE-reranker-v2-m3 + PEFT LoRA adapter (our fine-tune)
  2. **v1 (full FT)** — XLM-R full-finetuned cross-encoder (legacy)
  3. **pretrained** — Any HF cross-encoder (e.g. base BGE-reranker, no FT) for baseline

Usage:
    rr = Reranker.load_v2("data/models/bge-reranker-tr-legal-v2")
    rr = Reranker.load_v1("data/models/xlmr-reranker-tr-legal")
    rr = Reranker.load_pretrained("BAAI/bge-reranker-v2-m3")

    # Rerank hits from a first-stage retriever:
    new_hits = rr.rerank(query, hits, top_k=10)

Integration with src.pipeline.rag.RAGPipeline:
    pipeline.reranker = Reranker.load_v2(...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from src.retrieval.retriever import Hit

log = logging.getLogger("reranker")

Flavor = Literal["v2_lora", "v1_full", "pretrained"]


class Reranker:
    """Unified cross-encoder reranker, supports LoRA adapters and full-FT models."""

    def __init__(self, ce_model, flavor: Flavor, name: str):
        self._ce = ce_model
        self.flavor = flavor
        self.name = name

    @classmethod
    def load_v2(
        cls,
        model_dir: str | Path,
        base_model: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> "Reranker":
        """Load BGE-reranker + LoRA adapter (our v2 fine-tune)."""
        try:
            from sentence_transformers import CrossEncoder
            from peft import PeftModel
        except ImportError:
            raise SystemExit("Install: pip install sentence-transformers peft")

        model_dir = Path(model_dir)
        adapter_dir = model_dir / "lora_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")

        dev = device or ("cuda" if _cuda_available() else "cpu")
        log.info("Loading v2 reranker: base=%s + adapter=%s (device=%s)",
                 base_model, adapter_dir, dev)
        ce = CrossEncoder(base_model, num_labels=1, max_length=max_length, device=dev)
        ce.model = PeftModel.from_pretrained(ce.model, str(adapter_dir))
        ce.model.eval()
        return cls(ce, flavor="v2_lora", name=f"v2:{base_model}+lora")

    @classmethod
    def load_v1(
        cls,
        model_dir: str | Path,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> "Reranker":
        """Load full-finetuned cross-encoder (xlmr-reranker-tr-legal style)."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise SystemExit("Install: pip install sentence-transformers")

        dev = device or ("cuda" if _cuda_available() else "cpu")
        log.info("Loading v1 reranker: %s (device=%s)", model_dir, dev)
        ce = CrossEncoder(str(model_dir), num_labels=1, max_length=max_length, device=dev)
        return cls(ce, flavor="v1_full", name=f"v1:{Path(model_dir).name}")

    @classmethod
    def load_pretrained(
        cls,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> "Reranker":
        """Load a pretrained cross-encoder without any fine-tuning (zero-shot baseline)."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise SystemExit("Install: pip install sentence-transformers")

        dev = device or ("cuda" if _cuda_available() else "cpu")
        log.info("Loading pretrained reranker: %s (device=%s)", model_name, dev)
        ce = CrossEncoder(model_name, num_labels=1, max_length=max_length, device=dev)
        return cls(ce, flavor="pretrained", name=f"pretrained:{model_name}")

    def score(self, query: str, passages: list[str], batch_size: int = 32) -> np.ndarray:
        """Score (query, passage) pairs. Returns float array of length len(passages)."""
        if not passages:
            return np.array([], dtype=np.float32)
        pairs = [[query, p] for p in passages]
        if self.flavor == "v2_lora":
            return self._score_lora_pairs(pairs, batch_size)
        scores = self._ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def _score_lora_pairs(self, pairs: list[list[str]], batch_size: int) -> np.ndarray:
        """Run PEFT-wrapped rerankers without CrossEncoder's version-specific forward path."""
        import torch

        model = self._ce.model
        tokenizer = self._ce.tokenizer
        device = next(model.parameters()).device
        max_length = getattr(self._ce, "max_length", 512)
        chunks: list[np.ndarray] = []

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            features = tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            features = {name: value.to(device) for name, value in features.items()}
            with torch.inference_mode():
                logits = model(**features, return_dict=True).logits
            chunks.append(logits.reshape(-1).float().cpu().numpy())

        return np.concatenate(chunks).astype(np.float32, copy=False)

    def rerank(self, query: str, hits: list[Hit], top_k: int = 10,
               batch_size: int = 32) -> list[Hit]:
        """
        Re-score the given hits and return the top-k re-ordered.
        New Hit.score = reranker score; Hit.rank reset 0..k-1.
        """
        if not hits:
            return []
        scores = self.score(query, [h.text for h in hits], batch_size=batch_size)
        order = np.argsort(-scores)[:top_k]
        out: list[Hit] = []
        for new_rank, idx in enumerate(order):
            h = hits[int(idx)]
            out.append(Hit(
                doc_id=h.doc_id, score=float(scores[int(idx)]),
                kanun_short=h.kanun_short, kanun_full=h.kanun_full,
                madde_no=h.madde_no, baslik=h.baslik,
                text=h.text, url=h.url, rank=new_rank,
            ))
        return out


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---- CLI smoke test --------------------------------------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=["v2", "v1", "pretrained"], default="v2")
    parser.add_argument("--model-dir", default="data/models/bge-reranker-tr-legal-v2",
                        help="For v2/v1: local dir. For pretrained: HF model id.")
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3",
                        help="Base model for v2 LoRA (ignored otherwise).")
    parser.add_argument("--query", default="Cumhurbaşkanına hakaret cezası nedir?")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    model_path = repo_root / args.model_dir if args.flavor != "pretrained" else args.model_dir

    if args.flavor == "v2":
        rr = Reranker.load_v2(model_path, base_model=args.base_model)
    elif args.flavor == "v1":
        rr = Reranker.load_v1(model_path)
    else:
        rr = Reranker.load_pretrained(args.model_dir)

    candidates = [
        "Cumhurbaşkanına hakaret eden kişi bir yıldan dört yıla kadar hapis cezası alır.",
        "Tacir her türlü borcu için iflasa tabidir.",
        "Bir insanı kasten öldüren kişi müebbet hapis cezası alır.",
        "Hakaret suçunun cezası üç aydan iki yıla kadar hapis veya adli para cezasıdır.",
    ]
    scores = rr.score(args.query, candidates)
    print(f"\n=== Reranker: {rr.name} ===")
    print(f"Query: {args.query}\n")
    for c, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
        print(f"{s:+.4f}  {c[:80]}")


if __name__ == "__main__":
    _cli()
