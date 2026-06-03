"""
Evaluation metrics for the Turkish Legal RAG system.

Provides:
  - Retrieval metrics: Recall@k, MRR@k, nDCG@k (per-query and macro-averaged)
  - Answer-quality metrics: Exact Match, token-F1, BLEU, ROUGE-L (Turkish-aware tokenization)
  - Citation metrics: precision/recall over [KısaAd m.X] citations in the answer string

All functions take simple Python data — no framework lock-in — so they're easy to call
from notebooks or the run_eval CLI.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

# ---- Tokenization (matches retrieval/build_index tokenize spirit) ----------

_TR_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})


def normalize_tr(s: str) -> str:
    """Turkish-aware lowercase + NFKC + collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TR_LOWER_MAP).lower()
    return re.sub(r"\s+", " ", s).strip()


_TOKEN_RE = re.compile(r"[\wçğıöşüÇĞİÖŞÜ]+", re.UNICODE)


def tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_tr(s))


# ---- Retrieval metrics -----------------------------------------------------

def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    topk = retrieved_ids[:k]
    return len(set(topk) & gold_ids) / len(gold_ids)


def mrr_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---- Text overlap ----------------------------------------------------------

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_tr(pred) == normalize_tr(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_toks = tokens(pred)
    gold_toks = tokens(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_toks)
    r = overlap / len(gold_toks)
    return 2 * p * r / (p + r)


# ---- BLEU (corpus-level smoothing, up to 4-gram) ---------------------------

def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(pred: str, gold: str, max_n: int = 4) -> float:
    p_toks = tokens(pred)
    g_toks = tokens(gold)
    if not p_toks or not g_toks:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        p_ng = _ngrams(p_toks, n)
        g_ng = _ngrams(g_toks, n)
        if sum(p_ng.values()) == 0:
            precisions.append(0.0)
            continue
        overlap = sum((p_ng & g_ng).values())
        # add-1 smoothing
        precisions.append((overlap + 1) / (sum(p_ng.values()) + 1))
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = math.exp(1 - len(g_toks) / len(p_toks)) if len(p_toks) < len(g_toks) else 1.0
    return bp * math.exp(log_avg)


# ---- ROUGE-L (LCS-based F-measure) -----------------------------------------

def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str, beta: float = 1.0) -> float:
    p_toks = tokens(pred)
    g_toks = tokens(gold)
    if not p_toks or not g_toks:
        return 0.0
    lcs = _lcs_len(p_toks, g_toks)
    if lcs == 0:
        return 0.0
    p = lcs / len(p_toks)
    r = lcs / len(g_toks)
    return (1 + beta ** 2) * p * r / (r + beta ** 2 * p)


# ---- Citation parsing ------------------------------------------------------

CITATION_RE = re.compile(r"\[\s*([A-Za-zÇĞİÖŞÜçğıöşü0-9]+)\s+m\.?\s*([0-9A-Za-zçğıöşü/\-]+)\s*\]")


def extract_citations(text: str) -> list[tuple[str, str]]:
    """Return list of (kanun_short, madde_no) tuples parsed from the answer."""
    out = []
    for m in CITATION_RE.finditer(text or ""):
        short = normalize_tr(m.group(1))
        no = m.group(2).strip().lower()
        out.append((short, no))
    return out


def citation_scores(pred_text: str, gold_citations: set[tuple[str, str]]) -> dict:
    """Precision/recall/F1 of citations in the predicted answer vs gold set."""
    pred = set((normalize_tr(s), n.lower()) for s, n in extract_citations(pred_text))
    gold = set((normalize_tr(s), n.lower()) for s, n in gold_citations)
    if not pred and not gold:
        return {"p": 1.0, "r": 1.0, "f1": 1.0, "n_pred": 0, "n_gold": 0}
    if not pred:
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "n_pred": 0, "n_gold": len(gold)}
    if not gold:
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "n_pred": len(pred), "n_gold": 0}
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"p": p, "r": r, "f1": f1, "n_pred": len(pred), "n_gold": len(gold)}


# ---- Aggregation -----------------------------------------------------------

def macro_avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
