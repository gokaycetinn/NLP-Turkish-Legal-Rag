"""
Generate synthetic Turkish legal QA pairs from the mevzuat corpus.

For each madde we ask Gemini to produce:
  - 1-2 natural Turkish legal questions answerable from that madde
  - A grounded answer that cites [Kanun m.X]
  - A "difficulty" hint (easy lookup / medium reasoning / hard interpretation)

Output: data/finetune/synthetic_qa.jsonl with fields:
  question, answer, source_doc_id, kanun_short, kanun_full, madde_no, difficulty

This data feeds:
  - LLM SFT (citation-aware training)
  - Embedding contrastive training (positive pairs question↔madde)

Run:
    export GOOGLE_API_KEY=...   (or set in .env)
    python -m src.ingest.generate_synthetic_qa --input data/corpus/mevzuat_priority.jsonl --limit 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MODEL_NAME = "gemini-2.0-flash"  # default Gemini model
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq free-tier model
ANTHROPIC_MODEL_NAME = "claude-haiku-4-5-20251001"  # paid: $1/M in, $5/M out

PROMPT_TEMPLATE = """Aşağıda bir Türk kanun maddesinin tam metni verilmiştir. Bu maddeye dayanarak
TAM OLARAK {n_questions} adet doğal, gerçekçi hukuki soru üret. Her birine madde metninden
çıkarılabilen, gerekçeli ve atıflı bir cevap yaz.

ZORUNLU ZORLUK DAĞILIMI ({n_questions} soru için):
- 1. soru: "lookup" — maddeden doğrudan okunabilen olgusal soru (ör: "X süresi nedir?", "Y'nin cezası nedir?")
- 2. soru: "reasoning" — maddeyi yorumlama, çıkarım yapma, somut bir senaryoya uygulama (ör: "A kişisi X yaparsa ne olur? Maddeye göre durumu değerlendiriniz")
- 3. soru: "edge" — istisna, sınır durumu, karşıt durum veya muafiyet hakkında (ör: "Hangi durumlarda bu kural uygulanmaz?", "X koşulu sağlanmazsa sonuç ne olur?")

Diğer kurallar:
- Sorular Türkçe, doğal ve avukat/vatandaşın gerçek hayatta sorabileceği gibi olmalı. Garip, yapay cümle yapılarından kaçın.
- Cevaplar SADECE verilen madde metnindeki bilgiye dayanmalı (uydurma yok).
- Her cevap en az 1-2 cümle olmalı, sebep/koşul belirtilmeli — tek kelime cevap KABUL EDİLMEZ.
- Her cevabın SONUNDA "[{kanun_short} m.{madde_no}]" formatında atıf bulunmalı (köşeli parantez içinde, kısa ad ve madde no).
- "difficulty" alanı tam olarak şu üçünden biri olmalı: "lookup", "reasoning", "edge".

Madde Bilgisi:
- Kanun: {kanun_full} ({kanun_short})
- Madde No: {madde_no}
- Başlık: {baslik}

Madde Metni:
\"\"\"
{madde_text}
\"\"\"

ÇIKTI: Sadece geçerli JSON dizisi, hiçbir açıklama veya markdown ekleme. Tam olarak {n_questions} öğe içersin:
[
  {{"question": "...", "answer": "... [{kanun_short} m.{madde_no}]", "difficulty": "lookup"}},
  {{"question": "...", "answer": "... [{kanun_short} m.{madde_no}]", "difficulty": "reasoning"}},
  {{"question": "...", "answer": "... [{kanun_short} m.{madde_no}]", "difficulty": "edge"}}
]"""


log = logging.getLogger("synthetic_qa")


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _setup_gemini():
    try:
        import google.generativeai as genai
    except ImportError:
        raise SystemExit("Install: pip install google-generativeai python-dotenv")
    _load_env()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey "
            "then add to .env"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


class _GroqAdapter:
    """Wraps a Groq client to mimic Gemini's `.generate_content(prompt).text` API
    so the rest of the generator code path stays unchanged."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_content(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content

        class _R:
            pass

        r = _R()
        r.text = text
        return r


def _setup_groq():
    try:
        from groq import Groq
    except ImportError:
        raise SystemExit("Install: pip install groq")
    _load_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "then add to .env as: GROQ_API_KEY=gsk_..."
        )
    return _GroqAdapter(Groq(api_key=api_key), GROQ_MODEL_NAME)


class _AnthropicAdapter:
    """Wraps Anthropic client to mimic Gemini's `.generate_content(prompt).text` API."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def generate_content(self, prompt: str):
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=1500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""

        class _R:
            pass

        r = _R()
        r.text = text
        return r


def _setup_anthropic():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("Install: pip install anthropic")
    _load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Get a key at https://console.anthropic.com/settings/keys "
            "then add to .env as: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return _AnthropicAdapter(anthropic.Anthropic(api_key=api_key), ANTHROPIC_MODEL_NAME)


def iter_corpus(corpus_path: Path, min_text_len: int = 100, max_text_len: int = 4000) -> Iterator[dict]:
    """Yield maddeler from a JSONL corpus, filtered by text length."""
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r.get("text", "")
            if min_text_len <= len(t) <= max_text_len:
                yield r


def stratified_sample(records: list[dict], total: int, key_fn) -> list[dict]:
    """Take `total` records, balanced by `key_fn(record)`. Records should be pre-shuffled."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    per_group = total // len(groups) if groups else 0
    selected, leftovers = [], []
    for g in groups.values():
        selected.extend(g[:per_group])
        leftovers.extend(g[per_group:])
    random.shuffle(leftovers)
    selected.extend(leftovers[: max(0, total - len(selected))])
    return selected


def _parse_response(text: str) -> list[dict]:
    """Robustly extract a JSON array from the model's response."""
    text = text.strip()
    # Strip markdown fences if any
    if text.startswith("```"):
        text = text.split("```", 2)[-2] if text.count("```") >= 2 else text
        text = text.lstrip("json").strip()
    # Find first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def generate_for_madde(model, record: dict, n_questions: int = 2) -> list[dict]:
    """Generate QA pairs for a single madde record."""
    prompt = PROMPT_TEMPLATE.format(
        n_questions=n_questions,
        kanun_short=record["kanun_short"],
        kanun_full=record["kanun_full"],
        madde_no=record["madde_no"],
        baslik=record.get("baslik") or "(başlıksız)",
        madde_text=record["text"][:3500],
    )
    try:
        resp = model.generate_content(prompt)
        items = _parse_response(resp.text)
    except Exception as e:
        log.warning("Gemini call failed for %s m.%s: %s", record["kanun_short"], record["madde_no"], e)
        return []

    out = []
    for it in items:
        if not isinstance(it, dict) or "question" not in it or "answer" not in it:
            continue
        out.append({
            "question": it["question"],
            "answer": it["answer"],
            "difficulty": it.get("difficulty", "lookup"),
            "source_doc_id": record["doc_id"],
            "kanun_short": record["kanun_short"],
            "kanun_full": record["kanun_full"],
            "madde_no": record["madde_no"],
            "madde_text": record["text"],  # keep for embedding positives
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/corpus/mevzuat_priority.jsonl")
    parser.add_argument("--output", default="data/finetune/synthetic_qa.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Max maddeler to process")
    parser.add_argument("--n-questions", type=int, default=2)
    parser.add_argument("--backend", choices=["gemini", "groq", "anthropic"], default="gemini",
                        help="LLM backend: 'gemini' (GOOGLE_API_KEY), 'groq' (GROQ_API_KEY), 'anthropic' (ANTHROPIC_API_KEY)")
    parser.add_argument("--balanced", action="store_true",
                        help="Stratified sample: equal maddeler per kanun_short (requires --limit)")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between calls (rate limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", default="0/1",
                        help="Process only shard i out of N (format: 'i/N'). "
                             "Deterministic split by doc_id hash — same record always lands in same shard.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / args.input
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(iter_corpus(input_path))
    log.info("Loaded %d maddeler from %s", len(records), input_path)

    random.seed(args.seed)
    random.shuffle(records)

    # Deterministic sharding (same doc_id → same shard regardless of order)
    try:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    except ValueError:
        raise SystemExit(f"Invalid --shard: {args.shard!r}, expected 'i/N'")
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        raise SystemExit(f"Invalid shard: {shard_i}/{shard_n}")
    if shard_n > 1:
        import hashlib
        def _bucket(doc_id: str) -> int:
            return int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) % shard_n
        before = len(records)
        records = [r for r in records if _bucket(r["doc_id"]) == shard_i]
        log.info("Shard %d/%d: %d / %d maddeler", shard_i, shard_n, len(records), before)

    if args.balanced and args.limit:
        before = len(records)
        records = stratified_sample(records, args.limit, key_fn=lambda r: r["kanun_short"])
        log.info("Balanced sample: %d / %d maddeler across %d kanunlar",
                 len(records), before, len({r["kanun_short"] for r in records}))
    elif args.limit:
        records = records[: args.limit]

    if args.backend == "groq":
        model = _setup_groq()
        model_label = GROQ_MODEL_NAME
    elif args.backend == "anthropic":
        model = _setup_anthropic()
        model_label = ANTHROPIC_MODEL_NAME
    else:
        model = _setup_gemini()
        model_label = MODEL_NAME
    log.info("LLM backend: %s (model=%s)", args.backend, model_label)

    # Resume support: skip already-done doc_ids
    done = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["source_doc_id"])
                except Exception:
                    pass
        log.info("Resuming — already have %d unique source maddeler", len(done))

    with output_path.open("a", encoding="utf-8") as f:
        for rec in tqdm(records, desc="generating"):
            if rec["doc_id"] in done:
                continue
            pairs = generate_for_madde(model, rec, n_questions=args.n_questions)
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

    log.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
