"""
LLM-as-judge faithfulness scorer (RAGAS-style) using Groq (llama-3.1-8b-instant).

For each (question, answer, retrieved_context) triple we ask the model to:
  1. Split the answer into atomic claims
  2. For each claim, decide whether it's supported by the context (SUPPORTED / PARTIAL / UNSUPPORTED)
  3. Return a faithfulness score = supported / total_claims

Output is a dict per example; aggregated by run_eval.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger("judge")

JUDGE_MODEL = "llama-3.1-8b-instant"

PROMPT = """Sen bir hukuki RAG sisteminin "faithfulness" (kaynak-bağlılık) hakemisin.
Aşağıda bir soru, modelin cevabı ve cevabın dayanması gereken KAYNAK MADDELER var.

Görevin:
1. Cevabı atomik iddialara (kısa, doğrulanabilir cümleler) ayır.
2. Her iddiayı, SADECE kaynak maddelerdeki bilgiye dayanarak değerlendir:
   - SUPPORTED: iddia tamamen kaynaktan çıkarılabilir
   - PARTIAL: kısmen destekleniyor ama eksik/uydurma kısımlar var
   - UNSUPPORTED: kaynaklardan çıkarılamıyor (hallucination)
3. Genel "faithfulness" skorunu hesapla: SUPPORTED sayısı / toplam iddia sayısı (PARTIAL=0.5 say).

SORU: {question}

CEVAP:
\"\"\"
{answer}
\"\"\"

KAYNAK MADDELER:
\"\"\"
{context}
\"\"\"

ÇIKTI: Sadece geçerli JSON, markdown veya açıklama YOK:
{{
  "claims": [
    {{"claim": "...", "verdict": "SUPPORTED|PARTIAL|UNSUPPORTED", "reason": "kısa gerekçe"}}
  ],
  "faithfulness": 0.0
}}"""


def _setup():
    try:
        import groq as _groq
    except ImportError:
        raise SystemExit("Install: pip install groq")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
        except ImportError:
            pass
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set (.env or env var)")
    return _groq.Groq(api_key=api_key)


def _parse(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


class FaithfulnessJudge:
    def __init__(self):
        self.client = _setup()

    def score(self, question: str, answer: str, context: str, sleep: float = 8.0) -> dict:
        prompt = PROMPT.format(question=question, answer=answer, context=context[:4000])
        try:
            resp = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=1200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = resp.usage
            log.debug("tokens: in=%d out=%d", usage.prompt_tokens, usage.completion_tokens)
            self._total_input_tokens = getattr(self, "_total_input_tokens", 0) + usage.prompt_tokens
            self._total_output_tokens = getattr(self, "_total_output_tokens", 0) + usage.completion_tokens
            finish_reason = resp.choices[0].finish_reason
            if finish_reason == "length":
                log.warning("Response truncated by max_tokens — JSON likely incomplete")
            raw = resp.choices[0].message.content
            log.debug("raw (first 200): %s", raw[:200])
            parsed = _parse(raw)
        except Exception as e:
            log.warning("Judge call failed: %s", e)
            parsed = None
        time.sleep(sleep)
        if not parsed or "claims" not in parsed:
            return {"faithfulness": None, "claims": [], "error": "parse_failed"}
        claims = parsed.get("claims") or []
        if not claims:
            return {"faithfulness": None, "claims": [], "error": "no_claims"}
        score = 0.0
        for c in claims:
            v = (c.get("verdict") or "").upper()
            if v == "SUPPORTED":
                score += 1.0
            elif v == "PARTIAL":
                score += 0.5
        return {"faithfulness": score / len(claims), "claims": claims}
