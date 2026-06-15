"""
End-to-end RAG pipeline for Turkish legal QA.

Components:
  - Retriever (BM25 / dense / hybrid)
  - Optional cross-encoder reranker
  - LLM generator (HuggingFace causal LM or Gemini API)

Output: answer string + list of retrieved sources (Hit) for citation tracking.

Usage:
    rag = RAGPipeline.build(
        index_dir="data/index",
        embed_model="intfloat/multilingual-e5-large",
        llm_backend="gemini",   # or "hf"
        llm_model="gemini-2.5-pro",
    )
    out = rag.answer("Cumhurbaşkanına hakaret suçunun cezası nedir?")
    print(out["answer"])
    print(out["sources"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.retrieval.retriever import Retriever, Hit

log = logging.getLogger("rag")

LLMBackend = Literal["hf", "gemini", "dummy"]

SYSTEM_PROMPT = """Sen Türk hukuku alanında uzman bir asistansın. Sana sorulan soruyu,
SADECE aşağıda verilen kanun maddelerine dayanarak cevapla.

Kurallar:
1. Cevap mutlaka verilen maddelerden çıkarılabilir olmalı; uydurma bilgi verme.
2. Her iddianın sonunda atıf yap: "[KısaAd m.MaddeNo]" formatında (örn. "[TCK m.299]").
3. Eğer verilen maddelerde cevap yoksa açıkça "Verilen kaynaklarda bu sorunun cevabı bulunmamaktadır." de.
4. Net, kısa ve hukuki üslupta cevap ver. Madde metnini olduğu gibi tekrarlama; özetle ve uygula.
"""

USER_PROMPT_TEMPLATE = """Soru: {question}

Kaynak Maddeler:
{context}

Cevap:"""


def format_context(hits: list[Hit], max_chars: int = 6000) -> str:
    """Format retrieved hits as numbered context blocks with citation hints."""
    parts = []
    used = 0
    for h in hits:
        block = (
            f"[{h.kanun_short} m.{h.madde_no}]"
            + (f" ({h.baslik})" if h.baslik else "")
            + f"\n{h.text}"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


# ---- LLM backends ----------------------------------------------------------

class BaseLLM:
    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError


class DummyLLM(BaseLLM):
    """Returns the first context block — for pipeline smoke-testing without GPU/API."""
    def generate(self, system, user, max_new_tokens=512):
        ctx_start = user.find("Kaynak Maddeler:")
        ctx = user[ctx_start:] if ctx_start >= 0 else user
        first_block = ctx.split("\n\n")[1] if "\n\n" in ctx else ctx[:300]
        return f"(DUMMY) Verilen kaynaklara göre: {first_block[:300]}..."


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise SystemExit("Install: pip install google-generativeai")
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.environ.get("GOOGLE_API_KEY")
            except ImportError:
                pass
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)

    def generate(self, system, user, max_new_tokens=512):
        resp = self.model.generate_content(
            user,
            generation_config={"max_output_tokens": max_new_tokens, "temperature": 0.1},
        )
        return resp.text.strip()


class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_name: str, quantize_4bit: bool = True,
                 adapter_path: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise SystemExit("Install: pip install transformers torch")

        log.info("Loading HF model: %s (4bit=%s)", model_name, quantize_4bit)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if quantize_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                log.warning("bitsandbytes not available; loading in fp16")
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError:
                raise SystemExit("Install: pip install peft")
            log.info("Loading PEFT adapter: %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)

        self.tokenizer = tokenizer
        self.model = model

    def generate(self, system, user, max_new_tokens=512):
        # Chat format — works for Llama-3 / Trendyol-Llama instruct models
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=3072).to(self.model.device)
        eos_ids = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tokenizer.unk_token_id and im_end not in eos_ids:
            eos_ids.append(im_end)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            eos_token_id=eos_ids if eos_ids else None,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---- Pipeline --------------------------------------------------------------

@dataclass
class RAGPipeline:
    retriever: Retriever
    llm: BaseLLM
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    reranker: Optional[object] = None  # placeholder for cross-encoder
    candidate_k: int = 30

    @classmethod
    def build(cls, index_dir: str, embed_model: str = "intfloat/multilingual-e5-large",
              llm_backend: LLMBackend = "gemini", llm_model: str = "gemini-2.5-pro",
              top_k: int = 5, retrieval_mode: str = "hybrid",
              adapter_path: Optional[str] = None,
              reranker: Optional[object] = None,
              candidate_k: int = 50,
              quantize_4bit: bool = True) -> "RAGPipeline":
        retriever = Retriever.load(index_dir, embed_model=embed_model,
                                   load_dense=(retrieval_mode != "bm25"),
                                   load_bm25=(retrieval_mode != "dense"))
        if llm_backend == "gemini":
            llm = GeminiLLM(llm_model)
        elif llm_backend == "hf":
            llm = HuggingFaceLLM(
                llm_model,
                quantize_4bit=quantize_4bit,
                adapter_path=adapter_path,
            )
        elif llm_backend == "dummy":
            llm = DummyLLM()
        else:
            raise ValueError(f"Unknown backend: {llm_backend}")
        return cls(retriever=retriever, llm=llm, top_k=top_k,
                   retrieval_mode=retrieval_mode, reranker=reranker,
                   candidate_k=candidate_k)

    def answer(self, question: str) -> dict:
        candidate_k = self.candidate_k if self.reranker else self.top_k
        hits = self.retriever.search(question, top_k=candidate_k, mode=self.retrieval_mode)

        if self.reranker:
            hits = self.reranker.rerank(question, hits, top_k=self.top_k)
        else:
            hits = hits[: self.top_k]

        context = format_context(hits)
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [h.to_dict() for h in hits],
            "context_used_chars": len(context),
        }


# ---- CLI -------------------------------------------------------------------

def _cli():
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Soru (Türkçe)")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--embed-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--llm-backend", choices=["hf", "gemini", "dummy"], default="dummy")
    parser.add_argument("--llm-model", default="gemini-2.5-pro")
    parser.add_argument("--adapter-path", default=None,
                        help="PEFT/LoRA adapter dir (e.g. data/models/llm_adapter). Only for --llm-backend hf.")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reranker", choices=["none", "v1", "v2", "pretrained"], default="none")
    parser.add_argument("--reranker-dir", default="data/models/bge-reranker-tr-legal-v2",
                        help="For v2/v1: local model dir. For pretrained: HF id.")
    parser.add_argument("--candidate-k", type=int, default=50,
                        help="First-stage top-k when reranker is used.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]

    reranker = None
    if args.reranker != "none":
        from src.reranker import Reranker
        rr_path = repo_root / args.reranker_dir if args.reranker != "pretrained" else args.reranker_dir
        if args.reranker == "v2":
            reranker = Reranker.load_v2(rr_path)
        elif args.reranker == "v1":
            reranker = Reranker.load_v1(rr_path)
        else:
            reranker = Reranker.load_pretrained(args.reranker_dir)

    adapter_path = None
    if args.adapter_path:
        ap = Path(args.adapter_path)
        adapter_path = str(ap if ap.is_absolute() else repo_root / ap)

    rag = RAGPipeline.build(
        index_dir=str(repo_root / args.index_dir),
        embed_model=args.embed_model,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        top_k=args.top_k,
        retrieval_mode=args.mode,
        reranker=reranker,
        candidate_k=args.candidate_k,
        adapter_path=adapter_path,
    )
    out = rag.answer(args.question)
    print("\n=== ANSWER ===")
    print(out["answer"])
    print("\n=== SOURCES ===")
    for s in out["sources"]:
        print(f"  [{s['kanun_short']} m.{s['madde_no']}] score={s['score']:.4f}")


if __name__ == "__main__":
    _cli()
