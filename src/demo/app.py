"""
Gradio demo for the Turkish Legal RAG system.

Usage:
    python -m src.demo.app

Launches a Gradio app on a local port and (if `share=True`) creates a public
`*.gradio.live` URL valid for ~72 hours.

Environment variables (all optional):
    DEMO_INDEX_DIR        default: data/index_full
    DEMO_EMBED_MODEL      default: data/models/e5-large-tr-legal  (or HF id)
    DEMO_RERANKER_DIR     default: data/models/bge-reranker-tr-legal-v2
    DEMO_LLM_MODEL        default: Trendyol/Trendyol-LLM-7B-chat-v4.1.0
    DEMO_ADAPTER_PATH     default: data/models/llm_adapter
    DEMO_TOP_K            default: 5
    DEMO_CANDIDATE_K      default: 50
    DEMO_SHARE            default: 1 (Gradio public URL)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr

from src.pipeline.rag import RAGPipeline
from src.reranker import Reranker

log = logging.getLogger("demo")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_pipeline() -> RAGPipeline:
    log.info("Loading reranker...")
    reranker = Reranker.load_v2(_env("DEMO_RERANKER_DIR", "data/models/bge-reranker-tr-legal-v2"))

    log.info("Building RAG pipeline (this loads Trendyol-7B in 4-bit + LoRA, ~3-4 min)...")
    rag = RAGPipeline.build(
        index_dir=_env("DEMO_INDEX_DIR", "data/index_full"),
        embed_model=_env("DEMO_EMBED_MODEL", "data/models/e5-large-tr-legal"),
        llm_backend="hf",
        llm_model=_env("DEMO_LLM_MODEL", "Trendyol/Trendyol-LLM-7B-chat-v4.1.0"),
        top_k=int(_env("DEMO_TOP_K", "5")),
        retrieval_mode="hybrid",
        reranker=reranker,
        candidate_k=int(_env("DEMO_CANDIDATE_K", "50")),
        adapter_path=_env("DEMO_ADAPTER_PATH", "data/models/llm_adapter"),
    )
    log.info("Pipeline ready.")
    return rag


def make_handler(rag: RAGPipeline):
    def ask(question: str):
        question = (question or "").strip()
        if not question:
            return "Lütfen bir soru girin.", "", ""
        try:
            out = rag.answer(question)
        except Exception as e:
            log.exception("Pipeline error")
            return f"Hata: {e}", "", ""

        sources_md = []
        sources_full = []
        for i, s in enumerate(out["sources"], 1):
            head = f"**{i}. [{s['kanun_short']} m.{s['madde_no']}]** (score: {s['score']:.3f})"
            sources_md.append(head + "\n\n> " + s["text"][:300].replace("\n", " ") + "...")
            sources_full.append(
                f"### {i}. [{s['kanun_short']} m.{s['madde_no']}]"
                + (f" — {s.get('baslik', '')}" if s.get("baslik") else "")
                + f"\n\n{s['text']}\n"
            )
        return out["answer"], "\n\n".join(sources_md), "\n\n---\n\n".join(sources_full)

    return ask


def build_ui(rag: RAGPipeline) -> gr.Blocks:
    handler = make_handler(rag)

    with gr.Blocks(title="Türkçe Hukuki RAG") as demo:
        gr.Markdown(
            """
            # Türkçe Hukuki RAG Sistemi

            Citation-grounded Turkish legal question answering.

            **Pipeline:** 41,973 madde (mevzuat.gov.tr) → Hybrid retrieval (BM25 + e5-FT) → BGE-reranker-v2 (FT) → Trendyol-LLM-7B + QLoRA.

            **Kapsam:** Yalnızca birincil mevzuat (Kanun, CBK, Tüzük). Yargı kararları ve doktrin dahil değildir.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                q = gr.Textbox(
                    label="Soru",
                    placeholder="Örn: Cumhurbaşkanına hakaret suçunun cezası nedir?",
                    lines=3,
                )
                with gr.Row():
                    submit = gr.Button("Cevapla", variant="primary")
                    clear = gr.Button("Temizle")
                gr.Examples(
                    examples=[
                        "Cumhurbaşkanına hakaret suçunun cezası nedir?",
                        "Boşanma davasında nafaka koşulları nelerdir?",
                        "Tüketici hangi hallerde sözleşmeden cayabilir?",
                        "Suç işleyen çocuk hakkındaki güvenlik tedbirleri nelerdir?",
                        "İşveren işçiyi haklı sebeple hangi durumlarda derhal çıkarabilir?",
                    ],
                    inputs=q,
                )
            with gr.Column(scale=3):
                ans = gr.Textbox(label="Cevap (atıf formatında)", lines=8, show_copy_button=True)

        with gr.Tab("Kaynak Maddeler (özet)"):
            sources_summary = gr.Markdown()
        with gr.Tab("Kaynak Maddeler (tam metin)"):
            sources_full = gr.Markdown()

        submit.click(handler, inputs=q, outputs=[ans, sources_summary, sources_full])
        q.submit(handler, inputs=q, outputs=[ans, sources_summary, sources_full])
        clear.click(lambda: ("", "", "", ""), inputs=None, outputs=[q, ans, sources_summary, sources_full])

        gr.Markdown(
            """
            ---
            **Atıf formatı:** `[KısaAd m.MaddeNo]` (örn. `[TCK m.299]`, `[TMK m.598]`).

            *Bu sistem yalnızca akademik proje kapsamında bir gösterimdir; hukuki danışmanlık yerine geçmez.*
            """
        )

    return demo


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    rag = build_pipeline()
    demo = build_ui(rag)

    share = _env("DEMO_SHARE", "1") not in ("0", "false", "False", "")
    server_port = int(_env("DEMO_PORT", "7860"))
    log.info("Launching Gradio (share=%s, port=%s)...", share, server_port)
    demo.queue().launch(share=share, server_name="0.0.0.0", server_port=server_port)


if __name__ == "__main__":
    main()
