"""Stable Colab demo that avoids bitsandbytes and CUDA-specific extensions."""
from __future__ import annotations

import logging
import os

import gradio as gr

from src.pipeline.rag import RAGPipeline

log = logging.getLogger("colab-lite-demo")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_pipeline() -> RAGPipeline:
    model_name = _env("DEMO_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    log.info("Loading stable FP16 generator: %s", model_name)
    return RAGPipeline.build(
        index_dir=_env("DEMO_INDEX_DIR", "data/index_full"),
        llm_backend="hf",
        llm_model=model_name,
        top_k=int(_env("DEMO_TOP_K", "5")),
        retrieval_mode="bm25",
        reranker=None,
        adapter_path=None,
        quantize_4bit=False,
    )


def make_handler(rag: RAGPipeline):
    def ask(question: str):
        question = (question or "").strip()
        if not question:
            return "Lutfen bir soru girin.", ""

        try:
            out = rag.answer(question)
        except Exception as exc:
            log.exception("Pipeline error")
            return f"Hata: {exc}", ""

        sources = []
        for i, source in enumerate(out["sources"], 1):
            citation = f"[{source['kanun_short']} m.{source['madde_no']}]"
            preview = source["text"][:500].replace("\n", " ")
            sources.append(f"**{i}. {citation}**\n\n> {preview}...")
        return out["answer"], "\n\n".join(sources)

    return ask


def build_ui(rag: RAGPipeline) -> gr.Blocks:
    handler = make_handler(rag)
    with gr.Blocks(title="Turkce Hukuki RAG - Colab Lite") as demo:
        gr.Markdown(
            """
            # Turkce Hukuki RAG - Colab Lite

            Stabil Colab surumu: BM25 retrieval + Qwen2.5-1.5B FP16.
            Bu surum bitsandbytes, LoRA ve reranker kullanmaz.
            """
        )
        question = gr.Textbox(
            label="Soru",
            placeholder="Orn: Cumhurbaskanina hakaret sucunun cezasi nedir?",
            lines=3,
        )
        submit = gr.Button("Cevapla", variant="primary")
        answer = gr.Textbox(label="Cevap", lines=8, show_copy_button=True)
        sources = gr.Markdown(label="Kaynak maddeler")
        submit.click(handler, inputs=question, outputs=[answer, sources])
        question.submit(handler, inputs=question, outputs=[answer, sources])
        gr.Markdown("Bu akademik bir demodur; hukuki danismanlik yerine gecmez.")
    return demo


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    rag = build_pipeline()
    demo = build_ui(rag)
    port = int(_env("DEMO_PORT", "7860"))
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
