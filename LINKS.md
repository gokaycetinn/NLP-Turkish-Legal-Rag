# Proje Linkleri — CENG493 Dönem Projesi

## Kaggle Datasets

| Dataset | İçerik | Link |
|---|---|---|
| `turkish-legal-rag-system` | Fine-tuned embedder, reranker LoRA, LLM LoRA adapter, BM25 + FAISS index, corpus | https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-system |
| `turkish-legal-rag-finetune` | Sentetik QA, hard-negative triplet, HuggingFace SFT karışımı | https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-finetune |

## Kaggle Notebooks

| Notebook | Amaç | Link |
|---|---|---|
| `01_embedding_finetune` | multilingual-e5-large LoRA fine-tuning | https://www.kaggle.com/code/hasanemreusta/01-embedding-finetune |
| `02_reranker_finetune` | BGE-reranker-v2-m3 LoRA fine-tuning | https://www.kaggle.com/code/hasanemreusta/02-reranker-finetune |
| `03_llm_sft` | Trendyol-7B QLoRA SFT (Unsloth) | https://www.kaggle.com/code/hasanemreusta/03-llm-sft |
| `04_ablation_eval` | 8-hücreli ablasyon değerlendirmesi | https://www.kaggle.com/code/hasanemreusta/04-ablation-eval |
| `05_build_ft_index` | e5-FT FAISS index oluşturma | https://www.kaggle.com/code/hasanemreusta/05-build-ft-index |
| `06_faithfulness_judge` | Faithfulness değerlendirmesi | https://www.kaggle.com/code/hasanemreusta/06-faithfulness-judge |

## Demo (Colab + Gradio)

`notebooks/colab_demo.ipynb` dosyasını Google Colab'a yükleyin ve "Run All" ile çalıştırın.  
Fine-tuned Trendyol-7B generator ile public Gradio URL üretilir.

## Veri Kaynağı

- **Corpus:** https://www.mevzuat.gov.tr (Türkiye Cumhuriyeti Resmî Mevzuat)
