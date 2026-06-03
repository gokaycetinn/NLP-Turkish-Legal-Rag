<div align="center">

# HukukRAG: Optimized RAG Pipeline for Turkish Legal QA

**CENG493 — Natural Language Processing Term Project (May 2026)**

Open-source, end-to-end RAG stack fine-tuned for Turkish statutory law with reliable citations and measurable gains.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Model: Trendyol LLM](https://img.shields.io/badge/Trendyol_LLM-7B-000000.svg?style=flat-square&logo=huggingface)](https://huggingface.co/Trendyol/Trendyol-LLM-7B-chat-v4.1.0)
[![Kaggle: Models and Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-system)

---

**Authors:** Hasan Emre Usta · Omer Altintas · Kayra Dalcik · Gokay Cetinakdogan

</div>

> [!IMPORTANT]
> Accurate legal citations are critical. General-purpose LLMs are often unfamiliar with Turkish legal terminology, do not follow citation formats like [TCK m.299], and lack access to statutory text. This project fine-tunes the three core components—Embedding, Reranker, and LLM—on Turkish law to deliver practical, citation-aware legal QA.

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Quick Demo](#quick-demo)
- [Performance](#performance)
- [Data Sources](#data-sources)
- [Reproducibility](#reproducibility)
- [Key Links](#key-links)

---

## Project Overview
We build a production-grade Retrieval-Augmented Generation (RAG) system over **41,973 legal articles** from mevzuat.gov.tr. The pipeline is optimized end-to-end with domain-specific fine-tuning:
1. **Hybrid Retrieval:** Sparse BM25 plus dense e5-FT with RRF fusion.
2. **Cross-Encoder Reranker:** Fine-tuned BGE-v2 for precise reordering.
3. **LLM Generator:** Trendyol-7B chat model with QLoRA SFT for reliable citations.

**Result:** Compared to a BM25 + base LLM baseline, the system achieves **Citation F1 +0.414 (7.8x)** and **Recall@10 +0.323** on the dev benchmark.

---

## System Architecture

```mermaid
flowchart TD
    Q[User Query] --> R[Hybrid Retrieval\n(BM25 + e5-FT)]
    R -->|Top-50| RR[Cross-Encoder Reranker\n(BGE-v2 FT)]
    RR -->|Top-5| LLM[LLM Generator\n(Trendyol-7B SFT)]
    LLM --> A["... one to three years imprisonment [TCK m.299]"]
```

<details>
<summary><b>Model components and adaptations</b></summary>

| Component | Base model | Adaptation |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-large` | LoRA, 15,130 hard-negative triplets |
| Reranker | `BAAI/bge-reranker-v2-m3` | LoRA, listwise contrastive loss |
| Generator | `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` | QLoRA 4-bit SFT, Unsloth, loss 0.60 -> 0.33 |
| Judge (eval) | `Anthropic Claude Haiku 4.5` | RAGAS-style claim decomposition |

</details>

---

## Quick Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_demo.ipynb)

Open the notebook, run all cells, and use the Gradio public URL to query the system.

---

## Performance

Main comparison on the **122-question dev set** (A1 vs A5):

| Metric | A1 Base RAG | A5 FT-RAG | Delta |
|---|:---:|:---:|:---:|
| Recall@5 | 0.414 | 0.709 | +0.295 |
| Recall@10 | 0.480 | 0.803 | +0.323 |
| MRR@10 | 0.362 | 0.629 | +0.267 |
| Answer F1 | 0.227 | 0.359 | +0.132 |
| Citation F1 | 0.061 | 0.475 | +0.414 |
| Faithfulness | 0.615 | 0.704 | +0.089 |

<details>
<summary><b>Ablation contributions (8-cell design)</b></summary>

| Comparison (isolated factor) | Delta R@10 | Delta Citation F1 |
|---|:---:|:---:|
| A5 vs A5a (LLM SFT removed) | 0.000 | -0.340 |
| A5 vs A5b (Embedding FT removed) | -0.057 | -0.021 |
| A5 vs A5c (Reranker FT removed) | -0.024 | -0.019 |

</details>

---

## Data Sources

- **Statutory corpus:** 41,973 articles from mevzuat.gov.tr, 1,054 laws.
- **SFT data:** 3,026 synthetic QA pairs from Claude Haiku + ~40K HuggingFace legal QA.
- **Hard-negative mining:** 15,130 triplets from BM25 top-20.
- **Gold benchmark:** 178 manually authored questions (122 dev / 27 test), never used for training.

---

## Reproducibility

### 1) Install dependencies
```bash
git clone https://github.com/hasanemreusta/turkish-legal-rag
cd turkish-legal-rag
pip install -r requirements.txt
```

### 2) Download model weights and indices
```bash
pip install kaggle
# Ensure ~/.kaggle/kaggle.json is configured
kaggle datasets download -d hasanemreusta/turkish-legal-rag-system -p data --unzip
```

### 3) Run evaluation
**A1 Baseline (BM25 + base LLM):**
```bash
python -m src.eval.run_eval \
  --test data/test_set/dev_full.jsonl \
  --index-dir data/index_full \
  --mode bm25 \
  --llm-backend hf \
  --hf-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 \
  --out results/A1
```

**A5 Full FT-RAG:**
```bash
python -m src.eval.run_eval \
  --test data/test_set/dev_full.jsonl \
  --index-dir data/index_full \
  --embed-model data/models/e5-large-tr-legal \
  --mode hybrid --reranker v2 \
  --reranker-dir data/models/bge-reranker-tr-legal-v2 \
  --llm-backend hf \
  --hf-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 \
  --adapter-path data/models/llm_adapter \
  --out results/A5
```

Full ablation notebook: `notebooks/04_ablation_eval.ipynb`

---

## Key Links

| Material | Link |
|---|---|
| Model and index dataset | https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-system |
| Fine-tune dataset | https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-finetune |
| Demo notebook | notebooks/colab_demo.ipynb |
| Technical report | report/REPORT.md |
| Ablation results | results/ablation_summary.csv |

---
Prepared for CENG493 (Natural Language Processing).
