# Türk Hukuki Soru-Cevap için Optimize Edilmiş RAG Pipeline

**CENG493 — Natural Language Processing, Dönem Projesi — Mayıs 2026**

Hasan Emre Usta · Ömer Altıntaş · Kayra Dalçık · Gökay Çetinakdoğan

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-system)

---

## Hızlı Linkler

| Kaynak | Açıklama | Link |
|---|---|---|
| Model & Index Dataset | Fine-tuned embedder, reranker, LLM adapter, FAISS+BM25 index | [Kaggle ↗](https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-system) |
| Fine-tune Dataset | Sentetik QA, hard-negative triplet, HuggingFace SFT mix | [Kaggle ↗](https://www.kaggle.com/datasets/hasanemreusta/turkish-legal-rag-finetune) |
| Demo (Colab + Gradio) | `notebooks/colab_demo.ipynb` → Run All → public URL | [Colab ↗](notebooks/colab_demo.ipynb) |
| Teknik Rapor | Tam metodoloji, ablasyon analizi, hata analizi | [REPORT.md](report/REPORT.md) |
| Ablasyon Sonuçları (CSV) | 8 hücre × tüm metrikler | [ablation_summary.csv](results/ablation_summary.csv) |

---

## Özet

Bu çalışmada Türk hukuki metinler üzerinde çalışan bir Retrieval-Augmented Generation (RAG) sistemi sunulmaktadır. Sistem, mevzuat.gov.tr'den derlenen 41.973 kanun maddesini kapsayan bir corpus üzerinde hibrit retrieval, fine-tuned cross-encoder reranking ve QLoRA ile ince-ayarlı bir Türkçe LLM kullanmaktadır. 8 ablasyon hücresiyle yapılan kapsamlı deneylerde, vanilla BM25 + LLM baseline'ına kıyasla Citation F1 metriği 7.8× artış (+0.414 mutlak) ve Recall@10 +32.3 puan iyileşme elde edilmiştir.

---

## Problem ve Motivasyon

Türk hukuk sisteminde doğru kaynak alıntısı kritiktir: yanlış bir madde numarası gerçek hukuki sonuçlar doğurabilir. Mevcut genel amaçlı LLM'ler (1) Türkçe hukuk terminolojisine yabancıdır, (2) `[TCK m.299]` formatında alıntı üretmez, (3) mevzuat metinlerine erişimi yoktur. Bu çalışma, üç bileşeni de (embedding, reranker, LLM) hukuk alanına adapte eden uçtan uca bir pipeline önerir.

---

## Sistem Mimarisi

```
Soru
  ↓
Hibrit Retrieval  (BM25 + e5-FT, RRF top-50)
  ↓
Cross-Encoder Reranker  (BGE-v2 FT, top-5)
  ↓
LLM Generator  (Trendyol-7B-chat + QLoRA SFT)
  ↓
Alıntılı cevap: "... bir yıldan dört yıla kadar hapis cezası alır. [TCK m.299]"
```

### Bileşenler

| Bileşen | Base model | Adaptasyon |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-large` | LoRA, 15.130 hard-negative triplet |
| Reranker | `BAAI/bge-reranker-v2-m3` | LoRA, listwise contrastive loss |
| Generator | `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` (Qwen2) | QLoRA 4-bit SFT — Unsloth, 2h 6dk, loss 0.60→0.33 |
| Faithfulness judge | Anthropic Claude Haiku 4.5 | RAGAS-style claim decomposition (yalnızca eval) |

---

## Veri

| Kaynak | Boyut | Amaç |
|---|---|---|
| mevzuat.gov.tr (scrape) | 41.973 madde, 1.054 kanun | Retrieval corpus |
| Anthropic Haiku 4.5 (sentetik) | 3.026 QA çifti | LLM SFT |
| HuggingFace legal datasets | ~40K QA | LLM SFT mix |
| Hard-negative mining (BM25 top-20) | 15.130 triplet | Embedding FT |
| El yazımı gold benchmark | 178 soru (122 dev / 27 test) | Değerlendirme |

Gold benchmark 4 ekip üyesi tarafından yazılmış; hiçbir training aşamasına dahil edilmemiştir.

---

## 8-Hücreli Ablasyon Tasarımı

Her hücre **aynı** Trendyol-LLM-7B-chat-v4.1.0 generator'ı kullanır; sadece retrieval ve LLM adapter farklılaşır.

| Hücre | Embedding | Retrieval | Reranker | LLM | İzole eder |
|---|---|---|---|---|---|
| A1 | base e5 | BM25 | — | Trendyol vanilla | Baseline |
| A2 | base e5 | hybrid | — | Trendyol vanilla | +Hibrit retrieval |
| A3 | base e5 | hybrid | BGE pretrained | Trendyol vanilla | +Reranker |
| A4 | base e5 | hybrid | BGE-v2 FT | Trendyol vanilla | Reranker FT etkisi |
| **A5** | e5-FT | hybrid | BGE-v2 FT | Trendyol-SFT | **Tam optimize** |
| A5a | e5-FT | hybrid | BGE-v2 FT | Trendyol vanilla | LLM SFT izolasyonu |
| A5b | base e5 | hybrid | BGE-v2 FT | Trendyol-SFT | Embedding FT izolasyonu |
| A5c | e5-FT | hybrid | BGE pretrained | Trendyol-SFT | Reranker FT izolasyonu |

---

## Sonuçlar

### Ana Karşılaştırma (A1 vs A5, 122-soru dev seti)

| Metrik | A1 Base RAG | A5 FT-RAG | Δ |
|---|---|---|---|
| Recall@5 | 0.414 | 0.709 | **+0.295** |
| Recall@10 | 0.480 | 0.803 | **+0.323** |
| MRR@10 | 0.362 | 0.629 | **+0.267** |
| Answer F1 | 0.227 | 0.359 | **+0.132** |
| Citation F1 | 0.061 | 0.475 | **+0.414** |
| Faithfulness | 0.615 | 0.704 | **+0.089** |

### Bileşen Katkıları

| Karşılaştırma | İzole eder | Δ R@10 | Δ Cit-F1 |
|---|---|---|---|
| A5 vs A5a | LLM SFT | 0.000 | **+0.340** |
| A5 vs A5b | Embedding FT | +0.057 | +0.021 |
| A5 vs A5c | Reranker FT | +0.024 | +0.019 |

LLM SFT, Citation F1 kazancının büyük bölümünden sorumludur (+0.340); embedding FT retrieval'ı belirgin biçimde iyileştirir (+5.7 pp R@10); reranker FT her iki izolasyonda da pozitif etki gösterir.

### Faithfulness (Claude Haiku 4.5 judge, RAGAS-style, 8×122 soru)

| Ablasyon | Scored | Mean Faith |
|---|---|---|
| A1 | 111/122 | 0.615 |
| A2 | 106/122 | 0.603 |
| A3 | 105/122 | 0.694 |
| A4 | 107/122 | 0.705 |
| A5 | 114/122 | 0.704 |
| A5a | 110/122 | 0.727 |
| A5b | 115/122 | 0.691 |
| A5c | 113/122 | 0.707 |

---

## Yeniden Üretim

### Gereksinimler

```bash
git clone https://github.com/hasanemreusta/turkish-legal-rag
cd turkish-legal-rag
pip install -r requirements.txt
```

GPU gerekmez — retrieval, reranking ve indexing CPU/RTX 3050 (4 GB) üzerinde çalışır.
7B LLM çıkarımı için Kaggle T4 veya eşdeğeri gereklidir.

### Model Ağırlıkları (Kaggle Datasets)

```bash
pip install kaggle
# ~/.kaggle/kaggle.json yerleştirin
kaggle datasets download -d hasanemreusta/turkish-legal-rag-system -p data --unzip
```

İndirilen dataset içeriği:

```
data/
  index_full/                    # BM25 + FAISS
  corpus/mevzuat_full_normalized.jsonl
  models/
    e5-large-tr-legal/           # fine-tuned embedder
    bge-reranker-tr-legal-v2/    # reranker LoRA
    llm_adapter/                 # Trendyol QLoRA adapter
  test_set/dev_full.jsonl
```

### Ablasyon Değerlendirmesi

```bash
# A1 — Baseline RAG (BM25 + vanilla Trendyol-7B)
python -m src.eval.run_eval \
  --test data/test_set/dev_full.jsonl \
  --index-dir data/index_full \
  --mode bm25 \
  --llm-backend hf \
  --hf-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 \
  --out results/A1

# A5 — Full FT-RAG
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

Tüm 8 hücre için Kaggle notebook: `notebooks/04_ablation_eval.ipynb`

### Kendi Corpus'unuzla Değerlendirme

```bash
# 1. Index oluştur
python -m src.retrieval.build_index \
  --corpus YOUR_CORPUS.jsonl \
  --output-dir data/index_custom

# 2. Değerlendir
python -m src.eval.run_eval \
  --test YOUR_BENCHMARK.jsonl \
  --index-dir data/index_custom \
  --mode hybrid --reranker v2 \
  --out results/custom
```

Corpus JSONL şeması: `{"id": "TCK_m299", "kanun_short": "TCK", "madde_no": "299", "text": "..."}`

Benchmark JSONL şeması: `{"id": "q001", "question": "...", "answer": "...", "gold_doc_ids": ["TCK_m299"]}`

---

## Kaggle Notebooks

| Notebook | Amaç | Platform | Link |
|---|---|---|---|
| `01_embedding_finetune.ipynb` | multilingual-e5-large LoRA | Kaggle T4×2 | [↗](https://www.kaggle.com/code/hasanemreusta/01-embedding-finetune) |
| `02_reranker_finetune.ipynb` | BGE-reranker-v2-m3 LoRA | Kaggle T4×2 | [↗](https://www.kaggle.com/code/hasanemreusta/02-reranker-finetune) |
| `03_llm_sft.ipynb` | Trendyol-7B QLoRA (Unsloth) | Kaggle T4×1 | [↗](https://www.kaggle.com/code/hasanemreusta/03-llm-sft) |
| `04_ablation_eval.ipynb` | 8-hücreli ablasyon eval | Kaggle T4×1 | [↗](https://www.kaggle.com/code/hasanemreusta/04-ablation-eval) |
| `05_build_ft_index.ipynb` | e5-FT FAISS index oluşturma | Kaggle T4×1 | [↗](https://www.kaggle.com/code/hasanemreusta/05-build-ft-index) |
| `06_faithfulness_judge.ipynb` | Faithfulness değerlendirmesi (Haiku 4.5 judge) | Kaggle T4×1 | [↗](https://www.kaggle.com/code/hasanemreusta/06-faithfulness-judge) |

---

## Kısıtlamalar

- IIK, TTK, IYUK kanun kodları SFT eğitim setinde yetersiz temsil → bazı kategorilerde Cit-F1=0
- Tek-madde varsayımı: multi-hop muhakeme desteklenmiyor
- Morfolo jik stemming yok: BM25 "hakaret" ≠ "hakaretin" olarak işler
- Faithfulness judge: tek model (Haiku 4.5), çoklu hakem anlaşması ölçülmedi

---

## Lisans

MIT
