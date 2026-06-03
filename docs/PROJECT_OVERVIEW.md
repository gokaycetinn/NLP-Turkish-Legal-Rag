# Türkçe Hukuki RAG — Sistem Genel Bakışı

Bu doküman, sistemin iç işleyişini ve veri akışını ayrıntılı olarak açıklar. Kullanıcı odaklı özet için [README.md](../README.md) dosyasına bakın.

---

## 1. Problem ve Hedefler

mevzuat.gov.tr'den çekilmiş **41.973 kanun maddesi** üzerinde çalışan, **atıf-bilinçli (citation-aware)** bir Türkçe hukuki soru–cevap sistemi. Kullanıcının sorduğu doğal dil sorusuna karşılık sistem:

1. İlgili kanun maddelerini bulur (retrieval),
2. En alakalı olanlara odaklanır (reranker),
3. Bu maddelere dayanarak atıflı bir cevap üretir (LLM).

**Örnek çıktı:**

> **Soru:** Cumhurbaşkanına hakaret eden birine ne ceza verilir?
> **Cevap:** Cumhurbaşkanına hakaret eden kişi bir yıldan dört yıla kadar hapis cezası alır. Suç alenen işlenirse ceza altıda bir oranında artırılır. **[TCK m.299]**

### Hedefler

1. **Citation accuracy ≥ %95** — cevaptaki atıfların doğru maddeyi göstermesi.
2. **Faithfulness ≥ %85** — cevabın kaynak metinle uyumu (halüsinasyon ≤ %15).
3. **Retrieval Recall@5 ≥ %80** — doğru maddenin top-5 içinde dönmesi.
4. **Ticari API ceiling'e yaklaşmak** — açık ağırlıklı, lokal fine-tune'lu modelle.

---

## 2. Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                       KULLANICI SORUSU                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVER                                                   │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│   │  BM25    │ + │  Dense       │ = │  Hybrid (RRF)    │    │
│   │ (sparse) │   │ (e5-large)   │   │  fusion          │    │
│   └──────────┘   └──────────────┘   └──────────────────┘    │
│   Output: top-50 aday madde                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RERANKER                                                    │
│   Cross-encoder (BGE-v2-m3 + LoRA), query+passage birlikte   │
│   Output: top-5 en alakalı madde                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LLM GENERATOR                                               │
│   Trendyol-LLM-7B-chat-v4.1.0 + QLoRA SFT                    │
│   System prompt: "[KısaAd m.NO] formatında atıf yap,         │
│                   uydurma, kaynak yoksa belirt"              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                ATIFLI HUKUKI CEVAP
```

### Üç model — üç farklı amaç

| Model | Rol | Eğitim |
|---|---|---|
| `intfloat/multilingual-e5-large` | Dense vektör arama | LoRA + CachedMNRLoss (Kaggle T4×2) |
| `BAAI/bge-reranker-v2-m3` | İnce sıralama | LoRA binary cross-encoder (Kaggle T4×2) |
| `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` | Cevap üretimi | QLoRA 4-bit SFT, Unsloth (Kaggle T4×2) |

**Anthropic Haiku 4.5** yalnızca "data factory" (sentetik QA üretimi) ve "judge" (faithfulness ölçümü) rolünde kullanılır — son sistemde generator olarak yer almaz.

---

## 3. Veri Akışı ve Korpus

### Korpus oluşumu

```
mevzuat.gov.tr  →  PDF (1085 doc)  →  pdfplumber text  →  madde parser  →  JSONL
                                                            ↓
                                              kanun_short normalize
                                                            ↓
                                      data/corpus/mevzuat_full_normalized.jsonl
                                          41.973 madde, 1054 kanun
```

### Eğitim verisi katmanları

| Katman | Boyut | Kaynak | Kullanım |
|---|---|---|---|
| Sentetik QA | 3.026 çift | Anthropic Haiku 4.5 | LLM SFT (birincil), embedding positive |
| Hard negatives | 15.130 | BM25 mining | Embedding contrastive training |
| `yeniguno/turkish-law-eqa` | 21.593 | HuggingFace | LLM SFT karışım |
| `OrionCAF/turkish_law_qa_dataset` | 18.305 | HuggingFace | LLM SFT karışım |

### Gold test set

Eval için **ayrı tutulan** 178 elle yazılmış soru (final 149: dev 122 + test 27). Sentetik QA dağılımıyla disjoint — hiçbir fine-tune aşamasında kullanılmaz.

- 4 ekip üyesi tarafından yazıldı
- Hukukçu arkadaş validation
- Difficulty mix: lookup / reasoning / edge / no_answer / multi_hop

---

## 4. Bileşenler

### A. Veri Toplama (`src/ingest/`)

| Dosya | İşlev |
|---|---|
| `scrape_mevzuat.py` | mevzuat.gov.tr'den PDF indirir, pdfplumber ile text extract eder, madde parser çağırır |
| `madde_parser.py` | Kanun metnini "Madde X" başlıklarına ayırır, fıkra yapısını korur |
| `fetch_active_list.py` | mevzuat.gov.tr POST API'sinden aktif kanun listesini çeker |
| `normalize_kanun_short.py` | Generic ID'leri (örn. `T1_5237`) okunaklı kısaltmalara (TCK) çevirir |
| `download_hf_datasets.py` | HuggingFace Türkçe hukuki datasetlerini indirir |
| `generate_synthetic_qa.py` | Madde başına 3 QA çifti üretir (lookup / reasoning / edge), citation format zorunluluğu ile |

### B. Hard Negative Mining (`src/embedding/mine_hard_negatives.py`)

3.026 (question, madde) positive pair embedding fine-tune için sınırda kalır. Her positive için BM25 top-20'den positive olmayan distractor'lar çekilerek 15.130 efektif eğitim tuple elde edilir.

**Algoritma:**

1. Her QA için question'ı BM25'e ver, top-20 madde çek.
2. Doğru maddeyi filtre et, kalan ilk 5'i hard negative olarak al.
3. 5 negatif çıkmazsa, aynı kanundan rastgele madde ekle (in-domain pad).
4. Stats yaz: positive top-20'de bulundu mu? (kalite sinyali)

Kalite metrikleri: positive BM25 top-20'de bulunma oranı %92.7, ortalama positive rank 2.07.

### C. Retrieval (`src/retrieval/`)

| Dosya | İşlev |
|---|---|
| `build_index.py` | BM25Okapi (sparse) + FAISS IndexFlatIP (dense, normalized) |
| `retriever.py` | Üç mod: bm25 / dense / hybrid (RRF fusion k=60) |

**Türkçe tokenization:** lowercase + NFKC + Turkish-aware regex.

**E5 prefix:** indexlenirken `"passage: "`, sorgu sırasında `"query: "` (Multilingual E5 eğitim formatı).

**Hybrid (RRF) formülü:**

```
score(d) = Σ 1 / (k + rank_i(d))   her ranker i için
```

### D. Reranker (`src/reranker/cross_encoder.py`)

Üç flavor dispatch: v1 (XLM-R, historical), v2 (BGE-reranker-v2-m3 + LoRA, aktif), pretrained (LoRA'sız BGE).

### E. RAG Pipeline (`src/pipeline/rag.py`)

**Akış:**

1. `Retriever.search(question, top_k=30, mode='hybrid')`
2. (Opsiyonel) Reranker ile top-5'e in
3. `format_context(hits)` — atıflı blok formatı
4. System prompt + user prompt → LLM
5. LLM çıktısı: atıflı cevap

**LLM backend'leri:** HuggingFace (4-bit quant + adapter), Gemini, Anthropic, Groq, Dummy (testing).

### F. Evaluation Harness (`src/eval/`)

| Dosya | Sağladığı |
|---|---|
| `metrics.py` | Recall@{1,3,5,10}, MRR@10, nDCG@10, Exact Match, Token-F1, BLEU, ROUGE-L, Citation P/R/F1 |
| `judge.py` | RAGAS-style faithfulness — cevabı claim'lere ayır, her claim için kaynak desteği var mı LLM judge |
| `run_eval.py` | Uçtan uca değerlendirme CLI'ı |
| `aggregate_results.py` | Ablation hücreleri arası isolation/karşılaştırma tabloları |

**Komut:**

```powershell
python -m src.eval.run_eval `
    --test data/test_set/dev_full.jsonl `
    --mode hybrid --llm-backend anthropic --judge
```

**Çıktı:** `results/eval_<config>/{per_example.jsonl, summary.json}`

### G. Eğitim Notebook'ları (Kaggle T4×2)

| Notebook | İçerik |
|---|---|
| `01_embedding_finetune.ipynb` | e5-large LoRA, CachedMNRLoss + hard negatives |
| `02_reranker_finetune.ipynb` | XLM-R cross-encoder binary classification (v1) |
| `02b_reranker_v2.ipynb` | BGE-reranker-v2-m3 + LoRA (v2, aktif) |
| `03_llm_sft.ipynb` | Trendyol-7B QLoRA SFT (Unsloth) |
| `04_ablation_eval.ipynb` | 8-cell ablation batch evaluation |
| `05_build_ft_index.ipynb` | Fine-tuned e5 ile FAISS index build |

---

## 5. Dosya Yapısı

```
turkish-legal-rag/
├── data/
│   ├── corpus/
│   │   └── mevzuat_full_normalized.jsonl    (41973 madde)
│   ├── index_full/                          (BM25 + FAISS)
│   ├── finetune/
│   │   ├── synthetic_qa.jsonl               (3026 sentetik QA)
│   │   ├── embed_triplets_v2.jsonl          (15K hard negative triplet)
│   │   ├── yeniguno_law_eqa.jsonl
│   │   └── orioncaf_law_qa.jsonl
│   ├── scrape/                              (1085 PDF + active list)
│   ├── test_set/                            (gold dev/test JSONL)
│   └── models/                              (FT embedder, reranker, LLM adapter)
├── src/
│   ├── ingest/
│   ├── retrieval/
│   ├── embedding/
│   ├── reranker/
│   ├── llm/
│   ├── pipeline/
│   ├── eval/
│   └── demo/
├── notebooks/                               (Kaggle T4×2 training + eval)
├── results/                                 (ablation eval çıktıları)
├── docs/
│   └── PROJECT_OVERVIEW.md
├── README.md
└── requirements.txt
```

---

## 6. Komut Cheatsheet

```powershell
# --- VERİ ---
python -m src.ingest.fetch_active_list --tur 1 --tur 2 --tur 4
python -m src.ingest.scrape_mevzuat --mode full --turs 1 2 4
python -m src.ingest.normalize_kanun_short
python -m src.ingest.download_hf_datasets --include sft
python -m src.ingest.generate_synthetic_qa `
    --backend anthropic --balanced --limit 1000 --n-questions 3 `
    --output data/finetune/synthetic_qa.jsonl

# --- INDEX ---
python -m src.retrieval.build_index `
    --corpus data/corpus/mevzuat_full_normalized.jsonl `
    --output-dir data/index_full

python -m src.retrieval.retriever `
    --index-dir data/index_full --mode hybrid `
    "Cumhurbaşkanına hakaret cezası nedir?"

# --- EMBEDDING ---
python -m src.embedding.mine_hard_negatives `
    --qa data/finetune/synthetic_qa.jsonl `
    --output data/finetune/embed_triplets.jsonl

# --- EVAL ---
python -m src.eval.run_eval --test data/test_set/dev_full.jsonl --no-llm --mode bm25
python -m src.eval.run_eval --test data/test_set/dev_full.jsonl --no-llm --mode dense
python -m src.eval.run_eval --test data/test_set/dev_full.jsonl --no-llm --mode hybrid

python -m src.eval.run_eval `
    --test data/test_set/dev_full.jsonl `
    --mode hybrid --top-k 5 `
    --llm-backend hf --hf-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 `
    --judge --out results/eval_full
```

---

## 7. Sözlük

| Terim | Açıklama |
|---|---|
| **RAG** | Retrieval-Augmented Generation: relevant dokümanı bul, LLM'e ver, atıflı cevap üret. |
| **BM25** | Klasik sparse retrieval, kelime örtüşmesine dayalı TF-IDF benzeri puanlama. |
| **Dense retrieval** | Embedding-tabanlı: sorgu ve döküman aynı vektör uzayına gömülür, kosinüs benzerliği. |
| **Hybrid (RRF)** | Reciprocal Rank Fusion: `score = Σ 1/(k+rank)` ile sparse + dense sıralarını birleştirir. |
| **Hard negative** | Sorguya yakın görünen ama yanlış olan doküman; ayırt edici gradyan sağlar. |
| **LoRA / QLoRA** | Low-Rank Adaptation; sadece düşük-rütbeli matrisleri fine-tune eder. QLoRA bunu 4-bit kuantizasyonla birleştirir. |
| **Cross-encoder** | Query ve passage'i birlikte encode eder, tek skor döner. Yavaş ama doğru — reranker rolü için ideal. |
| **Faithfulness** | Cevabın kaynak metne sadakati; atıflı RAG'in temel kalite metriği. |
| **Difficulty tier** | lookup (düz olgusal) / reasoning (yorum) / edge (istisna) / multi_hop (çoklu madde) / no_answer (corpus dışı). |
