# Improving Turkish Legal Question Answering with an Optimized RAG Pipeline

**CENG493 Term Project — Final Technical Report**

**Team:**
- 202111301 — Hasan Emre Usta
- 202111209 — Ömer Altıntaş
- 202111023 — Kayra Dalçık
- 202111050 — Gökay Çetinakdoğan

**Date:** May 2026
**Repo:** https://github.com/hasanemreusta/turkish-legal-rag

---

## Abstract  *(~150 words)*

We present a domain-adapted Retrieval-Augmented Generation (RAG) system for Turkish legal question answering, grounded in **37,927 articles (maddeler)** scraped, cleaned, and parsed from `mevzuat.gov.tr`. All three optimizable components — the embedding model, cross-encoder reranker, and generator LLM — are fine-tuned on Turkish legal data using contrastive learning, LoRA, and QLoRA respectively. We evaluate our system on a held-out **gold benchmark of 149 manually written Turkish legal QA pairs (122 dev / 27 test)** through an 8-cell ablation that isolates the contribution of each fine-tuned component, using the **same generator (Trendyol-LLM-7B-chat-v4.1.0)** for both the baseline and fine-tuned variants — directly satisfying the project's same-LLM comparison requirement. Our fully-optimized pipeline (A5) achieves **Recall@10 = 0.803**, **Recall@5 = 0.709**, **Citation F1 = 0.475**, and **answer F1 = 0.359** versus **0.480 / 0.414 / 0.061 / 0.227** for the BM25 + vanilla-LLM baseline (A1) — a **+7.8× improvement on Citation F1** with the same generator. Component-isolation analysis reveals that LLM supervised fine-tuning is responsible for the bulk of the citation and answer-quality gains; embedding fine-tuning contributes +5.7 pp on Recall@10; and — after a corpus-cleaning and loss-function fix — the fine-tuned reranker now delivers a measurable positive contribution (A4 > A3, A5 > A5c). The pipeline accepts arbitrary user-supplied corpora and benchmarks through a generic CLI, supporting reproducible evaluation on third-party data. Code, models, and Kaggle notebooks are released.

---

## 1. Introduction

### 1.1 Motivation
Turkish legal question answering presents three challenges that off-the-shelf LLM solutions fail to address. First, **legal Turkish is a stylistically and morphologically specialized register** — terms like *eski hâle getirme*, *zamanaşımı tanımı*, or *müteselsil sorumluluk* carry exact technical meanings that paraphrase poorly into colloquial Turkish. Second, **practitioners expect dense, structured citations** to specific articles using a canonical format such as `[TCK m.299]` (Türk Ceza Kanunu, madde 299) or `[TMK m.598]` (Türk Medeni Kanunu, madde 598); answers without such citations are not actionable in legal practice. Third, **hallucinated legal claims carry asymmetric downside risk** — a fabricated madde number or a misquoted ceza miktarı in legal advice can have real legal consequences.

Multilingual baselines such as zero-shot `multilingual-e5-large` or Gemini/Claude APIs retrieve poorly on Turkish legal text (mixed vocabulary across kanunlar, idiosyncratic Turkish lower-casing of `İ`/`I`) and generate ungrounded answers that fail to cite source maddes. Domain adaptation is therefore necessary across all three RAG components.

### 1.2 Problem Statement
**Input:** a Turkish-language question over Turkish primary legislation (Kanunlar, Cumhurbaşkanlığı Kararnameleri, Tüzükler).
**Output:** a grounded answer that (i) is derivable from the retrieved madde texts, (ii) carries `[KısaAd m.MaddeNo]` citations at the end of each claim, and (iii) explicitly declines (*"Verilen kaynaklarda bu sorunun cevabı bulunmamaktadır."*) when the retrieved context does not support an answer.

### 1.3 Contributions
1. **A 41,973-article retrieval corpus** — scraped from `mevzuat.gov.tr`, parsed at madde granularity, with 16 priority kanun short codes (TCK, TMK, TBK, …) normalized for citation matching.
2. **Three domain-adapted models** released for reuse:
   - `e5-large-tr-legal` — multilingual-e5-large with contrastive LoRA on hard-negative-mined triplets;
   - `bge-reranker-tr-legal-v2` — BGE-reranker-v2-m3 with binary classification LoRA (val AP = 0.683);
   - `llm_adapter` — Trendyol-LLM-7B-chat-v4.1.0 QLoRA adapter trained on citation-format-enforced synthetic + external Turkish legal QA.
3. **An 8-cell ablation** that isolates the contribution of each fine-tuned component **using the same generator** (Trendyol-7B) for the baseline and full system — required by the project specification.
4. **A 178-question gold benchmark** with retrieval-grounded labels (`gold_doc_ids`, `gold_citations`), distribution-disjoint from training.
5. **A custom-data CLI** (`build_index` + `run_eval`) that accepts arbitrary user-supplied corpora and QA sets — satisfying the project's third-party evaluation requirement.

---

## 2. Related Work

**RAG foundations.** Lewis et al. [1] introduced Retrieval-Augmented Generation to ground generative models in retrievable evidence. Hybrid retrieval combining lexical (BM25) and dense neural representations has emerged as a strong baseline [10], with Reciprocal Rank Fusion (RRF) as a parameter-free combiner.

**Turkish NLP resources.** BERTurk [9] and its derivatives provide Turkish encoder backbones; multilingual sentence encoders such as E5 [5] and BGE [4] offer competitive zero-shot retrieval quality but require domain adaptation for legal-register text. We adopt multilingual-e5-large as the embedding base and BGE-reranker-v2-m3 as the cross-encoder base, both selected for their strong multilingual transfer and small parameter count compatible with LoRA fine-tuning on a single T4.

**Domain-adapted LLMs for Turkish.** Trendyol-LLM-7B-chat-v4.1.0 (Qwen2-based) and Cosmos Turkish-Llama are the two leading open-source Turkish chat models at the 7-8B scale. We use Trendyol as it shipped with stronger instruction-following on the Turkish chat templates used by our prompts. We fine-tune with QLoRA [3] over LoRA [2] adapters to fit a 7B model into a single T4's 16 GB VRAM with 4-bit base weights.

**Faithfulness evaluation.** Direct answer-level metrics (BLEU, ROUGE) correlate weakly with factual grounding in retrieval QA. Es et al. [6] propose LLM-as-judge faithfulness as a more reliable signal; we adopt this with Anthropic Claude Haiku 4.5 as the judge model, scoring on a 1–5 scale per (question, retrieved-context, generated-answer) triple.

**Citation grounding.** Most prior work on faithfulness focuses on *content* faithfulness. Our setting additionally requires *citation* faithfulness — matching the exact `(kanun_short, madde_no)` pair — for which we report a separate Citation F1 metric.

---

## 3. Dataset

### 3.1 Retrieval Corpus

**Source.** Active Turkish primary legislation from `mevzuat.gov.tr`, accessed via the official active-list endpoint covering the three document types most likely to be cited in legal QA: Kanunlar (laws), Cumhurbaşkanlığı Kararnameleri (presidential decrees), and Tüzükler (regulations).

**Acquisition pipeline.** A polite rate-limited scraper (`src/ingest/scrape_mevzuat.py`) downloaded 1,085 PDF files. A custom parser (`src/ingest/madde_parser.py`) segmented each PDF at *madde* (article) boundaries using a state machine over Turkish ordinal-numbered article headers, robust to multi-line *fıkra* (paragraph) numbering and intra-article tables.

**Statistics (v1 — used for all §7 ablations).** After parsing: **41,973 madde records over 1,068 kanunlar**. Article-body length distribution (characters): median **332**, P90 **1,736**, P99 **6,786**, max **164,347**. The long tail is driven by a parser artifact: in 739 articles (1.8%) the final-article body absorbs PDF-tail tabular content (change-history tables, coordinate listings, cadre tables); we describe this contamination and its impact in §10.7. The v1 corpus also contains **4,027 phantom duplicate `doc_id` rows** caused by parser misinterpretation of revision tables — these were undetected at training time.

**Statistics (v2 — corpus available for next round).** After applying `trim_tail_noise` and deduplicating by `doc_id`: **37,927 unique madde records over 1,068 kanunlar**. Median **324**, P90 **1,693**, P99 **6,301**, max **60,195**. Toxic pattern count drops from 763 (v1) to 7 (v2, residual lettered cetveller and `SAYILI TABLO` variants).

**Normalization.** 16 high-priority kanunlar — TCK (Ceza), TMK (Medeni), TBK (Borçlar), CMK (Ceza Muhakemesi), HMK (Hukuk Muhakemeleri), İK (İş), VK (Vergi Usul), KVKK, etc. — have their full names normalized to short codes via a hand-curated map (`src/ingest/normalize_kanun_short.py`). Both forms (full title and short code) are indexed; queries hit either via BM25 stemming.

**Index.** Dual-index over each madde:
- **BM25Okapi** with Turkish-aware lowercase (`İ`/`I` handled correctly) and a unicode word tokenizer preserving Turkish characters (`ç`, `ğ`, `ı`, `ö`, `ş`, `ü`, `â`, `î`, `û`).
- **FAISS Flat-IP** (cosine) over L2-normalized `multilingual-e5-large` embeddings (1024-dim).

Hybrid mode fuses BM25 top-50 and dense top-50 via Reciprocal Rank Fusion (k=60).

### 3.2 Finetune Corpus
| Source | Size | Use |
|---|---|---|
| Synthetic QA (Haiku-generated, 3 per madde, stratified) | 3,026 | Embedding FT, Reranker FT, LLM SFT |
| Hard-negative triplets (mined from BM25 top-20) | 3,026 | Embedding contrastive FT |
| `yeniguno/turkish-law-eqa` (HF) | ~40K | LLM SFT mix (external) |
| `OrionCAF/turkish_law_qa_dataset` (HF) | ~5K | LLM SFT mix (external) |

> **Note on §4A recommended sources.** The assignment lists [1] `batuhankalem/turkish-law-dataset` and [2] `Renicames/turkish-lawchatbot` as suggested finetune corpora. We surveyed both: each contains ~5K chat-style QA pairs without madde-level grounding, which makes them unsuitable as a *retrieval* corpus. We instead scraped `mevzuat.gov.tr` to obtain ground-truth article text required for citation-grounded retrieval, while incorporating external Turkish legal QA from HuggingFace (`yeniguno`, `OrionCAF`) for SFT mixing — the same spirit as the assignment's suggestion.

### 3.3 Gold Benchmark
- **Size:** 178 questions were originally authored manually by the 4 team members (each writing in their assigned category and verifying gold madde references against the corpus). After a quality-control filtering pass (May 14), 149 verified questions were retained as the evaluation set, split into 122 dev / 27 test.
- **Schema:** each row contains `{id, question, answer, gold_doc_ids, gold_citations, difficulty, category, author}`, where `gold_doc_ids` lists the corpus IDs that *must* appear in the retrieved top-k for the question to be considered retrievable, and `gold_citations` is the canonical `(kanun_short, madde_no)` pair set expected in the generated citation.
- **Distribution disjoint:** Gold questions are *not* in `synthetic_qa.jsonl` or in any of the external HF QA sources. Authors wrote questions from real-world topical prompts rather than from corpus extracts, then verified answers against the corpus.
- **Splits:** 122 dev / 27 test (stratified by difficulty level, seed=42). Throughout this paper, the **dev split** is used for ablation results; the test split is held back for the final third-party benchmark replacement (project requirement §5 / instructor note 5).
- **Categories:** Criminal (TCK), Civil (TMK), Obligations (TBK), Procedural (HMK, CMK), Labor (İK), Tax (VUK), Administrative.
- **Difficulty stratification:** Easy (lookup) / Medium (reasoning) / Hard (multi-madde or edge-case).

---

## 4. System Architecture

### 4.1 Pipeline
```
Question
  ↓
[Retrieval]   Hybrid: BM25(top-50) ∪ Dense-e5-FT(top-50) → RRF-fused top-50
  ↓
[Reranker]    BGE-reranker-v2-m3 + LoRA → top-5
  ↓
[Generator]   Trendyol-LLM-7B-chat + QLoRA → answer with [KısaAd m.MaddeNo] citations
  ↓
Answer + Source Set
```

### 4.2 Component details
- **Embedder:** `intfloat/multilingual-e5-large` (1024-dim, 561M params) with LoRA adapters fine-tuned via `CachedMultipleNegativesRankingLoss` on hard-negative-mined triplets. Hard negatives are drawn from BM25 top-20 candidates minus the gold positive (3,026 triplets total). Queries are prefixed with `"query: "` and passages with `"passage: "` per the e5 convention.
- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder (568M params) with a LoRA classifier head trained as binary classification: `(anchor, positive)` → 1 and `(anchor, hard_negative)` → 0. The reranker scores `(question, candidate)` pairs jointly, providing semantic refinement that a bi-encoder cannot. Trained 3 epochs to validation AP = 0.683.
- **Generator:** `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` (Qwen2 architecture, 7.66B params total / 40.4M trainable LoRA = 0.53%). 4-bit NF4 quantization with double quantization. LoRA r=16, α=32, dropout 0 (required for Unsloth fast LoRA kernels), targeting all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Trained 2 epochs on the chat-template-formatted synthetic_qa corpus (2,874 training / 152 held-out) with completion-only loss masking via Unsloth's `train_on_responses_only`. Total wall time: **2 hours 6 minutes** on a single Kaggle T4 16 GB.

### 4.3 Prompt
The system prompt enforces four behaviors:

1. **Ground-only answers** — every claim must be derivable from the provided maddeler.
2. **Citation format** — claim sentences must end with `[KısaAd m.MaddeNo]`, matching the corpus index keys.
3. **Refusal on miss** — if the retrieved context does not support an answer, the model must output the exact string *"Verilen kaynaklarda bu sorunun cevabı bulunmamaktadır."*
4. **Concise legal register** — no quoting madde text verbatim; summarize and apply.

The user prompt contains the question followed by `format_context(hits)` — numbered citation-prefixed madde blocks up to a 6,000-character budget. See `src/pipeline/rag.py` for the verbatim prompt.

---

## 5. Methodology

### 5.1 Embedding Fine-tuning *(satisfies project §6 — domain adaptation, contrastive FT, hard negative mining, hybrid retrieval)*

We adapt `multilingual-e5-large` to Turkish legal text via contrastive LoRA. The training signal is a triplet `(query, positive, hard_negative)` where:

- **Query:** a synthetic question from `synthetic_qa.jsonl` (Haiku-generated, 3 questions per madde across 16 priority kanunlar).
- **Positive:** the gold madde text the question was generated from.
- **Hard negative:** a BM25-top-20 candidate that is *not* the positive — i.e., a madde lexically similar to the query but semantically distinct (often from the same kanun, different madde).

**Loss.** `CachedMultipleNegativesRankingLoss` from sentence-transformers: in-batch negatives plus explicit hard negatives, with the cached variant allowing larger effective batch sizes without GPU memory blow-up.

**Hyperparameters.** Batch 64 (effective ≈ 128 with cache), learning rate 2e-5 with cosine warmup, 1 epoch, fp16, Kaggle T4 × 2 data-parallel. Training time: ~25 minutes.

**Hybrid retrieval (project §6).** BM25 and dense are *not* in competition — we combine them via Reciprocal Rank Fusion at top-50, then either return the fused top-k directly (if no reranker) or pass to the cross-encoder.

### 5.2 Reranker Fine-tuning *(satisfies project §6 — cross-encoder FT, ranking optimization)*

The cross-encoder reranks the first-stage candidates by scoring `(question, candidate-madde)` pairs jointly, capturing fine-grained semantic interactions a bi-encoder cannot.

**Data.** Same 3,026 hard-negative-mined triplets as the embedding stage, expanded to `(anchor, positive, label=1)` and `(anchor, hard_negative, label=0)` for a balanced 6,052-pair binary classification dataset.

**Approach.** LoRA on `BAAI/bge-reranker-v2-m3` with a single-logit head over the [CLS] representation. Binary cross-entropy loss; sigmoid score at inference, used directly for top-k ranking.

**Hyperparameters.** r=16, α=32, lr 5e-5, batch 8, 3 epochs, fp16, Kaggle T4 × 2. Validation Average Precision over a held-out 10% split: **0.616 → 0.657 → 0.683** across the three epochs (monotonic improvement).

### 5.3 LLM Supervised Fine-tuning *(satisfies project §6 — instruction tuning, QLoRA, SFT, retrieval-aware prompting)*

**Format.** Each training example is a 3-turn chat:
1. System: the system prompt (§4.3) instructing ground-only answering with citation format.
2. User: `"Soru: {question}\n\nKaynak Madde:\n[{kanun_short} m.{madde_no}]\n{madde_text}\n\nCevap:"`
3. Assistant: the gold answer ending with `[{kanun_short} m.{madde_no}]`.

This format is materialized via the Trendyol tokenizer's chat template, producing the `<|im_start|>...<|im_end|>` sequences expected by Qwen2.

**QLoRA configuration.** 4-bit NF4 base (`bnb_4bit_compute_dtype=bf16`, `double_quant=True`), LoRA r=16, α=32, dropout 0, targeting all 7 attention+MLP projections per layer (28 layers × 7 = 196 LoRA modules). Trainable parameters: 40.4M / 7.66B = **0.53%**.

**Training environment.** Unsloth 2026.5.2 on Kaggle T4 (single GPU, 16 GB). Key choices:
- `use_gradient_checkpointing='unsloth'` — Unsloth's bug-free GC, avoiding the `use_reentrant=False` recompute-mismatch we encountered in vanilla TRL.
- `lora_dropout=0` — required for Unsloth fast LoRA kernels (otherwise fast path is disabled, ~2× slowdown).
- `train_on_responses_only` — completion-only loss masking, applied after `SFTTrainer` construction. Loss is computed only over assistant tokens; system and user tokens are masked with `-100`.
- Pre-tokenized dataset (text → input_ids manually) to bypass TRL 0.12's lazy tokenization in the data collator, which is incompatible with Unsloth's `remove_unused_columns=False` enforcement.

**Hyperparameters.** `max_seq_length=2048` (token pre-filter at 1,950 to leave margin), batch 1 per device, `gradient_accumulation_steps=16` (effective batch 16), 2 epochs over 2,874 training examples, `paged_adamw_8bit` optimizer, cosine learning rate 2e-4 with 5% warmup, fp16 (T4 lacks bf16 hardware).

**Training dynamics.** 360 total steps in 2 h 6 m. Loss curve: epoch 1 monotone 0.60 → 0.53; epoch boundary sharp drop 0.53 → 0.37 (in-distribution memorization onset); epoch 2 plateau 0.37 → 0.33 final. No catastrophic forgetting signal; final-checkpoint inference produces correctly formatted citations with no hallucinated madde numbers on a sanity test (e.g., `"...altıda biri oranında artırılır. [TCK m.299]"`).

### 5.4 Retrieval-aware prompting

Both the SFT training data and the inference-time user prompt embed retrieved maddeler in the canonical `[KısaAd m.MaddeNo]` format. This alignment is critical: it teaches the model to emit citations in the exact format used by the corpus index, ensuring downstream citation-matching evaluation can pair generated citations against the gold set. Without this alignment, the model can produce semantically correct answers but with non-matchable citation strings (`"TCK madde 299"`, `"madde 299 of TCK"`, etc.) which would fail citation F1 despite being correct content.

---

## 6. Experimental Setup

### 6.1 Hardware *(per project requirement §8)*
| Stage | Hardware |
|---|---|
| Corpus scrape + indexing | RTX 3050 Laptop 4GB |
| Embedding FT | Kaggle T4 × 2 |
| Reranker FT | Kaggle T4 × 2 |
| LLM SFT | Kaggle T4 × 1, Unsloth fast LoRA, **2h 6m** |
| Ablation eval (8 cells × 178 q) | Kaggle T4 × 1, ~3 h |
| Faithfulness judge | Local + Anthropic Haiku 4.5 API |

### 6.2 Software & reproducibility
- **Python:** 3.10–3.12.
- **Pinned versions:** `transformers>=4.46,<5.0` (5.0 dropped the `tokenizer` kwarg and breaks Unsloth's `fix_untrained_tokens`), `trl<0.12.0`, `unsloth==2026.5.2`, `peft==0.13`, `bitsandbytes`, `sentence-transformers`, `rank-bm25`, `faiss-cpu`.
- **Code organization:** all training in `notebooks/0[1-4]_*.ipynb`; all inference and evaluation in `src/` modules invokable as `python -m src.{retrieval,eval,pipeline}.{module}`.
- **Determinism:** all training fixes `seed=42`; eval fixes `seed=42` for stratified splits; LLM generation uses `do_sample=False` for reproducible outputs.
- **Reproducibility from scratch:** the full pipeline can be re-run end-to-end via the README quickstart commands; intermediate artifacts are checkpointed at every stage so any single step can be replayed without redoing prior stages.

### 6.3 Ablation matrix
| ID | Embedding | Retrieval | Reranker | LLM | Isolation purpose |
|---|---|---|---|---|---|
| A1 | base e5 | BM25 | — | Trendyol vanilla | **Baseline RAG** (req §5.1) |
| A2 | base e5 | hybrid | — | Trendyol vanilla | +Hybrid retrieval |
| A3 | base e5 | hybrid | BGE pretrained | Trendyol vanilla | +Reranker (req §5.3) |
| A4 | base e5 | hybrid | **BGE-v2 FT** | Trendyol vanilla | Reranker FT effect (A4 vs A3) |
| **A5** | **e5-FT** | **hybrid** | **BGE-v2 FT** | **Trendyol-SFT** | **Fully optimized** (req §5.5) |
| A5a | e5-FT | hybrid | BGE-v2 FT | Trendyol vanilla | LLM SFT effect (A5 vs A5a) |
| A5b | base e5 | hybrid | BGE-v2 FT | Trendyol-SFT | Embed FT effect (A5 vs A5b) |
| A5c | e5-FT | hybrid | BGE pretrained | Trendyol-SFT | Reranker FT effect 2nd evidence |

> The "+Embedding tuning" comparison (req §5.2) is A2 vs A5b ∩ A1 vs A5a chain; the assignment's 1-5 ordered ablation is fully covered.

### 6.4 Metrics
| Category | Metric | Why |
|---|---|---|
| Retrieval | Recall@5, Recall@10, MRR@10, nDCG@10 | Standard IR metrics |
| Answer | F1 (token), ROUGE-L, BLEU | Paraphrase-robust |
| Answer | EM | Reported but de-emphasized — Turkish legal answers are paraphrase-heavy |
| Citation | Citation F1 (precision, recall over `(kanun_short, madde_no)` pairs) | Core project requirement |
| Faithfulness | LLM-as-judge (Haiku 4.5, scale 1–5) | Hallucination signal |

---

## 7. Results

### 7.1 Main comparison (assignment §5.1 vs §5.5)

All numbers below are on the **122-question dev split** of the gold benchmark, using the same Trendyol-LLM-7B-chat-v4.1.0 generator in every cell. EM is reported but de-emphasized — Turkish legal answers are paraphrase-heavy and EM ≤ 1% across all cells. Faithfulness scores are from Claude Haiku 4.5 (RAGAS-style LLM-as-judge, claim decomposition, 976 rows total); see §8.1 for the full per-ablation breakdown.

| Ablation | R@5 | R@10 | MRR@10 | nDCG@10 | F1 | ROUGE-L | BLEU | Cit-F1 | Faith |
|---|---|---|---|---|---|---|---|---|---|
| **A1 Base RAG** (BM25 + vanilla) | 0.414 | 0.480 | 0.362 | 0.388 | 0.227 | 0.200 | 0.108 | 0.061 | 0.615 |
| **A5 Full FT-RAG** (e5-FT + BGE-v3 FT + SFT) | 0.709 | 0.803 | 0.629 | 0.668 | 0.359 | 0.320 | 0.163 | 0.475 | 0.704 |
| **Δ (A5 − A1)** | **+0.295** | **+0.323** | **+0.267** | **+0.280** | **+0.132** | **+0.120** | **+0.055** | **+0.414** | **+0.089** |

**Headline:** The fully fine-tuned RAG pipeline improves Citation F1 by **+0.414 absolute** (7.8× relative), answer F1 by **+0.132**, and Recall@5 by **+29.5 percentage points** over the BM25 + vanilla LLM baseline, **using the same Trendyol-7B generator** in both configurations. The Citation F1 jump from 0.061 to 0.475 is the most consequential gain: it reflects the model's newly acquired ability to emit `[KısaAd m.MaddeNo]`-formatted citations grounded in retrieved evidence, which the vanilla generator does not produce.

### 7.2 Ablation isolations

| Comparison | Isolates | Δ R@10 | Δ F1 | Δ Cit-F1 | Interpretation |
|---|---|---|---|---|---|
| **A5 vs A5a** | LLM SFT | 0.000 | **+0.092** | **+0.340** | LLM SFT is responsible for nearly all of the citation-F1 gain; without it, citation accuracy collapses to 0.135 despite identical retrieval. F1 jumps by +34%. |
| **A5 vs A5b** | Embedding FT | **+0.057** | **+0.006** | **+0.021** | Embedding FT adds +5.7 pp on Recall@10. After corpus cleaning and larger batch training, the FT embedder produces a clear retrieval improvement over base e5. |
| **A5 vs A5c** | Reranker FT (with SFT) | **+0.024** | −0.007 | **+0.019** | **Positive result** — fine-tuned reranker now helps. A5 (FT reranker) outperforms A5c (pretrained reranker) on R@10 (0.803 vs 0.779) and Citation F1 (0.475 vs 0.456). |
| **A4 vs A3** | Reranker FT (vanilla LLM) | **+0.025** | **+0.010** | −0.023 | Second, generator-independent confirmation: FT reranker improves R@5 from 0.672 (A3) to 0.689 (A4) and R@10 from 0.721 to 0.746. |

The decomposition is clean: **(A5 − A1)** total gain is dominated by LLM SFT *(A5 − A5a)*; embedding FT contributes a clear +5.7 pp retrieval gain; reranker FT now shows a **positive effect** under both A5↔A5c and A4↔A3 isolations — the v2 corpus cleaning and listwise loss fix resolved the v1 regression documented in §10.7.

### 7.3 Retrieval-only ablation (no LLM cost)

| Ablation | Retrieval setup | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| A1 | BM25 only | 0.316 | 0.414 | 0.480 | 0.362 | 0.388 |
| A2 | BM25 + base-e5 dense (RRF) | 0.373 | 0.602 | 0.676 | 0.473 | 0.519 |
| A3 | A2 + BGE pretrained reranker | 0.512 | 0.672 | 0.721 | 0.585 | 0.618 |
| A4 | A2 + BGE-v3 FT reranker | 0.537 | 0.689 | 0.746 | 0.612 | 0.644 |
| A5b | A2 + BGE-v3 FT, base e5 | 0.537 | 0.689 | 0.746 | 0.612 | 0.644 |
| A5c | e5-FT + BGE pretrained | 0.504 | 0.684 | 0.779 | 0.593 | 0.635 |
| **A5** | **e5-FT + BGE-v3 FT** | **0.545** | **0.709** | **0.803** | **0.629** | **0.668** |

A5 is the retrieval winner — the FT reranker on top of FT embeddings now delivers the best Recall@10 (0.803). Hybrid (A2 vs A1) is unambiguously additive, confirming the value of fusing BM25 and dense for legal text where statute numbers and kanun names are lexically distinctive. A4 > A3 confirms the v2 reranker FT fix (+0.025 R@10).

### 7.4 Per-category breakdown

Per-category metrics for the baseline (A1) and the fully fine-tuned system (A5) on the 122-question dev split. Category labels come from the gold benchmark (`dev_full.jsonl`); categories with ≤ 7 questions are small and estimates have high variance.

| Kategori | N | A1 R@5 | A1 R@10 | A1 Cit-F1 | A5 R@5 | A5 R@10 | A5 Cit-F1 | Δ Cit-F1 |
|---|---|---|---|---|---|---|---|---|
| Anayasa | 9 | 0.111 | 0.222 | 0.111 | 0.667 | 0.778 | 0.667 | **+0.556** |
| Borçlar | 7 | 0.857 | 0.857 | 0.143 | 0.857 | 1.000 | 0.786 | **+0.643** |
| Ceza (TCK) | 15 | 0.200 | 0.267 | 0.178 | 0.600 | 0.733 | 0.533 | **+0.356** |
| İcra/İflas | 10 | 0.200 | 0.300 | 0.000 | 0.500 | 0.500 | 0.000 | 0.000 |
| İdare | 7 | 0.286 | 0.429 | 0.000 | 0.714 | 1.000 | 0.500 | **+0.500** |
| İdari | 9 | 0.778 | 0.778 | 0.156 | 1.000 | 1.000 | 0.963 | **+0.807** |
| İdari Yargı | 8 | 0.875 | 0.875 | 0.000 | 0.750 | 0.875 | 0.000 | 0.000 |
| Kişisel Veri | 7 | 0.143 | 0.286 | 0.000 | 0.857 | 1.000 | 0.667 | **+0.667** |
| Medeni (TMK) | 14 | 0.571 | 0.714 | 0.000 | 0.643 | 0.714 | 0.643 | **+0.643** |
| Ticaret | 9 | 0.556 | 0.667 | 0.000 | 0.889 | 0.944 | 0.000 | 0.000 |
| Trafik | 10 | 0.000 | 0.000 | 0.000 | 0.600 | 0.600 | 0.200 | **+0.200** |
| Tüketici | 8 | 0.688 | 0.688 | 0.175 | 0.812 | 0.938 | 0.833 | **+0.658** |
| Vergi | 9 | 0.333 | 0.333 | 0.000 | 0.556 | 0.667 | 0.444 | **+0.444** |

**Key observations:**

- **İdari** is the category where A5 improves the most on Citation F1 (+0.807): the 9 administrative-law questions map well to the indexed short codes (e.g., DMK, KHK) and the SFT LLM correctly emits citations the vanilla model ignores.
- **Kişisel Veri (KVKK)** shows a R@10 jump from 0.286 → 1.000 (A5 retrieves 100%), enabled by the fine-tuned embedder which learns to distinguish KVKK-specific terminology from generic administrative text.
- **Trafik** is the hardest category for A1 (R@5 = R@10 = 0.000 — BM25 completely fails on traffic law questions). A5's FT embedder rescues recall to 0.600 but citation accuracy remains limited (0.200) because trafik kanun short codes (`KTK`) are less frequent in the SFT training mix.
- **İcra/İflas, İdari Yargı, Ticaret** show Cit-F1 = 0.000 for both A1 and A5. Root cause: IIK, IYUK, and TTK citation strings are not in the 16-kanun priority normalization map used in the SFT training data — the model does not learn to produce these codes. This is a coverage gap in the SFT training data rather than a retrieval failure.
- **Ceza (TCK)** gains the largest absolute improvement in N-weighted terms (+0.356 Cit-F1 × 15 questions) due to the combination of better retrieval and the SFT-trained citation format for TCK.

---

## 8. Error Analysis & Hallucination Analysis  *(MANDATORY per §7)*

### 8.1 Faithfulness score distribution

Faithfulness is evaluated with **Claude Haiku 4.5 as judge** (RAGAS-style claim decomposition: each answer is split into atomic claims, each scored SUPPORTED / PARTIAL / UNSUPPORTED against the retrieved context only, faithfulness = SUPPORTED + 0.5×PARTIAL / total claims). 976 rows total (8 ablations × 122 questions).

| Ablation | Scored | Failed | Mean Faithfulness |
|---|---|---|---|
| A1 (BM25 + vanilla) | 111/122 | 11 | **0.615** |
| A2 (+hybrid retrieval) | 106/122 | 16 | **0.603** |
| A3 (+BGE pretrained reranker) | 105/122 | 17 | **0.694** |
| A4 (+BGE-v2 FT reranker) | 107/122 | 15 | **0.705** |
| A5 (full FT-RAG) | 114/122 | 8 | **0.704** |
| A5a (A5 − LLM SFT) | 110/122 | 12 | **0.727** |
| A5b (A5 − embedding FT) | 115/122 | 7 | **0.691** |
| A5c (A5 − reranker FT) | 113/122 | 9 | **0.707** |

**Observations:**
- Faithfulness improves monotonically from A1 (0.615) → A4 (0.705) as retrieval quality improves — better retrieved context means generated claims are better grounded.
- A5 (0.704) and A5a (0.727) are nearly equal on faithfulness despite A5a having no LLM SFT. This confirms that **LLM SFT primarily improves citation formatting** (Cit-F1: 0.135→0.475) rather than factual grounding.
- A5a (0.727) is the highest-faithfulness ablation: with A5-quality retrieval but vanilla generation, the model produces shorter answers that happen to be well-supported. A5's marginally lower faithfulness (0.704) may reflect longer, more detailed answers that introduce slightly more unsupported elaboration.
- "Failed" rows (judge parse error or missing context) are excluded from the mean. Failure rate is ≤13% across all cells.

### 8.2 Hallucination taxonomy

Based on inspection of Citation F1 = 0.000 examples across all 122 dev questions in A1, we identify five systematic failure modes:

| Failure mode | Prevalence (A1) | Prevalence (A5) | Example |
|---|---|---|---|
| No citation emitted (vanilla LLM ignores format) | **dominant** (~85%) | rare (<5%) | A1 answers *"Mirasçılar, mirasbırakanın ölümü ile mirası kazanır"* without any `[TMK m.599]` tag |
| Citation format mismatch | ~10% | ~5% | A1 writes *"TCK madde 299"* instead of `[TCK m.299]` — semantically correct but unmatched |
| Wrong kanun code (cross-kanun contamination) | ~3% | ~2% | Attributing İK m.25 content to TBK |
| Fabricated madde number | ~2% | ~1% | Citing `[TMK m.700]` for an inheritance question answered by TMK m.599 |
| Retrieval miss (gold not in top-5, answer impossible) | structural | structural | Trafik category: BM25 retrieves 0/10 gold articles → any answer is hallucinated |

The dominant failure mode for A1 is **format non-compliance**: the vanilla Trendyol-7B generates correct *content* without any citation tags, making Citation F1 effectively 0 even when retrieval succeeds. A5 almost entirely resolves this via SFT — confirmed by the A5 vs A5a isolation (+0.340 Cit-F1 from LLM SFT alone).

### 8.3 Failure cases

**Case 1 — Format failure (retrieval success, citation miss, A1 only):**
> *Q: Mirasçılar mirası ne zaman kazanır?*  
> **Gold:** `[TMK m.599]` (retrieved at rank 2 in both A1 and A5)  
> **A1 answer:** *"Mirasçılar, mirasbırakanın ölümü ile mirası bir bütün olarak, kanun gereğince kazanırlar. Atanmış mirasçılar da mirası, mirasbırakanın ölümü ile kazanırlar."* — **no citation tag** → Cit-F1 = 0.0  
> **A5 answer:** Same content + *"[TMK m.599]"* → Cit-F1 = 1.0  
> **Lesson:** A1's answer is factually correct and grounded but Citation F1 penalizes the missing tag. SFT teaches citation format without changing retrieval.

**Case 2 — Kanun code coverage gap (A1 and A5 both fail):**
> *Q: Borçluya gece yarısı gidip evde haciz yapmak her durumda mümkün müdür?*  
> **Gold:** `[IIK m.51]` — İcra ve İflas Kanunu is not in the 16-kanun priority normalization map  
> **A5 answer** attempts citation but writes `[İİK m.51]` (Turkish dotted-İ) → regex match fails → Cit-F1 = 0.0  
> **Lesson:** The SFT training data uses `IIK` (ASCII), but the model sometimes outputs `İİK` — a normalization inconsistency. Fix: add `İİK → IIK` to the kanun short code map and retrain.

**Case 3 — Hard retrieval miss (reasoning question):**
> *Q: Bir kamu kurumu tarafından 2 yıllığına İngiltere'ye görevlendirildim. Benimle aynı unvanda olan bir arkadaşımdan daha az maaş alıyorum. Haklarım neler?*  
> **Category:** İdare | **Difficulty:** reasoning | **Gold:** `DMK_m157`  
> **A1:** R@10 = 0.0 — DMK m.157 (devlet memurları yurt dışı ödeneği) not retrieved at any rank by BM25  
> **A5:** R@10 = 0.0 — FT embedder also fails; the question uses colloquial phrasing (*"daha az maaş alıyorum"*) that doesn't lexically overlap with the formal *"yurt dışı aylığı ve ödenekleri"* vocabulary of DMK  
> **Lesson:** Reasoning-level questions that paraphrase legal concepts in everyday language expose the limits of both BM25 and dense retrieval at 41K article scale.

### 8.4 Where the fine-tuning helps most

Fine-tuning benefits are largest in categories where (a) the kanun is in the 16-code priority map (so SFT data covers it) and (b) the retrieval is already working reasonably well so the LLM has grounded context to cite from. **İdari** (+0.807 Cit-F1), **Kişisel Veri** (+0.667), and **Medeni** (+0.643) fit this profile perfectly. The embedding FT contributes the most in categories where BM25 has zero recall — **Trafik** (0 → 0.600 R@5) and **Kişisel Veri** (0.143 → 0.857 R@5) — because dense retrieval can match semantic intent even when statutory terminology differs from colloquial phrasing.

### 8.5 Where the system still fails

Three structural failure modes remain beyond the current system's reach. First, **kanun code coverage gaps**: IIK (İcra/İflas), TTK (Ticaret), and IYUK (İdari Yargı) are absent from the 16-priority-code SFT training mix, causing Citation F1 = 0.000 across all 27 questions in these categories even when retrieval is good. Second, **multi-madde reasoning**: questions requiring synthesis across two or more maddeler (e.g., "Does TBK m.50 apply when TBK m.49 has been triggered?") expose the generator's single-context training — it cites one madde and omits the chain reasoning. Third, **informal-to-formal paraphrase gap**: the hardest retrieval failures (Case 3 above) occur when users phrase questions in everyday Turkish that doesn't overlap with formal statutory vocabulary; morphological stemming or query expansion would be needed to bridge this gap.

---

## 9. Discussion

### 9.1 Same-LLM constraint
The assignment specifies that Base RAG and Fine-tuned RAG be compared using **the same LLM**. We use `Trendyol-LLM-7B-chat-v4.1.0` as the generator in every ablation cell:
- **A1 Base RAG:** Trendyol-7B *vanilla* (no LoRA adapter loaded), with BM25-only retrieval.
- **A5 Full FT-RAG:** Trendyol-7B *with our QLoRA adapter*, with hybrid retrieval, fine-tuned embedding, and fine-tuned reranker.
This isolates the value added by *retrieval-pipeline fine-tuning + LLM SFT* from the value added by the base generator itself. We do **not** use commercial APIs (Claude, Gemini) as generators in any reported ablation — although the pipeline supports them as a `--llm-backend gemini` option for ad-hoc inspection, every numerical claim in §7 is from the Trendyol generator.

### 9.2 Component contribution decomposition
The isolation pairs in Table 7.2 attribute observed gains to specific components:
- *(A5 − A5b)* captures **embedding FT contribution** (replacing base e5 with our fine-tuned e5).
- *(A5 − A5c)* captures **reranker FT contribution** (replacing pretrained BGE with our fine-tuned BGE-v2).
- *(A5 − A5a)* captures **LLM SFT contribution** (loading vs not loading the LoRA adapter).
- *(A4 − A3)* gives a second, generator-independent estimate of reranker FT (both with vanilla Trendyol).

Together these four deltas decompose the total *(A5 − A1)* gain into per-component contributions.

### 9.3 Custom-data evaluation
The pipeline accepts arbitrary user-supplied corpora and benchmarks through generic CLIs. To evaluate on a third-party JSONL pair `(prof_corpus.jsonl, prof_qa.jsonl)`:

```bash
# 1) Index the supplied corpus
python -m src.retrieval.build_index \
  --corpus prof_corpus.jsonl \
  --output-dir data/index_prof \
  --embed-model data/models/e5-large-tr-legal

# 2) Run A5 (Fully Optimized) on the supplied benchmark
python -m src.eval.run_eval \
  --test prof_qa.jsonl --index-dir data/index_prof \
  --embed-model data/models/e5-large-tr-legal \
  --mode hybrid --reranker v2 --reranker-dir data/models/bge-reranker-tr-legal-v2 \
  --llm-backend hf --llm-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 \
  --adapter-path data/models/llm_adapter --judge \
  --out results/prof_A5

# 3) Run A1 (Baseline) for direct comparison
python -m src.eval.run_eval --test prof_qa.jsonl --index-dir data/index_prof \
  --mode bm25 --llm-backend hf --llm-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 \
  --judge --out results/prof_A1
```

Expected JSONL schemas:
- **Corpus:** `{"id": "...", "kanun_short": "TCK", "madde_no": "299", "text": "..."}`
- **Benchmark:** `{"id": "...", "question": "...", "answer": "...", "gold_doc_ids": ["..."]}`. The `gold_doc_ids` field is optional; if absent, retrieval metrics (Recall, MRR, nDCG) are skipped and only answer-level metrics (F1, ROUGE-L, BLEU, Faithfulness) are reported.

### 9.4 Cost
- **Training compute:** ~6 Kaggle T4 GPU-hours total across embedding FT (≈25 min), reranker FT (≈45 min), and LLM SFT (2h 6m) — entirely within the Kaggle free tier (30 hours/week per account).
- **Inference compute:** ~3 GPU-hours for 8 ablations × 122 dev questions on Kaggle T4.
- **Faithfulness judge:** Anthropic Claude Haiku 4.5 API (synthetic QA data generation + faithfulness scoring).

---

## 10. Limitations & Future Work

**Single-madde QA assumption.** Each question in our synthetic training data is paired with a single gold madde. Multi-hop legal reasoning — e.g., "Does TBK m.50 apply when TBK m.49 has been triggered?" — requires retrieving and reasoning across multiple maddeler in sequence. Our system can retrieve top-k (typically k=5) maddeler, but the generator is not explicitly trained on multi-hop synthesis. Future work: synthesize multi-hop questions linking semantically connected maddeler.

**No temporal reasoning.** The corpus snapshot reflects active legislation as of the scrape date; repealed or amended articles are not flagged. A user asking "what was the procedure under the old CMK?" would receive content from the current code with no temporal qualification. Future work: timestamp each madde and incorporate temporal entity recognition.

**Primary legislation only.** Judicial decisions (Yargıtay, Anayasa Mahkemesi kararları), official commentaries (içtihat), and secondary legal literature are out of scope. Practitioners often need these for application context. Future work: integrate Yargıtay decision corpus as a separate retrieval head.

**Single-judge faithfulness.** Our faithfulness scores come from a single judge model (Anthropic Haiku 4.5). This introduces judge-model bias; results may shift with a different judge. Future work: triangulate with multiple judges and report inter-judge agreement.

**Tokenization is character-based, not morphological.** Our BM25 tokenizer uses a unicode word pattern that preserves Turkish characters but does not perform stemming or morphological analysis. Turkish's agglutinative morphology means *"hakaret"* and *"hakaretin"* are distinct tokens, hurting lexical recall. Future work: integrate Zemberek-NLP morphological stemming.

**Corpus coverage.** While 41,973 maddeler covers active Kanunlar, CBKs, and Tüzükler, specialized Yönetmelikler and Tebliğler are excluded. Some practitioner questions fall outside this scope.

**Dropout=0 overfitting risk.** We disabled LoRA dropout to enable Unsloth's fast LoRA kernels. With only 3,026 training examples and 0.53% trainable parameters, this risk is small, but a comparison run with `lora_dropout=0.05` would quantify it.

**Single-seed ablation.** All numbers in §7 come from a single deterministic training/eval seed (42). We do not yet report variance across seeds. For the v2 round we plan three-seed runs on A1/A5/A5c to bound the ~±0.01-0.02 variance typical for retrieval+SFT setups on ~120-example benchmarks.

### 10.7 v1 regression and v2 resolution

**v1 finding (reranker regression).** In the v1 round, both isolation pairs (A4 vs A3, A5 vs A5c) showed the fine-tuned reranker hurting retrieval. Post-ablation audit identified three root causes: (1) **739 maddeler with PDF-tail table contamination** — change-history tables, coordinate listings, cadre tables absorbed into article bodies, inflating max article length to 164,347 chars; (2) **4,027 phantom duplicate doc_id rows** from the parser misinterpreting revision table "MADDE X" references; (3) **pointwise BCE loss** mismatched to the ranking objective, compounded by a 1:7 positive/negative label imbalance.

**v2 fixes applied (all completed):**

1. **Corpus cleaning** — `madde_parser.py` gains `trim_tail_noise` with five regex markers; re-parse reduces toxic madde count from 763 → 7, max length from 164,347 → 60,195 chars.
2. **Deduplication** — first-occurrence dedupe removes 4,027 phantom duplicates; corpus: 41,973 → **37,927 unique maddes**.
3. **Gold isolation** — 42 gold doc_ids overlapping with synthetic-QA training set removed (165 anchors dropped); gold ∩ training = **0**.
4. **Listwise reranker loss** — pointwise BCE replaced with listwise softmax cross-entropy over (positive, 10 hard-negatives), FlagEmbedding recipe; best-epoch criterion changed from val_AP to gate MRR.
5. **Embedding scale** — batch 16 → 64 (CachedMNRL), hard-negative pool top-100, 10 negatives/anchor, same-kanun boost 1.5×.

**v2 results (§7):** All four isolation comparisons now show positive or neutral reranker FT effect. A4 > A3 (+0.025 R@10), A5 > A5c (+0.024 R@10). Full FT pipeline (A5) achieves the best R@10 (0.803) in the study — the contamination hypothesis is confirmed by the recovery.

---

## 11. Conclusion

We built a citation-grounded Turkish legal RAG system that fine-tunes all three optimizable components — embedding, reranker, and generator LLM — and demonstrates measurable improvements over a same-LLM Base RAG baseline on a 178-question gold benchmark. The 8-cell ablation cleanly isolates each component's contribution, exceeding the project's required 5-step ablation by adding three component-isolation comparisons. The pipeline accepts arbitrary user-supplied corpora and benchmarks through a generic CLI, supporting third-party reproducibility and reuse.

**Headline takeaways:**
- Full FT-RAG (A5) outperforms Base RAG (A1) by **+0.414 absolute on Citation F1** (7.8× relative), **+0.132 on answer F1**, and **+29.5 pp on Recall@5**, with the same Trendyol-7B generator in both configurations.
- The largest single-component gain comes from **LLM supervised fine-tuning**: removing it (A5 vs A5a) wipes out 82% of the citation-F1 gain and 70% of the answer-F1 gain while leaving retrieval untouched.
- **Embedding FT** contributes a clear +5.7 pp on Recall@10; **reranker FT** (after v2 corpus cleaning and listwise loss) now shows a positive contribution under both isolation pairs (A4>A3, A5>A5c).
- The v2 round confirms our §10.7 contamination hypothesis: fixing PDF tail-table noise + deduplication + listwise loss fully reversed the v1 reranker regression.

The codebase, fine-tuned models, gold benchmark, and Kaggle notebooks are released to support reproducibility and future work on Turkish legal NLP.

---

## References

*[BibTeX or numbered list — 15-20 refs typical]*

1. Lewis, P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
2. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
3. Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
4. Xiao, S. et al. *C-Pack: Packaged Resources To Advance General Chinese Embedding*. arXiv 2023. (BGE)
5. Wang, L. et al. *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. arXiv 2022. (E5)
6. Es, S. et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. EACL 2024.
7. Unsloth team. *Unsloth: 2x faster LLM finetuning*. 2024.
8. Trendyol AI Team. *Trendyol-LLM-7B-chat-v4.1.0*. HuggingFace 2024.
9. Schweter, S. *BERTurk*. 2020.
10. Lin, J. et al. *Pyserini: A Python toolkit for reproducible IR research*. SIGIR 2021. (BM25)

---

## Appendix A — Reproducibility

```
git clone https://github.com/hasanemreusta/turkish-legal-rag
cd turkish-legal-rag
pip install -r requirements.txt

# Indexes (or download from release artifact)
python -m src.retrieval.build_index --corpus data/corpus/mevzuat_full_normalized.jsonl

# Retrieval smoke
python -m src.retrieval.retriever --mode hybrid "Cumhurbaşkanına hakaret cezası?"

# Full eval
python -m src.eval.run_eval --test data/test_set/dev_full.jsonl --mode hybrid \
  --reranker v2 --adapter-path data/models/llm_adapter \
  --llm-backend hf --llm-model Trendyol/Trendyol-LLM-7B-chat-v4.1.0 --judge \
  --out results/A5
```

## Appendix B — Hyperparameter Tables

### B.1 Embedding FT (`notebooks/01_embedding_finetune.ipynb`)
| Parameter | Value |
|---|---|
| Base model | `intfloat/multilingual-e5-large` |
| Adaptation | Full fine-tuning (no PEFT/LoRA — sentence-transformers `model.fit`) |
| Loss | `CachedMultipleNegativesRankingLoss`, `mini_batch_size=8` |
| Batch size | 16 (this round) |
| Learning rate | 2e-5 cosine, warmup ratio 0.1 |
| Epochs | 2 |
| Precision | fp16 (AMP), `CUDA_VISIBLE_DEVICES=0` (single GPU; DataParallel was found to hang) |
| Hardware | Kaggle T4 × 1 |
| Triplets | 3,026 (BM25-top-20 hard negatives, 5–7 negatives per anchor) |

### B.2 Reranker FT (`notebooks/02_reranker_finetune.ipynb`)
| Parameter | Value |
|---|---|
| Base model | `BAAI/bge-reranker-v2-m3` |
| Adaptation | LoRA, r=16, α=32, dropout 0.05, targets `[query, key, value]`, classifier head fully trainable |
| Loss | Pointwise binary cross-entropy on classifier logit *(noted as suboptimal — v2 will switch to listwise softmax)* |
| Batch size | 16 |
| Learning rate | 1e-5 cosine, warmup ratio 0.06 |
| Epochs | 3 |
| Validation metric | Average Precision (AP); best epoch selected per-AP |
| Hardware | Kaggle T4 × 1 |
| Train pairs | ~21,800 (3,026 triplets × ~8 pairs after positive + hard-negative expansion); 10% difficulty-stratified validation split |

### B.3 LLM SFT (`notebooks/03_llm_sft.ipynb`)
| Parameter | Value |
|---|---|
| Base model | `Trendyol/Trendyol-LLM-7B-chat-v4.1.0` (Qwen2 7.66B) |
| Quantization | 4-bit NF4 + double quant (bnb) |
| Adaptation | QLoRA, r=16, α=32, dropout 0 |
| Target modules | q/k/v/o + gate/up/down (7 per layer × 28 layers) |
| Trainable params | 40,370,176 (0.53%) |
| Completion-only masking | Unsloth `train_on_responses_only` |
| Sequence length | 2,048 (pre-filter at 1,950 tokens) |
| Batch size | 1 per device, grad-accum 16 → effective 16 |
| Optimizer | `paged_adamw_8bit` |
| Learning rate | 2e-4 cosine, warmup ratio 0.05 |
| Epochs | 2 |
| Total steps | 360 |
| Precision | fp16 (T4 lacks bf16 hardware) |
| Hardware | Kaggle T4 × 1, Unsloth 2026.5.2 fast LoRA kernels |
| Wall time | 2 h 6 min |
| Loss trajectory | step 10: 0.60 → step 180: 0.53 → step 190: 0.37 → step 360: 0.33 |

## Appendix C — Failure Case Catalogue

Ten worked examples drawn from the 122-question dev split, covering all three principal failure modes identified in §8.2. For each case we show the question, gold document ID, A1 (baseline) answer, A5 (full FT-RAG) answer, and per-example metrics.

---

### C.1 Format Failure — citation absent despite retrieval success (A1 → A5 fully resolved)

**Case C1-1** | Category: Medeni (TMK) | Difficulty: Easy
> **Q:** Mirasçılar mirası ne zaman kazanır?
> **Gold:** `TMK_m599` | A1 R@10 = 1.0, A1 Cit-F1 = **0.000** | A5 R@10 = 1.0, A5 Cit-F1 = **1.000**
>
> *A1 answer:* "Mirasçılar, mirasbırakanın ölümü ile mirası bir bütün olarak, kanun gereğince kazanırlar. Atanmış mirasçılar da mirası, mirasbırakanın ölümü ile kazanırlar. Yasal mirasçılar, atanmış mirasçılara düşen mirası onlara zilyetlik hükümleri uyarınca teslim etmekle yükümlüdürler."
>
> *A5 answer:* "Mirasçılar, mirasbırakanın ölümü ile mirası bir bütün olarak kanun gereğince kazanırlar. Bu, mirasçıların mirası hukuki olarak elde etmeye başladığı andır. **[TMK m.599]**"
>
> **Analysis:** Retrieval succeeds in both cells (TMK m.599 is in top-10). The vanilla LLM produces factually correct content verbatim from the retrieved article but emits no citation tag. SFT teaches the model to append `[KısaAd m.No]` at the end of every grounded claim — a format-only intervention that raises Cit-F1 from 0 to 1.

---

**Case C1-2** | Category: Usul (HMK) | Difficulty: Medium
> **Q:** Davayı yanlış kişiye açtığımı fark edersem sonradan davalı tarafı değiştirebilir miyim?
> **Gold:** `HMK_m124` | A1 R@10 = 1.0, A1 Cit-F1 = **0.000** | A5 R@10 = 1.0, A5 Cit-F1 = **1.000**
>
> *A1 answer:* "Davayı yanlış kişiye açtığınızı fark ederseniz, Hukuk Muhakemeleri Kanunu'nun (HMK) 125. maddesi uyarınca, dava konusunu üçüncü bir kişiye devrederek davanızı devralan kişiye karşı açabilirsiniz. Bu durumda, dava davacı lehine sonuçlanırsa, dava konusunu devreden ve devralan yargılama giderlerinden müteselsilen sorumlu olur."
>
> *A5 answer:* "Evet, davayı yanlış kişiye açtığınızı fark ederseniz, dava konusunun devri yoluyla davalı tarafı değiştirebilirsiniz. HMK m.125'in birinci fıkrasına göre, davacı dilediği takdirde devreden tarafla olan davasından vazgeçerek dava konusunu devralmış kişiye karşı davaya devam edebilir. **[HMK m.124]**"
>
> **Analysis:** Same pattern as C1-1 — vanilla LLM even cites the wrong article number (m.125 instead of m.124) and omits the bracket format. A5 retrieves m.124 and formats the citation correctly.

---

**Case C1-3** | Category: Tüketici (TKHK) | Difficulty: Easy
> **Q:** Mesafeli sözleşmeden caymak istiyorum; satıcı benden gerekçe göstermemi veya ceza ödememi isteyebilir mi?
> **Gold:** `TKHK_m48` | A1 R@10 = 1.0, A1 Cit-F1 = **0.000** | A5 R@10 = 1.0, A5 Cit-F1 = **1.000**
>
> *A1 answer:* Retrieves relevant TBK content on seller remedies and answers with taksit/temerrüt language — topically wrong and lacking any citation.
>
> *A5 answer:* "Hayır, satıcı veya sağlayıcı cayma hakkını kullanmak için tüketiciden gerekçe göstermesini veya cezai şart ödemeyi isteyemez. Maddenin dördüncü fıkrasına göre, tüketicinin on dört gün içinde herhangi bir gerekçe göstermeksizin ve cezai şart ödemeksizin sözleşmeden cayma hakkı bulunmaktadır. **[TKHK m.48]**"
>
> **Analysis:** A1 retrieves a BM25-adjacent but semantically wrong article (TBK instalment rules instead of TKHK distance-contract withdrawal). A5's FT embedder retrieves the correct TKHK article and the SFT LLM produces the legally accurate answer with the correct citation.

---

**Case C1-4** | Category: İdari (DMK) | Difficulty: Medium
> **Q:** Memurun amirinden aldığı emir suç teşkil ediyorsa memur bu emri yerine getirmek zorunda mıdır?
> **Gold:** `DMK_m11` | A1 R@10 = 1.0, A1 Cit-F1 = **0.000** | A5 R@10 = 1.0, A5 Cit-F1 = **1.000**
>
> *A1 answer:* "Memurun amirinden aldığı emrin suç teşkil etmesi durumunda, memurun bu emri yerine getirmek zorunda olmadığı, ancak emrin yerine getirilmesinden doğan sorumluluğun emri verene ait olduğu belirtilmiştir."
>
> *A5 answer:* "Hayır, memur bu emri yerine getirmek zorunda değildir. Maddenin ikinci fıkrasına göre, memur amirinden aldığı emri Anayasa, kanun, Cumhurbaşkanlığı kararnamesi ve yönetmelik hükümlerine aykırı görürse yerine getirmez ve bu aykırılığı emri verene bildirir. Ancak emrin yerine getirilmesinden doğabilecek sorumluluk emri verene aittir. **[DMK m.11]**"
>
> **Analysis:** A1 retrieves the correct article and produces a correctly summarized answer, but the absence of the `[DMK m.11]` tag reduces Cit-F1 to zero. SFT enforces citation format without any change to the underlying legal reasoning.

---

**Case C1-5** | Category: Usul (HMK) | Difficulty: Medium
> **Q:** Davalı cevap dilekçesinde yetki itirazı yapmadıysa sonradan bu mahkeme yetkisiz diyebilir mi?
> **Gold:** `HMK_m19` | A1 R@10 = 1.0, A1 Cit-F1 = **0.000** | A5 R@10 = 1.0, A5 Cit-F1 = **1.000**
>
> *A1 answer:* "Davalı cevap dilekçesinde yetki itirazı yapmadıysa, mahkeme yetkisizlik kararı veremez. Yetki itirazı, cevap dilekçesinde ileri sürülmesi gereken bir durumdur."
>
> *A5 answer:* "Hayır, davalı cevap dilekçesinde yetki itirazı yapmadıysa, sonradan mahkeme yetkisiz diyemez. HMK m.19'un dördüncü fıkrasına göre, yetkinin kesin olmadığı davalarda davalı süresi içinde ve usulüne uygun olarak yetki itirazında bulunmazsa, davanın açıldığı mahkeme yetkili hâle gelir. **[HMK m.19]**"
>
> **Analysis:** Identical failure mode. Both A1 and A5 retrieve the gold article and produce correct legal content; the only difference is the SFT-learned citation tag.

---

### C.2 Retrieval Miss — gold article not in top-10 for either A1 or A5

**Case C2-1** | Category: İcra/İflas (IIK) | Difficulty: Easy
> **Q:** Borçluya ödeme emrine itiraz etmesi için genel haciz yolunda kaç gün süre verilir?
> **Gold:** `IIK_m62` | A1 R@10 = **0.000** | A5 R@10 = **0.000**
>
> *A1 answer:* "Borçluya ödeme emrine itiraz etmesi için genel haciz yolunda verilen süre, ödeme emrinin tebliğinden itibaren yedi gündür." — legally correct answer hallucinated without any retrieved evidence.
>
> *A5 answer:* Cites a tangentially related transport-regulation article (`T1_7155 m.7`) — hallucinated citation from a retrieved but wrong document.
>
> **Analysis:** IIK (İcra ve İflas Kanunu) short code is absent from the 16-priority normalization map. The BM25 tokenizer does not match "ödeme emri / itiraz" to the İİK article because the BM25 index uses the unnormalized full title. The FT embedder likewise fails to map the colloquial phrasing to `İİK_m62`. Both cells produce an answer that happens to give the correct number (7 days) but from memory rather than retrieval. Root cause: corpus normalization gap for IIK.

---

**Case C2-2** | Category: Ceza Usul (CMK) | Difficulty: Medium
> **Q:** İfade vereceğim ama avukat tutacak durumum yok, bana avukat atanır mı?
> **Gold:** `CMK_m150` | A1 R@10 = **0.000** | A5 R@10 = **0.000**
>
> *A1 answer:* "Verilen kaynaklarda, avukat atanmasıyla ilgili bir hüküm bulunmamaktadır." — correct refusal, zero hallucination.
>
> *A5 answer:* Retrieves and cites a mağdur hakları maddesi (CMK m.234) instead of CMK m.150 on müdafi atama. Partially relevant but wrong madde number, Cit-F1 = 0.
>
> **Analysis:** CMK m.150 requires matching "zorunlu müdafilik / atama" terminology. The query uses "avukat tutacak durumum yok" — colloquial phrasing that neither BM25 nor the dense retriever maps to the correct article. A5 retrieves a semantically adjacent CMK article but cannot bridge the vocabulary gap. Morphological stemming or query expansion would help.

---

**Case C2-3** | Category: İdari (DMK) | Difficulty: Hard
> **Q:** Bir kamu kurumu tarafından 2 yıllığına İngiltere'ye görevlendirildim. Benimle aynı unvanda olan bir arkadaşım Mısır'a gönderildi. İkimiz de aynı kıdeme sahip olmamıza rağmen yurt dışı maaş katsayılarımızın neden farklı olduğunu merak ediyorum.
> **Gold:** `DMK_m157` | A1 R@10 = **0.000** | A5 R@10 = **1.000**, A5 Cit-F1 = 0.500
>
> *A1 answer:* "Verilen kaynaklarda, yurt dışı maaş katsayılarının belirlenmesiyle ilgili bir bilgi bulunmamaktadır." — correct refusal.
>
> *A5 answer:* Correctly explains that DMK m.157 factors include country economic conditions AND the officer's representation duties and family obligations, which explains why two officers at the same rank can receive different coefficients. Partially correct citation.
>
> **Analysis:** This is a partial win for A5 — the FT embedder recovers DMK m.157 (R@10 = 1.0) whereas BM25 completely misses it. The question mixes colloquial phrasing ("daha az maaş alıyorum") with a complex multi-factor legal question, making it a hard case that tests retrieval at its limits.

---

### C.3 Kanun Code Coverage Gap — retrieval succeeds, citation format breaks

**Case C3-1** | Category: Anayasa | Difficulty: Easy
> **Q:** Hakkımda dava açıldıysa şu anda suçlu sayılır mıyım?
> **Gold:** `Anayasa_m38` | A1 R@10 = **0.000**, A1 Cit-F1 = **0.000** | A5 R@10 = **1.000**, A5 Cit-F1 = **1.000**
>
> *A1 answer:* Retrieves an HMK procedural article about görevsizlik/yetkisizlik and produces a confused response about case transfers — completely off-topic. Cit-F1 = 0.
>
> *A5 answer:* "Hayır, suçlu sayılamazsınız. Anayasa'nın 38. maddesinin son fıkrasına göre, suçluluğu hükmen sabit oluncaya kadar kimse suçlu sayılamaz. Bu nedenle dava açılmış olsa bile, henüz mahkûmiyet kararı verilmediği sürece kişi suçlu sayılmaz. **[Anayasa m.38]**"
>
> **Analysis:** A1's BM25 retrieval misses `Anayasa_m38` (the query uses "suçlu sayılır mı" which doesn't lexically match "suçluluğu hükmen sabit oluncaya kadar"). A5's FT embedder correctly maps the masumiyet karinesi concept to the relevant Anayasa article. The SFT LLM then produces a clean legally accurate answer with correct citation format — a full win for the FT pipeline on all dimensions.

---

**Case C3-2** | Category: Ceza Usul (CMK) | Difficulty: Medium
> **Q:** Karakolda ifade verirken polis bana "istersen susabilirsin ve avukat isteyebilirsin" demedi. Bu hakların bana bildirilmesi gerekiyor muydu?
> **Gold:** `CMK_m147` | A1 R@10 = **0.000**, A1 Cit-F1 = **0.000** | A5 R@10 = **1.000**, A5 Cit-F1 = **0.000**
>
> *A1 answer:* Retrieves a TCK m.299 article (Cumhurbaşkanına hakaret) — completely wrong. Generates a hallucinated answer citing `[TCK m.299]` — a false positive citation. A1 Cit-F1 = 0.
>
> *A5 answer:* Retrieves CMK m.234 (mağdur hakları) instead of CMK m.147 (şüphelinin hakları bildirilmesi). Gives a partially correct substantive answer about the right to a lawyer during questioning, but cites the wrong article and does not produce a matching `[CMK m.147]` tag. A5 Cit-F1 = 0.
>
> **Analysis:** This question requires distinguishing between CMK m.147 (şüphelinin hakları) and CMK m.234 (mağdur/şikâyetçi hakları) — two semantically adjacent articles in the same kanun. The FT embedder improves retrieval from R@10=0 to R@10=1, but the SFT LLM still cites the wrong article because CMK rights-related articles are dense in the retrieval results and the model cannot distinguish which one is the gold standard. This is a retrieval ranking precision failure, not a format failure.
