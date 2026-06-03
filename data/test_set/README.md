# Gold Test Set — Yazım Rehberi

CENG493 Türkçe Hukuki RAG projesi için **150 soru** elle yazılacak. Bu set RAG sisteminin gerçek performansını ölçmek için kullanılacak ("altın standart"). Sentetik QA datasından **tamamen ayrı** tutulmalı, aksi takdirde eval skorları şişer.

## Neden manuel yazılıyor?

RAG sistemini Anthropic Haiku ile üretilen sentetik QA üzerinde eğiteceğiz. Eğer eval de LLM-yazımı olursa training distribution = eval distribution olur → yanlı sonuçlar.

**Gold set'in farklı olması şart:**
- İnsan tarafından yazılmış cümle yapısı
- Gerçek hayatta avukat/vatandaş tarzında doğal sorular
- Yargıtay kararları veya hukuk forumlarından esinlenen senaryolar

## ⚠️ ChatGPT/Claude kullanım kuralları

**YASAK**: Doğrudan "şu madde için 10 soru üret" deyip kopyalamak. Bu sentetik QA setiyle aynı dağılıma düşer, eval'i bozar.

**İZİNLİ** (akıllı hibrit yaklaşım):
1. Önce kanun maddesini oku
2. **Kendi cümlelerinle 1-2 soru yaz**
3. ChatGPT'ye sor: "Bu madde hakkında **benim aklıma gelmeyen** 5 farklı açıdan soru üret" → brainstorm aracı olarak
4. ChatGPT çıktısından 1-2 fikri seç, **kendi diline çevir**
5. Cevabı manuel yaz, madde metnine bak

**En iyi yöntem**: Türk hukuk forumlarından (avukat.net, ekşi sözlük) gerçek soru tarzını oku, uyarla.

## Soru kalitesi kriterleri

Her soru şunlara sahip olmalı:

| Kriter | Açıklama |
|---|---|
| Doğal Türkçe | Avukat/vatandaşın gerçekte soracağı dil |
| Madde-bağlı cevap | Cevap belirli bir madde(ler)den çıkarılabilmeli |
| Net citation | Cevap formatı: `... [TCK m.299]` |
| Difficulty etiketi | `lookup` / `reasoning` / `edge` / `multi_hop` / `no_answer` |

### Difficulty tipleri

- **lookup** (~%40): Maddeden doğrudan okunabilen olgusal soru. "X süresi nedir?" "Y cezası ne?"
- **reasoning** (~%30): Senaryoyu yoruma sokmayı gerektirir. "A kişisi X yaparsa madde gereği ne olur?"
- **edge** (~%20): İstisna, sınır durumu, muafiyet. "X koşulu sağlanmazsa ne olur?"
- **multi_hop** (~%5): Birden fazla maddenin birleşimi gerekir.
- **no_answer** (~%5): Cevap corpus'ta YOK. Modelden "Verilen kaynaklarda bu sorunun cevabı bulunmamaktadır." beklenir. Hallucination kontrolü.

## Format (JSONL)

Her soru bir JSON satırı, `dev.jsonl` veya `test.jsonl` içinde:

```json
{
  "id": "q001",
  "question": "Cumhurbaşkanına hakaret suçunun cezası nedir?",
  "answer": "Cezası bir yıldan dört yıla kadar hapistir. [TCK m.299]",
  "gold_doc_ids": ["TCK_m299"],
  "gold_citations": [["TCK", "299"]],
  "difficulty": "lookup",
  "category": "ceza",
  "author": "hasan",
  "notes": "düz olgusal soru"
}
```

**Alan açıklamaları:**
- `id`: q001, q002 ... (artan sıra, kişi başına ayrı: q001-q040 hasan, q041-q080 ali...)
- `question`: doğal Türkçe soru cümlesi
- `answer`: doğru cevap + citation
- `gold_doc_ids`: ilgili madde doc_id listesi (`<KISA_AD>_m<NO>`). Multi-hop için birden fazla
- `gold_citations`: `[["KISA_AD", "MADDE_NO"], ...]` — citation eval için
- `difficulty`: yukarıdaki 5 tipten biri
- `category`: ceza / medeni / borçlar / ticaret / icra / vergi / idari / sosyal güvenlik
- `author`: kim yazdı (kalite kontrolü için)
- `notes`: opsiyonel — neyi test ediyor

**`no_answer` durumunda**: `gold_doc_ids: []`, `gold_citations: []`, `answer: "Verilen kaynaklarda bu sorunun cevabı bulunmamaktadır."`

## Doğru madde nasıl bulunur?

1. **mevzuat.gov.tr** üzerinden ara
2. Veya local corpus'tan: kanun adıyla grep at:
   ```powershell
   Select-String -Path "data/corpus/mevzuat_full_normalized.jsonl" -Pattern '"TCK_m299"'
   ```
3. doc_id'yi memorial format: `<KISA_AD>_m<MADDE_NO>` (örn `TCK_m299`, `TMK_m285`, `Anayasa_m38`)
4. Geçici maddeler: `TCK_m_Gecici_5` veya `TCK_mGeçici 5` (parser'ın ürettiği format)

## İş bölümü (4 kişi)

Her kişi **40 soru** yazsın → toplam 160, hedef 150 (kalite filtresi sonrası).

| Kişi | Sorumlu kanunlar | id aralığı |
|---|---|---|
| **A (Hasan)** | Anayasa, TCK, CMK, KVKK | q001-q040 |
| **B (Gökay)** | TMK, TBK, HMK, TKHK | q041-q080 |
| **C** | TTK, IIK, IYUK, IsK | q081-q120 |
| **D** | VUK, DMK, KTK, SSGSS | q121-q160 |

Her kişi kanun başına ~10 soru, difficulty dağılımı:
- 4 lookup + 3 reasoning + 2 edge + 1 multi_hop/no_answer

## Kalite kontrol süreci

1. Her kişi 40 sorusunu yazıp `data/test_set/dev_<initials>.jsonl` olarak commit eder
2. **Karşılıklı review**: A'nın sorularını B kontrol eder, B'ninkini C, vs.
3. Reviewer şunları kontrol eder:
   - Soru doğal Türkçe mi? (yapay/ChatGPT-vari değil)
   - Cevap gerçekten o maddeden çıkıyor mu?
   - Citation doğru mu (kanun + madde numarası)?
   - Difficulty etiketi uygun mu?
4. Düzeltmeler sonrası tüm dosyalar birleştirilir → `dev.jsonl` (100 soru) + `test.jsonl` (50 soru)

## Eval'de nasıl kullanılır?

Eval harness zaten hazır:

```powershell
python -m src.eval.run_eval --test data/test_set/dev.jsonl --no-llm  # retrieval-only
python -m src.eval.run_eval --test data/test_set/dev.jsonl --llm-backend anthropic --judge
```

Çıktı: `results/eval_<config>/{per_example.jsonl,summary.json}` — Recall@k, MRR, citation accuracy, faithfulness vs.

## Süre tahmini

- Soru başı ortalama: 5-10 dakika (madde okuma + ChatGPT brainstorm + manuel yazım + cevap doğrulama)
- 40 soru/kişi × 7 dk = ~4-5 saat
- 1-2 günlük dağıtık iş (paralel)

## Yardımcı dosyalar

- [examples.jsonl](examples.jsonl) — 10 örnek soru (Hasan tarafından yazıldı, referans için)
- [template.jsonl](template.jsonl) — boş şablon, üzerine yaz
- [assignments.md](assignments.md) — iş dağılımı detayı
