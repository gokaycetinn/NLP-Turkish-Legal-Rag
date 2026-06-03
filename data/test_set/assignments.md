# Gold Test Set — İş Dağılımı

**Hedef:** 4 kişi × 40 soru = 160 soru → kalite filtresi sonrası 150 final.

## Dağılım

| Kişi | id aralığı | Sorumlu kanunlar | Yaklaşık madde sayısı |
|---|---|---|---|
| **A — Hasan** | q001-q040 | Anayasa, TCK, CMK, KVKK | Anayasa 205 + TCK 350 + CMK 363 + KVKK 36 |
| **B - Gökay** | q041-q080 | TMK, TBK, HMK, TKHK | TMK 1032 + TBK 652 + HMK 465 + TKHK 95 |
| **C** | q081-q120 | TTK, IIK, IYUK, IsK | TTK 1560 + IIK 500 + IYUK 86 + IsK 141 |
| **D** | q121-q160 | VUK, DMK, KTK, SSGSS | VUK 498 + DMK 878 + KTK 200 + SSGSS 337 |

Her kişi sorumlu olduğu **4 kanun** için ~10 soru yazsın. Tipler:
- 4 lookup (direkt olgusal)
- 3 reasoning (senaryoya uygulama)
- 2 edge (istisna/sınır)
- 1 multi_hop veya no_answer

## Adım adım iş akışı

### 1. Kanun listesini al
Her kişi için 4 kanun. Önce hangi kanundan kaç soru yazacağını planla (eşit dağıt).

### 2. Maddeleri keşfet
mevzuat.gov.tr veya lokal corpus:
```powershell
# Kanun başına maddeleri görmek için (örn TCK):
Select-String -Path "data/corpus/mevzuat_full_normalized.jsonl" -Pattern '"kanun_short": "TCK"' | Select-Object -First 20
```

### 3. Her soru için:
1. **Madde oku** (300-500 char ana içerik)
2. **2-3 dakika düşün**: avukat/vatandaş bu maddeyi gerçek hayatta nasıl sorardı?
3. **Kendi cümlenle yaz** — örn "X yaparsam ne olur?", "Y koşulu nedir?", "Z hakkım var mı?"
4. (Opsiyonel) **ChatGPT brainstorm**: "Bu madde hakkında **aklıma gelmeyen** 3 farklı açıdan soru üret" — sadece fikir için, kopyalama
5. **Cevabı manuel yaz**: madde metninden alıntı + kendi özetleme + sonunda `[KISA_AD m.NO]`
6. **doc_id'yi doğrula**: 
   ```powershell
   Select-String -Path "data/corpus/mevzuat_full_normalized.jsonl" -Pattern '"doc_id": "TCK_m299"' | Select-Object -First 1
   ```

### 4. Kendi dosyanı yaz
- `data/test_set/dev_<initials>.jsonl` (örn `dev_hasan.jsonl`)
- 40 satır, her satır 1 JSON kayıt
- Format: bkz. [examples.jsonl](examples.jsonl)

### 5. Karşılıklı review
- A → B'nin sorularını kontrol
- B → C'nin
- C → D'nin
- D → A'nın
- Yorum yaz: `notes` alanına ekle veya markdown'da liste

## Süre hedefi

- Soru başı: 5-10 dk (madde okuma + brainstorm + yazım + doğrulama)
- 40 soru × 7 dk = ~4-5 saat
- 1-2 günde tamamlanır (paralel)

## Kalite kriterleri (review checklist)

- [ ] Soru doğal Türkçe, gerçek bir kişinin sorabileceği gibi
- [ ] ChatGPT'den kopyalanmış değil (cümle yapısı tek-düze değil)
- [ ] Cevap gerçekten o maddeden çıkarılabilir
- [ ] Citation format doğru: `[KISA_AD m.NO]` (örn `[TCK m.299]`)
- [ ] doc_id corpus'ta gerçekten var
- [ ] Difficulty etiketi uygun (lookup/reasoning/edge/multi_hop/no_answer)
- [ ] no_answer sorularında cevap "Verilen kaynaklarda..." formatında

## Birleştirme (final)

Tüm dosyalar geldikten sonra:

```powershell
# Tüm dev_*.jsonl dosyalarını birleştir
Get-Content data/test_set/dev_*.jsonl | Set-Content data/test_set/all_raw.jsonl

# 100 soru dev + 50 test split
# (sonra script ile rastgele böleceğiz)
```

## Notlar

- Sorularda gerçek isim/kişisel veri kullanma (genel "X kişisi", "bir avukat" gibi)
- Çok yeni güncel kanunlar (2024-2026 değişiklikleri) için temkinli ol — corpus güncel mi kontrol et
- Politik/dini hassas sorulardan kaçın — projeyi notlandırmada zorlaştırır
