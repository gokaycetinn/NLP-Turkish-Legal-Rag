"""
Parse Turkish legal text into article-level (madde) chunks.

Input: raw text of a kanun (after PDF/HTML extraction)
Output: list of dicts with {madde_no, baslik, metin, fikralar} per article.

Strategy:
- Detect MADDE markers via regex (multiple variants: "MADDE 1", "Madde 1-", "Geçici Madde 1")
- Each article extends until the next MADDE marker or end-of-text
- Fıkralar (paragraphs) are split by (1), (2), (3)... pattern inside an article
- Optional cleanup of headers/footers/page numbers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# Madde marker patterns. Order matters — more specific first.
MADDE_PATTERNS = [
    # "Geçici Madde 1", "GEÇİCİ MADDE 1", with optional dash and title
    re.compile(
        r"(?im)^\s*(?P<full>(?:GEÇİCİ|Geçici|GECICI|Gecici)\s+MADDE\s+(?P<no>\d+)[\s\.\-–]*)",
        re.MULTILINE,
    ),
    # "Ek Madde 1" / "EK MADDE 1"
    re.compile(
        r"(?im)^\s*(?P<full>(?:EK|Ek)\s+MADDE\s+(?P<no>\d+)[\s\.\-–]*)",
        re.MULTILINE,
    ),
    # Standard "MADDE 1", "Madde 1-", "MADDE 1." (most common)
    re.compile(
        r"(?im)^\s*(?P<full>MADDE\s+(?P<no>\d+)[\s\.\-–]*)",
        re.MULTILINE,
    ),
]

# Fıkra (paragraph) marker: "(1)", "(2)" at line start
FIKRA_PATTERN = re.compile(r"^\s*\((\d+)\)\s*", re.MULTILINE)

# Common page-footer junk in mevzuat PDFs
NOISE_PATTERNS = [
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # standalone page numbers
    re.compile(r"^.{0,80}Sayfa\s*:\s*\d+.*$", re.MULTILINE | re.IGNORECASE),
]

# PDF tail-attached tables that get absorbed into the final madde body.
# Yürürlük maddesi ("Bu Kanun hükümlerini Cumhurbaşkanı yürütür.") sonrası
# PDF'in son sayfalarında iliştirilmiş değişiklik/cetvel/koordinat tabloları,
# son madde gövdesine yapışıyor. 739 madde (corpus'un %1.8'i) bu kontaminasyonu
# taşıyor ve max madde uzunluğunu 164,347 char'a çıkarıyor.
TAIL_NOISE_MARKERS = [
    # "X SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN/..." — değişiklik tablosu
    re.compile(r"\d+\s+SAYILI\s+KANUNA\s+EK\s+VE\s+DE[ĞG]İ[ŞS]İKLİK\s+GETİREN", re.IGNORECASE),
    # "(I) SAYILI CETVEL" / "I SAYILI LİSTE" / "(1) SAYILI LİSTE" — cetvel/liste tabloları
    # Cover all variants: with/without parens, Roman/digit, CETVEL or LİSTE
    re.compile(r"(?:^|\n)\s*\(?\s*[IVX\d]+\s*\)?\s+SAYILI\s+(?:CETVEL|L[İI]STE)", re.IGNORECASE),
    # "KOORDİNAT LİSTESİ" — coğrafi alan kanunlarında X/Y koordinat tabloları
    re.compile(r"KOORD[İI]NAT\s+L[İI]STES[İI]", re.IGNORECASE),
    # "Nokta No y X" — koordinat tablosu başlığı (alternatif giriş noktası)
    re.compile(r"^\s*Nokta\s+No\s+[yY]\s+[xX]\s*$", re.MULTILINE),
    # "YÜRÜRLÜĞE GİRİŞ TARİHLERİNİ GÖSTERİR ..." — geçiş tablosu başlığı
    re.compile(r"Y[ÜU]R[ÜU]RL[ÜU][ĞG]E\s+G[İI]R[İI][ŞS]\s+TAR[İI]HLER[İI]N[İI]\s+G[ÖO]STER[İI]R", re.IGNORECASE),
]


def trim_tail_noise(text: str) -> str:
    """Madde gövdesinden PDF kuyruk tablosu kontaminasyonunu kes.

    Birden fazla marker varsa en erken konumdaki kullanılır.
    Marker bulunamazsa metin değişmeden döner.
    """
    earliest = len(text)
    for pat in TAIL_NOISE_MARKERS:
        m = pat.search(text)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest].rstrip()


@dataclass
class Madde:
    madde_no: str            # e.g. "1", "Geçici 3", "Ek 5"
    madde_type: str          # "normal" | "gecici" | "ek"
    baslik: Optional[str]    # article heading line if present
    metin: str               # full article text
    fikralar: List[str] = field(default_factory=list)  # paragraph splits

    def to_dict(self):
        return asdict(self)


def clean_text(text: str) -> str:
    """Strip page numbers, headers, normalize whitespace."""
    for pat in NOISE_PATTERNS:
        text = pat.sub("", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _classify(full_marker: str) -> str:
    f = full_marker.lower()
    if "geçici" in f or "gecici" in f:
        return "gecici"
    if f.startswith("ek "):
        return "ek"
    return "normal"


def find_all_madde_markers(text: str):
    """
    Return list of (start_pos, end_pos_of_marker, madde_no, madde_type, full_marker)
    sorted by start position. Deduplicates overlapping matches (since "Geçici Madde"
    contains "Madde" — we keep the most specific match per position).
    """
    matches = []
    seen_positions = set()

    # Order matters: process geçici/ek before plain to claim them first
    for pat in MADDE_PATTERNS:
        for m in pat.finditer(text):
            start = m.start()
            # Skip if a more-specific match already claimed this region
            overlap = False
            for s in seen_positions:
                if abs(start - s) < 5:
                    overlap = True
                    break
            if overlap:
                continue
            seen_positions.add(start)
            full = m.group("full").strip()
            no = m.group("no")
            mtype = _classify(full)
            matches.append((start, m.end(), no, mtype, full))

    matches.sort(key=lambda x: x[0])
    return matches


def split_fikralar(article_text: str) -> List[str]:
    """Split article body into numbered paragraphs (fıkra) if pattern (1) (2) found."""
    parts = FIKRA_PATTERN.split(article_text)
    if len(parts) < 3:
        return [article_text.strip()] if article_text.strip() else []

    # parts = [pre, '1', body1, '2', body2, ...]
    fikralar = []
    # Discard pre-text before first (1)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            fikralar.append(parts[i + 1].strip())
    return [f for f in fikralar if f]


def parse_kanun_text(raw_text: str, kanun_kisa_ad: str = "") -> List[Madde]:
    """
    Parse the full text of a kanun into a list of Madde objects.
    """
    text = clean_text(raw_text)
    markers = find_all_madde_markers(text)
    if not markers:
        return []

    maddeler: List[Madde] = []
    for i, (start, marker_end, no, mtype, full) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        body = text[marker_end:end].strip()
        body = trim_tail_noise(body)

        # First line after marker is often the heading
        first_line, _, rest = body.partition("\n")
        baslik = None
        if first_line and len(first_line) < 120 and not FIKRA_PATTERN.match(first_line):
            baslik = first_line.strip()
            metin = rest.strip() if rest else first_line.strip()
        else:
            metin = body

        if not metin:
            continue

        # Prefix madde_no with type if non-standard
        prefix = ""
        if mtype == "gecici":
            prefix = "Geçici "
        elif mtype == "ek":
            prefix = "Ek "

        maddeler.append(
            Madde(
                madde_no=f"{prefix}{no}",
                madde_type=mtype,
                baslik=baslik,
                metin=metin,
                fikralar=split_fikralar(metin),
            )
        )

    return maddeler


if __name__ == "__main__":
    # Quick self-test on a synthetic input
    sample = """
    MADDE 1 - Amaç
    (1) Bu Kanunun amacı kişilerin hak ve özgürlüklerini korumaktır.
    (2) Aynı zamanda kamu düzenini sağlamayı amaçlar.

    MADDE 2 - Kapsam
    Bu Kanun tüm gerçek ve tüzel kişileri kapsar.

    Geçici Madde 1
    (1) Bu Kanunun yürürlüğe girmesinden önceki olaylar hakkında eski hükümler uygulanır.

    Ek Madde 1 - İlave hükümler
    Bu maddeyle ek düzenlemeler getirilmiştir.
    """

    maddeler = parse_kanun_text(sample, "TEST")
    for m in maddeler:
        print(f"--- Madde {m.madde_no} ({m.madde_type}) ---")
        if m.baslik:
            print(f"Başlık: {m.baslik}")
        print(f"Metin: {m.metin[:100]}...")
        print(f"Fıkra sayısı: {len(m.fikralar)}")
        print()
