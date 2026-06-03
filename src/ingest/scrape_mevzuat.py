"""
Scrape Turkish legislation from mevzuat.gov.tr.

Strategy:
1. For each target (mevzuat_no, mevzuat_tur, tertip), try multiple endpoints:
   - PDF: https://www.mevzuat.gov.tr/MevzuatMetin/{tur}.{tertip}.{no}.pdf
   - DOCX: https://www.mevzuat.gov.tr/MevzuatMetin/{tur}.{tertip}.{no}.docx
   - HTML: https://www.mevzuat.gov.tr/mevzuat?MevzuatNo={no}&MevzuatTur={tur}&MevzuatTertip={tertip}
2. PDF is most reliable for full text. Save raw PDF first.
3. Extract text with pdfplumber (preserves layout for madde detection).
4. Run madde_parser to produce article-level chunks.
5. Emit a JSONL file with one madde per line + a metadata file.

Run locally (sandbox cannot reach mevzuat.gov.tr — TR IP required):
    python -m src.ingest.scrape_mevzuat --mode pilot
    python -m src.ingest.scrape_mevzuat --mode full
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm

from .madde_parser import parse_kanun_text
from .mevzuat_targets import PRIORITY_KANUNLAR, MEVZUAT_TUR_CODES

# ---- Configuration ----------------------------------------------------------

BASE = "https://www.mevzuat.gov.tr"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

DEFAULT_SLEEP_SEC = 2.0  # rate-limit between requests (be polite)
RETRY_COUNT = 3
RETRY_BACKOFF = 5.0  # seconds, exponential

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "scrape" / "raw"
CORPUS_DIR = DATA_DIR / "corpus"

log = logging.getLogger("scrape_mevzuat")


# ---- HTTP helpers -----------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(REQUEST_HEADERS)
    return s


def _get_with_retry(session: requests.Session, url: str, timeout: int = 30) -> Optional[requests.Response]:
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            if r.status_code == 404:
                return None
            log.warning("GET %s → %s (attempt %d)", url, r.status_code, attempt)
        except requests.RequestException as e:
            log.warning("GET %s failed: %s (attempt %d)", url, e, attempt)
        time.sleep(RETRY_BACKOFF * attempt)
    return None


# ---- Endpoint builders ------------------------------------------------------

def pdf_url(no: int, tur: int, tertip: int) -> str:
    # Pattern: /MevzuatMetin/{tur}.{tertip}.{no}.pdf
    return f"{BASE}/MevzuatMetin/{tur}.{tertip}.{no}.pdf"


def docx_url(no: int, tur: int, tertip: int) -> str:
    return f"{BASE}/MevzuatMetin/{tur}.{tertip}.{no}.docx"


def html_url(no: int, tur: int, tertip: int) -> str:
    return f"{BASE}/mevzuat?MevzuatNo={no}&MevzuatTur={tur}&MevzuatTertip={tertip}"


# ---- PDF text extraction ----------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF, preserving layout for madde detection."""
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit("Install pdfplumber: pip install pdfplumber")

    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            parts.append(t)
    return "\n".join(parts)


# ---- Single-target scrape ---------------------------------------------------

def scrape_one(target: tuple, session: requests.Session, force: bool = False) -> Optional[dict]:
    """
    target = (mevzuat_no, mevzuat_tur, tertip, short_name, full_name)
    Returns metadata dict with paths + parse stats, or None on failure.
    """
    no, tur, tertip, short_name, full_name = target
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = RAW_DIR / f"{short_name}_{no}.pdf"
    if pdf_path.exists() and not force:
        log.info("PDF already exists, reusing: %s", pdf_path.name)
    else:
        url = pdf_url(no, tur, tertip)
        log.info("Downloading: %s (%s)", full_name, url)
        resp = _get_with_retry(session, url)
        if not resp:
            log.error("Failed to download %s — skipping", full_name)
            return None
        pdf_path.write_bytes(resp.content)
        time.sleep(DEFAULT_SLEEP_SEC)  # rate limit

    # Extract + parse
    try:
        text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        log.error("PDF extract failed for %s: %s", full_name, e)
        return None

    maddeler = parse_kanun_text(text, kanun_kisa_ad=short_name)
    if not maddeler:
        log.warning("No maddeler parsed for %s — check PDF/parser", full_name)

    return {
        "kanun_no": no,
        "mevzuat_tur": tur,
        "mevzuat_tur_label": MEVZUAT_TUR_CODES.get(tur, "Bilinmiyor"),
        "tertip": tertip,
        "short_name": short_name,
        "full_name": full_name,
        "url": html_url(no, tur, tertip),
        "pdf_path": str(pdf_path.relative_to(DATA_DIR.parent)),
        "raw_text_length": len(text),
        "madde_count": len(maddeler),
        "maddeler": [m.to_dict() for m in maddeler],
    }


# ---- Output writers ---------------------------------------------------------

def write_corpus_jsonl(records: Iterable[dict], output_path: Path):
    """
    Flatten per-kanun records into one-madde-per-line JSONL for indexing.
    Each line has fields suitable for retrieval + citation:
      doc_id, kanun_short, kanun_full, kanun_no, madde_no, madde_type,
      baslik, text, url
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            if not rec:
                continue
            for m in rec["maddeler"]:
                doc_id = f"{rec['short_name']}_m{m['madde_no']}".replace(" ", "_")
                line = {
                    "doc_id": doc_id,
                    "kanun_short": rec["short_name"],
                    "kanun_full": rec["full_name"],
                    "kanun_no": rec["kanun_no"],
                    "madde_no": m["madde_no"],
                    "madde_type": m["madde_type"],
                    "baslik": m.get("baslik"),
                    "text": m["metin"],
                    "fikralar": m.get("fikralar", []),
                    "url": rec["url"],
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    log.info("Wrote %d madde records to %s", written, output_path)


def write_kanun_metadata(records: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        {
            "short_name": r["short_name"],
            "full_name": r["full_name"],
            "kanun_no": r["kanun_no"],
            "madde_count": r["madde_count"],
            "raw_text_length": r["raw_text_length"],
            "url": r["url"],
        }
        for r in records if r
    ]
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote metadata for %d kanunlar to %s", len(summary), output_path)


# ---- Modes ------------------------------------------------------------------

def run_pilot(force: bool = False):
    """Pilot: scrape just 3 well-known kanunlar (TCK, TMK, İş K.) for validation."""
    pilot_short_names = {"TCK", "TMK", "IsK"}
    targets = [t for t in PRIORITY_KANUNLAR if t[3] in pilot_short_names]
    log.info("Pilot scrape: %d kanunlar", len(targets))
    return _run(targets, output_suffix="pilot", force=force)


def run_priority(force: bool = False):
    """Run on the 15 curated priority kanunlar."""
    log.info("Priority scrape: %d kanunlar", len(PRIORITY_KANUNLAR))
    return _run(PRIORITY_KANUNLAR, output_suffix="priority", force=force)


def _load_active_list(tur: int) -> list[tuple]:
    """Load the active list JSON for a given tur and convert to scrape targets."""
    path = DATA_DIR / "scrape" / f"active_list_tur{tur}.json"
    if not path.exists():
        log.error(
            "active_list_tur%d.json not found at %s — run fetch_active_list first:\n"
            "    python -m src.ingest.fetch_active_list --tur %d",
            tur, path, tur,
        )
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    targets = []
    for r in records:
        try:
            no = int(r["mevzuatNo"])
        except (TypeError, ValueError):
            continue
        tertip = int(r.get("mevzuatTertip") or 5)
        ad = r.get("mevAdi", f"Mevzuat_{no}")
        short = f"T{tur}_{no}"
        targets.append((no, tur, tertip, short, ad))
    return targets


def run_full(force: bool = False, turs: list[int] | None = None):
    """
    Full active-mevzuat scrape across requested tur codes.
    Default: Kanunlar (1) + CBK (2) + Tüzükler (4) — primary legislation.
    """
    turs = turs or [1, 2, 4]
    all_targets = []
    for tur in turs:
        ts = _load_active_list(tur)
        log.info("Loaded %d targets from tur=%d", len(ts), tur)
        all_targets.extend(ts)
    if not all_targets:
        raise SystemExit("No targets loaded — run fetch_active_list first.")
    log.info("Full scrape: %d documents across turs=%s", len(all_targets), turs)
    return _run(all_targets, output_suffix="full", force=force)


def _run(targets, output_suffix: str, force: bool):
    session = _session()
    records = []
    for t in tqdm(targets, desc="scraping"):
        rec = scrape_one(t, session, force=force)
        if rec:
            records.append(rec)

    corpus_path = CORPUS_DIR / f"mevzuat_{output_suffix}.jsonl"
    meta_path = CORPUS_DIR / f"mevzuat_{output_suffix}_meta.json"
    write_corpus_jsonl(records, corpus_path)
    write_kanun_metadata(records, meta_path)

    total_madde = sum(r["madde_count"] for r in records)
    print(f"\n[OK] Done. {len(records)} kanunlar, {total_madde} maddeler total.")
    print(f"  Corpus:   {corpus_path}")
    print(f"  Metadata: {meta_path}")


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape mevzuat.gov.tr")
    parser.add_argument(
        "--mode",
        choices=["pilot", "priority", "full"],
        default="pilot",
        help="pilot=3 kanun, priority=16 kanun, full=all active (from fetch_active_list output)",
    )
    parser.add_argument(
        "--turs", type=int, nargs="+", default=[1, 2, 4],
        help="Mevzuat tur codes to include in --mode full (default: 1=Kanun, 2=CBK, 4=Tüzük)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if PDF exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.mode == "pilot":
        run_pilot(force=args.force)
    elif args.mode == "priority":
        run_priority(force=args.force)
    elif args.mode == "full":
        run_full(force=args.force, turs=args.turs)


if __name__ == "__main__":
    main()
