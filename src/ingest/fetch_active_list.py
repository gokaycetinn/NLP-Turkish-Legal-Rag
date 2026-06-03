"""
Fetch the master list of all active Turkish legislation from mevzuat.gov.tr.

Uses the DataTables AJAX endpoint discovered by inspecting the page bundle:
    POST /anasayfa/MevzuatDatatable

Response is paginated JSON; we iterate over all pages and aggregate.

Output: data/scrape/active_list_{tur}.json
Each entry: {mevzuatNo, mevAdi, kabulTarih, mevzuatTertip, mevzuatTur, url}
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

BASE = "https://www.mevzuat.gov.tr"
ENDPOINT = f"{BASE}/anasayfa/MevzuatDatatable"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Content-Type": "application/json; charset=utf-8",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE}/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
}

# MevzuatTur codes for the listing endpoint
TUR_CODES = {
    1: "Kanunlar",
    2: "Cumhurbaşkanlığı Kararnameleri",
    3: "KHK / Kanun Hükmünde Kararnameler",  # may need verification
    4: "Tüzükler",
    7: "Yönetmelikler",
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "scrape"

log = logging.getLogger("fetch_active_list")


# Server rejects length>200 with HTTP 600 "FormValidate"
MAX_PAGE_SIZE = 200


def fetch_page(session: requests.Session, mevzuat_tur: int, start: int = 0, length: int = MAX_PAGE_SIZE) -> dict:
    """Fetch a single page from the DataTable endpoint."""
    body = {
        "draw": 1,
        "start": start,
        "length": length,
        "parameters": {
            "AranacakIfade": "",
            "AranacakYer": "2",
            "TamCumle": False,
            "MevzuatTur": mevzuat_tur,
            "GenelArama": True,
        },
    }
    resp = session.post(ENDPOINT, data=json.dumps(body), timeout=60)
    if resp.status_code != 200:
        log.error("HTTP %d at start=%d length=%d: %s", resp.status_code, start, length, resp.text[:100])
        resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError:
        log.error("Non-JSON response: %s", resp.text[:200])
        raise


def fetch_all_for_tur(session: requests.Session, mevzuat_tur: int) -> list[dict]:
    """Fetch all records for a given mevzuat type."""
    records: list[dict] = []
    first = fetch_page(session, mevzuat_tur, start=0, length=1)
    total = first.get("recordsTotal", 0)
    log.info("Tur=%d → recordsTotal=%d", mevzuat_tur, total)

    if total == 0:
        return records

    page_size = MAX_PAGE_SIZE
    start = 0
    consecutive_failures = 0
    while start < total:
        try:
            data = fetch_page(session, mevzuat_tur, start=start, length=page_size)
            chunk = data.get("data", [])
            records.extend(chunk)
            log.info("  Got %d records (cumulative %d/%d)", len(chunk), len(records), total)
            if not chunk:
                break
            consecutive_failures = 0
            start += page_size
            time.sleep(1.0)
        except Exception as e:
            consecutive_failures += 1
            log.warning("Page failed at start=%d (attempt %d): %s", start, consecutive_failures, str(e)[:200])
            if consecutive_failures >= 3:
                # Try smaller page size
                if page_size > 50:
                    log.warning("Reducing page_size %d -> 50", page_size)
                    page_size = 50
                    consecutive_failures = 0
                    continue
                log.error("Giving up at start=%d", start)
                break
            time.sleep(5.0)

    return records


def normalize_record(r: dict) -> dict:
    """Keep only the fields we need."""
    return {
        "mevzuatNo": r.get("mevzuatNo"),
        "mevAdi": r.get("mevAdi"),
        "kabulTarih": r.get("kabulTarih"),
        "resmiGazeteTarihi": r.get("resmiGazeteTarihi"),
        "resmiGazeteSayisi": r.get("resmiGazeteSayisi"),
        "mevzuatTertip": r.get("mevzuatTertip"),
        "mevzuatTur": r.get("mevzuatTur"),
        "mevzuatTurEnumString": r.get("mevzuatTurEnumString"),
        "url": r.get("url"),
    }


def save(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([normalize_record(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %d records to %s", len(records), path)


def main():
    parser = argparse.ArgumentParser(description="Fetch active legislation list from mevzuat.gov.tr")
    parser.add_argument(
        "--tur",
        type=int,
        nargs="+",
        default=[1],
        help="MevzuatTur codes (1=Kanun, 2=CBK, 3=KHK, 4=Tüzük, 7=Yönetmelik)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    session = requests.Session()
    session.headers.update(HEADERS)

    for tur in args.tur:
        records = fetch_all_for_tur(session, tur)
        out_path = DATA_DIR / f"active_list_tur{tur}.json"
        save(records, out_path)


if __name__ == "__main__":
    main()
