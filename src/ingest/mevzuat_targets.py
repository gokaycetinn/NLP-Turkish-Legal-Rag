"""
Curated priority list of Turkish primary legislation to scrape from mevzuat.gov.tr.

Used for the pilot scrape (Hafta 1). The full active-kanunlar list is discovered
dynamically via fetch_active_kanun_list() in scrape_mevzuat.py.

Each entry: (mevzuat_no, mevzuat_tur, tertip, short_name, full_name)
- mevzuat_tur: 1=Kanun, 4=KHK, see MEVZUAT_TUR_CODES
- tertip: 5=Aktüel (current/active)
"""

MEVZUAT_TUR_CODES = {
    1: "Kanun",
    2: "Cumhurbaşkanlığı Kararnamesi",
    3: "Yönetmelik",
    4: "KHK",
    5: "Tüzük",
    7: "Yönetmelik (Bakanlık)",
}

# Anayasa is treated separately — it has MevzuatNo=2709
ANAYASA = (2709, 1, 5, "Anayasa", "Türkiye Cumhuriyeti Anayasası")

# 15 priority kanunlar for pilot scrape
PRIORITY_KANUNLAR = [
    ANAYASA,
    (5237, 1, 5, "TCK", "Türk Ceza Kanunu"),
    (5271, 1, 5, "CMK", "Ceza Muhakemesi Kanunu"),
    (4721, 1, 5, "TMK", "Türk Medeni Kanunu"),
    (6098, 1, 5, "TBK", "Türk Borçlar Kanunu"),
    (6102, 1, 5, "TTK", "Türk Ticaret Kanunu"),
    (6100, 1, 5, "HMK", "Hukuk Muhakemeleri Kanunu"),
    (4857, 1, 5, "IsK", "İş Kanunu"),
    (6698, 1, 5, "KVKK", "Kişisel Verilerin Korunması Kanunu"),
    (213, 1, 4, "VUK", "Vergi Usul Kanunu"),
    (2004, 1, 3, "IIK", "İcra ve İflas Kanunu"),
    (2577, 1, 5, "IYUK", "İdari Yargılama Usulü Kanunu"),
    (6502, 1, 5, "TKHK", "Tüketicinin Korunması Hakkında Kanun"),
    (5510, 1, 5, "SSGSS", "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu"),
    (657, 1, 5, "DMK", "Devlet Memurları Kanunu"),
    (2918, 1, 5, "KTK", "Karayolları Trafik Kanunu"),
]


def get_priority_list():
    return PRIORITY_KANUNLAR
