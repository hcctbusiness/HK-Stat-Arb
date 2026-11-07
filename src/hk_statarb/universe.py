"""Hard-coded universes (fixed up front, not selected on performance).

Both lists are today's constituents/listings picked in hindsight, so the study carries
survivorship bias (see README › Limitations).
"""
from __future__ import annotations

# ---------------------------------------------------------------- HK pairs universe
SECTORS: dict[str, list[str]] = {
    "Banks & Insurers": [
        "0005.HK", "2388.HK", "0939.HK", "1398.HK", "3988.HK", "3968.HK", "1288.HK",
        "3328.HK", "0998.HK", "2318.HK", "2628.HK", "2601.HK", "1299.HK", "1339.HK",
    ],
    "Property": [
        "0016.HK", "0012.HK", "0017.HK", "0101.HK", "0083.HK", "0688.HK", "1109.HK",
        "0823.HK", "0004.HK", "0019.HK", "0001.HK",
    ],
    "Telecom & Tech": [
        "0700.HK", "0941.HK", "0762.HK", "0728.HK", "0992.HK", "2382.HK", "0285.HK", "0981.HK",
    ],
    "Utilities": [
        "0002.HK", "0003.HK", "0006.HK", "1038.HK", "0836.HK", "0384.HK", "2688.HK", "0270.HK", "0902.HK",
    ],
    "Energy & Materials": [
        "0857.HK", "0386.HK", "0883.HK", "1088.HK", "1171.HK", "2899.HK", "0358.HK", "3323.HK", "0914.HK",
    ],
    "Consumer": [
        "0027.HK", "1928.HK", "0291.HK", "0322.HK", "0151.HK", "2020.HK", "1044.HK", "0066.HK",
        "2319.HK", "0868.HK", "0220.HK",
    ],
}
TICKERS: list[str] = [t for v in SECTORS.values() for t in v]
SECTOR_OF: dict[str, str] = {t: s for s, v in SECTORS.items() for t in v}

BENCH = "2800.HK"   # Tracker Fund of Hong Kong (HSI ETF; exempt from HK stamp duty)
INDEX = "^HSI"
FX = "CNYHKD=X"     # HKD per 1 CNY

# ---------------------------------------------------------------- A/H dual listings
# name: (H-share ticker, A-share ticker)
AH_PAIRS: dict[str, tuple[str, str]] = {
    "CCB": ("0939.HK", "601939.SS"),
    "ICBC": ("1398.HK", "601398.SS"),
    "Bank of China": ("3988.HK", "601988.SS"),
    "ABC": ("1288.HK", "601288.SS"),
    "BoCom": ("3328.HK", "601328.SS"),
    "China Merchants Bank": ("3968.HK", "600036.SS"),
    "China CITIC Bank": ("0998.HK", "601998.SS"),
    "Minsheng Bank": ("1988.HK", "600016.SS"),
    "Ping An": ("2318.HK", "601318.SS"),
    "China Life": ("2628.HK", "601628.SS"),
    "CPIC": ("2601.HK", "601601.SS"),
    "CITIC Securities": ("6030.HK", "600030.SS"),
    "China Shenhua": ("1088.HK", "601088.SS"),
    "Sinopec": ("0386.HK", "600028.SS"),
    "PetroChina": ("0857.HK", "601857.SS"),
    "Anhui Conch": ("0914.HK", "600585.SS"),
    "BYD": ("1211.HK", "002594.SZ"),
    "Zijin Mining": ("2899.HK", "601899.SS"),
    "Jiangxi Copper": ("0358.HK", "600362.SS"),
    "China Railway": ("0390.HK", "601390.SS"),
    "CRCC": ("1186.HK", "601186.SS"),
    "Great Wall Motor": ("2333.HK", "601633.SS"),
    "Tsingtao Brewery": ("0168.HK", "600600.SS"),
    "China Coal": ("1898.HK", "601898.SS"),
    "ZTE": ("0763.HK", "000063.SZ"),
}
