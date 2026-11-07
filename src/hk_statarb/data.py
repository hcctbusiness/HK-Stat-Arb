"""Data loading and calendar alignment.

* Total-return (dividend-adjusted) closes come from ``quantcore.data.prices`` and are used
  for **returns** and for pair formation.
* The A/H premium is a **price-level** comparison, so it must use *unadjusted* closes:
  Yahoo's dividend back-adjustment scales history by each listing's own dividend yield,
  which differs between the A and H lines and would distort historical premium levels.
  ``raw_closes`` therefore downloads split-adjusted but not dividend-adjusted closes.
"""
from __future__ import annotations

import pandas as pd

import quantcore as qc


def total_return_prices(tickers: list[str], start: str = "2011-01-01") -> pd.DataFrame:
    return qc.data.prices(tickers, start=start)


def raw_closes(tickers: list[str], start: str = "2011-01-01", refresh: bool = False) -> pd.DataFrame:
    """Unadjusted (split-adjusted only) closes from Yahoo Finance, cached alongside QuantCore's cache."""
    key = "raw_" + "_".join(sorted(t.replace("^", "").replace("=", "") for t in tickers)) + f"_{start}"
    path = qc.data._cache_path(key[:150])
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False, threads=True)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    px = px[[t for t in tickers if t in px.columns]].dropna(how="all")
    px.to_csv(path)
    return px


def to_calendar(series: pd.DataFrame | pd.Series, calendar: pd.DatetimeIndex, limit: int = 5) -> pd.DataFrame | pd.Series:
    """Map a series onto another market's trading calendar using only past-or-same-day values.

    Each calendar date receives the latest observation dated **on or before** it (as-of
    join), carried forward for at most ``limit`` calendar rows. Shanghai/Shenzhen close at
    15:00 HKT, before Hong Kong's 16:00 close, so a same-day A-share close is known when the
    H-share signal is formed. Beyond ``limit`` rows (long mainland holidays) the value is NaN.
    """
    union = series.index.union(calendar)
    return series.reindex(union).ffill(limit=limit).reindex(calendar)


def lagged_fx(fx: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """CNY→HKD rate known at the HK close of each date.

    Yahoo FX daily bars are stamped with the trade date but settle after the HK close,
    so the **previous** FX bar is used (strictly before t). CNYHKD moves ~0.2% a day at
    most, so this costs little accuracy and removes a subtle look-ahead.
    """
    return to_calendar(fx.dropna().shift(1), calendar, limit=10)
