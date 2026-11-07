"""A/H-share premium: construction, signal and portfolios.

Premium_i,t = (A-share close in CNY × CNYHKD_{t−1}) / H-share close in HKD − 1,
computed on the Hong Kong trading calendar from **unadjusted** closes. A-share closes are
as-of joined (latest close on or before t, max 5 HK bars stale), FX uses the previous FX
bar (see :mod:`hk_statarb.data`).

Implementable strategy (H-leg only): foreign investors cannot easily short A-shares
(Northbound Stock Connect does not allow naked shorting and securities lending is very
limited), so we trade only Hong Kong lines. At each month end, each name's premium is
z-scored against its own trailing 252-day history; the ``n_long`` names with the highest
z (H unusually cheap relative to A *by its own standards*) are bought equal-weight, and
the basket's market beta is hedged with a short 2800.HK position sized by trailing
252-day betas. Weights then drift until the next month end.

Theoretical two-leg reference: same names, long H / short A (in HKD) dollar-for-dollar.
**Not implementable** for most investors; shown only to isolate the convergence effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import lagged_fx, to_calendar


def premium(h_raw: pd.DataFrame, a_raw: pd.DataFrame, fx: pd.Series, pairs: dict[str, tuple[str, str]],
            calendar: pd.DatetimeIndex, a_limit: int = 5) -> pd.DataFrame:
    """A/H premium per name on ``calendar`` (decimal, e.g. 0.30 = A trades 30% above H)."""
    fxc = lagged_fx(fx, calendar)
    out = {}
    for name, (h, a) in pairs.items():
        if h not in h_raw or a not in a_raw:
            continue
        hp = h_raw[h].reindex(calendar)
        ap = to_calendar(a_raw[a].dropna(), calendar, limit=a_limit)
        out[name] = ap * fxc / hp - 1.0
    return pd.DataFrame(out, index=calendar)


def ah_index(prem: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional median / quartiles of the premium (an equal-weight analogue of the HSAHP index)."""
    return pd.DataFrame({"median": prem.median(axis=1), "q25": prem.quantile(0.25, axis=1),
                         "q75": prem.quantile(0.75, axis=1), "n": prem.notna().sum(axis=1)})


def premium_z(prem: pd.DataFrame, window: int = 252, min_periods: int = 126) -> pd.DataFrame:
    """Time-series z-score of each name's premium against its own trailing history (inclusive of t)."""
    m = prem.rolling(window, min_periods=min_periods).mean()
    s = prem.rolling(window, min_periods=min_periods).std()
    return (prem - m) / s


def rolling_beta(rets: pd.DataFrame, bench: pd.Series, window: int = 252) -> pd.DataFrame:
    cov = rets.rolling(window, min_periods=window // 2).cov(bench)
    var = bench.rolling(window, min_periods=window // 2).var()
    return cov.div(var, axis=0)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def h_only_targets(z: pd.DataFrame, h_of: dict[str, str], betas: pd.DataFrame, n_long: int = 5,
                   hedge: str = "2800.HK", hedged: bool = True) -> pd.DataFrame:
    """Month-end targets: long top-``n_long`` H shares by premium z, short beta-matched ``hedge``."""
    rows = {}
    for d in month_ends(z.index):
        zz = z.loc[d].dropna()
        if len(zz) < 2 * n_long:
            continue
        top = zz.nlargest(n_long).index
        w = pd.Series(1.0 / n_long, index=[h_of[k] for k in top])
        if hedged:
            b = betas.loc[d, w.index].fillna(1.0).clip(0.0, 2.5)
            w[hedge] = -float((w * b).sum())
        rows[d] = w
    return pd.DataFrame(rows).T.fillna(0.0)


def two_leg_targets(z: pd.DataFrame, pairs: dict[str, tuple[str, str]], a_hkd_cols: dict[str, str], n_long: int = 5) -> pd.DataFrame:
    """Reference only: long H / short A(HKD) on the same top-``n_long`` names (gross 2)."""
    rows = {}
    for d in month_ends(z.index):
        zz = z.loc[d].dropna()
        if len(zz) < 2 * n_long:
            continue
        w = {}
        for k in zz.nlargest(n_long).index:
            w[pairs[k][0]] = 1.0 / n_long
            w[a_hkd_cols[k]] = -1.0 / n_long
        rows[d] = pd.Series(w)
    return pd.DataFrame(rows).T.fillna(0.0)


def drift_targets(targets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Turn rebalance-date targets into the drifted weight path QuantCore should trade.

    On a rebalance date d the post-trade book equals the target; on other days the
    previous book drifts with that day's returns (same algebra as the engine), so turnover
    occurs only on rebalance dates. Output is indexed like ``prices`` (weights at close t).
    """
    cols = list(targets.columns)
    rets = prices[cols].pct_change(fill_method=None).fillna(0.0).to_numpy()
    tgt = targets.reindex(prices.index)
    is_rb = tgt.notna().any(axis=1).to_numpy()
    T = tgt.fillna(0.0).to_numpy()
    W = np.zeros((len(prices), len(cols)))
    for t in range(len(prices)):
        if is_rb[t]:
            W[t] = T[t]
        elif t > 0:
            h, r = W[t - 1], rets[t]
            W[t] = h * (1 + r) / (1 + h @ r)
    return pd.DataFrame(W, index=prices.index, columns=cols)


def control_targets(z: pd.DataFrame, h_of: dict[str, str], betas: pd.DataFrame, hedge: str = "2800.HK") -> pd.DataFrame:
    """Control portfolio: every A/H name with a valid signal, equal-weight, same beta hedge.

    Separates the premium *signal* from the *universe* (dual-listed SOE-heavy H shares vs the HSI).
    """
    rows = {}
    for d in month_ends(z.index):
        names = z.loc[d].dropna().index
        if len(names) < 10:
            continue
        w = pd.Series(1.0 / len(names), index=[h_of[k] for k in names])
        b = betas.loc[d, w.index].fillna(1.0).clip(0.0, 2.5)
        w[hedge] = -float((w * b).sum())
        rows[d] = w
    return pd.DataFrame(rows).T.fillna(0.0)
