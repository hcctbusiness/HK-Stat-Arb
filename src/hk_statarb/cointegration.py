"""Pair selection statistics: Engle-Granger cointegration, OLS hedge ratio, OU half-life, BH-FDR."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def ols_hedge(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS of y on [1, x] → (alpha, beta)."""
    X = np.column_stack([np.ones_like(x), x])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(a), float(b)


def half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life (in bars) from the AR(1) regression Δs_t = c + λ s_{t−1} + ε.

    The discrete AR coefficient is φ = 1 + λ, and the half-life is −ln 2 / ln φ.
    Returns +inf when the spread is not mean-reverting (φ ≥ 1).
    """
    s = np.asarray(spread, float)
    ds, lag = np.diff(s), s[:-1]
    _, lam = ols_hedge(ds, lag)
    phi = 1.0 + lam
    if phi >= 1.0 or phi <= 0.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))


def engle_granger(y: np.ndarray, x: np.ndarray) -> float:
    """Engle-Granger two-step cointegration test p-value (MacKinnon), y regressed on x with a constant."""
    return float(coint(y, x, trend="c", autolag="AIC")[1])


def benjamini_hochberg(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Boolean mask of hypotheses rejected by the Benjamini-Hochberg FDR procedure at level q."""
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    mask = np.zeros(m, bool)
    mask[order[:k]] = True
    return mask


@dataclass(frozen=True)
class SelectionParams:
    p_max: float = 0.05            # raw Engle-Granger p-value cut-off (after ×2 direction adjustment)
    hl_min: float = 2.0            # half-life bounds in trading days
    hl_max: float = 63.0
    beta_min: float = 0.5          # hedge-ratio bounds (log-price regression)
    beta_max: float = 2.0
    max_pairs: int = 20            # keep the top-N by p-value
    fdr_q: float = 0.10            # BH level for the multiple-testing-controlled variant


def test_pairs(logpx: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    """Run every same-sector pair test on formation-window log prices.

    Both regression directions are tested and the smaller p-value is kept, multiplied by 2
    (Bonferroni for the two directions) so that choosing the direction is not a free option.
    """
    rows = []
    names = [c for c in logpx.columns if logpx[c].notna().all()]
    by_sector: dict[str, list[str]] = {}
    for n in names:
        by_sector.setdefault(sectors[n], []).append(n)
    for sec, members in by_sector.items():
        for a, b in combinations(sorted(members), 2):
            ya, yb = logpx[a].to_numpy(), logpx[b].to_numpy()
            p_ab, p_ba = engle_granger(ya, yb), engle_granger(yb, ya)
            y, x = (a, b) if p_ab <= p_ba else (b, a)
            yv, xv = logpx[y].to_numpy(), logpx[x].to_numpy()
            alpha, beta = ols_hedge(yv, xv)
            resid = yv - alpha - beta * xv
            rows.append({"y": y, "x": x, "sector": sec, "p": min(1.0, 2 * min(p_ab, p_ba)),
                         "alpha": alpha, "beta": beta, "half_life": half_life(resid),
                         "resid_mean": float(resid.mean()), "resid_sd": float(resid.std(ddof=1))})
    return pd.DataFrame(rows)


def select_pairs(tests: pd.DataFrame, params: SelectionParams = SelectionParams(), fdr: bool = False) -> pd.DataFrame:
    """Apply filters. ``fdr=False``: p < p_max, top-N by p. ``fdr=True``: BH-FDR across *all* tests."""
    if tests.empty:
        return tests
    t = tests.copy()
    t["bh"] = benjamini_hochberg(t["p"].to_numpy(), params.fdr_q)
    ok = (t["half_life"].between(params.hl_min, params.hl_max)
          & t["beta"].between(params.beta_min, params.beta_max))
    ok &= t["bh"] if fdr else (t["p"] < params.p_max)
    return t[ok].sort_values("p").head(params.max_pairs).reset_index(drop=True)
