"""Hong Kong trading costs on top of QuantCore's linear cost model.

QuantCore's :class:`~quantcore.CostModel` charges the same commission + slippage on every
asset plus a borrow fee on gross shorts. HK stamp duty, however, applies only to stock
trades: ETFs such as the Tracker Fund (2800.HK) have been exempt since 2015. This module
therefore runs ``qc.backtest`` with commission, slippage and the default borrow fee, then
deducts stamp duty (and any asset-specific extra borrow) per asset, using exactly the same
drifted-book turnover definition as the engine.

Stamp duty: 0.13% per side from Aug-2021 to Nov-2023, 0.10% before and after. We simplify
to **0.10% per side throughout** (slightly flatters 2021-23).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

import quantcore as qc


@dataclass(frozen=True)
class HKCosts:
    commission_bps: float = 3.0       # broker commission incl. exchange/SFC/FRC levies (~1 bp)
    slippage_bps: float = 5.0
    stamp_bps: float = 10.0           # per side, HK stocks only
    borrow_bps_annual: float = 150.0  # HK short-borrow fee on gross shorts
    stamp_exempt: tuple[str, ...] = ("2800.HK",)
    stamp_override: dict = field(default_factory=dict)   # e.g. A-share legs {ticker: bps}
    borrow_override: dict = field(default_factory=dict)  # {ticker: bps per year}

    def scaled(self, k: float) -> "HKCosts":
        return replace(self, commission_bps=self.commission_bps * k, slippage_bps=self.slippage_bps * k,
                       stamp_bps=self.stamp_bps * k, borrow_bps_annual=self.borrow_bps_annual * k,
                       stamp_override={a: v * k for a, v in self.stamp_override.items()},
                       borrow_override={a: v * k for a, v in self.borrow_override.items()})

    @property
    def model(self) -> qc.CostModel:
        return qc.CostModel(self.commission_bps, self.slippage_bps, self.borrow_bps_annual)


def asset_turnover(held: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per-asset |trade| against the drifted book — identical to QuantCore's definition."""
    rets = prices[held.columns].reindex(held.index).pct_change(fill_method=None).fillna(0.0)
    prev = held.shift(1).fillna(0.0)
    r_prev = rets.shift(1).fillna(0.0)
    denom = (1 + (prev * r_prev).sum(axis=1)).replace(0, np.nan)
    drifted = (prev * (1 + r_prev)).div(denom, axis=0).fillna(0.0)
    return (held - drifted).abs()


def run(weights: pd.DataFrame, prices: pd.DataFrame, costs: HKCosts, name: str) -> qc.BacktestResult:
    """``qc.backtest`` (lag=1) + per-asset stamp duty and borrow overrides."""
    res = qc.backtest(weights, prices, costs.model, lag=1, name=name)
    held = res.weights
    trade = asset_turnover(held, prices)
    stamp = pd.Series({a: (0.0 if a in costs.stamp_exempt else costs.stamp_override.get(a, costs.stamp_bps)) for a in held.columns}) / 1e4
    extra_borrow = pd.Series({a: costs.borrow_override.get(a, costs.borrow_bps_annual) - costs.borrow_bps_annual for a in held.columns}) / 1e4 / 252
    extra = trade.mul(stamp, axis=1).sum(axis=1) + held.clip(upper=0).abs().mul(extra_borrow, axis=1).sum(axis=1)
    out = qc.BacktestResult(res.returns - extra, res.gross_returns, held, res.turnover, res.costs + extra, name=name)
    out.meta["stamp"] = trade.mul(stamp, axis=1).sum(axis=1)
    return out


def trim(res: qc.BacktestResult, start, end=None, name: str | None = None) -> qc.BacktestResult:
    sl = slice(start, end)
    out = qc.BacktestResult(res.returns[sl], res.gross_returns[sl], res.weights[sl], res.turnover[sl], res.costs[sl], name=name or res.name)
    out.meta = {k: v[sl] for k, v in res.meta.items()}
    return out
