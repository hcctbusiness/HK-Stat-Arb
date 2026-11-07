"""Walk-forward pairs-trading engine (formation → trading windows) producing a drifted weight path.

Timeline per window (all parameters fixed up front):

* **Formation** (252 bars): test every same-sector pair on log total-return prices,
  estimate the OLS hedge ratio, residual mean/sd, OU half-life and leg-return volatility.
* **Trading** (next 126 bars): trade the selected pairs on the z-score state machine.
  Positions still open on the last bar of the window are closed. The next window's
  formation period ends where this trading period starts (re-formed every 126 bars).

Spread & z-score:

* ``static``: s_t = y_t − α − β x_t with formation OLS (α, β); z = (s_t − μ_form) / σ_form.
* ``kalman``: β_t, α_t from :func:`hk_statarb.kalman.kalman_hedge` started at the formation
  OLS estimate (V_e = formation residual variance, q_β = 1e-5, q_α = 1e-6, x centred on its
  formation mean) and run through formation then trading bars; z = e_t / √Q_t (prior
  forecast error over its predicted s.d.).

Sizing: each pair is dollar-neutral (+g/2 in one leg, −g/2 in the other; the hedge ratio
defines the *signal*, as in Gatev-Goetzmann-Rouwenhorst) with pair gross
g = min(1.0, 2·risk / σ_ann(r_y − r_x)) so every pair carries the same ex-ante risk
(``risk_per_pair`` annualised, formation volatility). New entries are scaled down so that
total gross never exceeds ``gross_cap``.

Weights are produced as the **drifted** book (legs drift with prices between trades), so
QuantCore's turnover — and therefore cost — is incurred only on entry/exit days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cointegration as ci
from .kalman import kalman_hedge
from .signals import ENTER, EXIT, STOP, TIME, zscore_positions


@dataclass(frozen=True)
class PairsParams:
    formation: int = 252
    trading: int = 126
    entry: float = 2.0
    exit_z: float = 0.0
    stop: float = 4.0
    time_stop_hl: float = 3.0       # time stop after 3 × formation half-life
    risk_per_pair: float = 0.05     # ex-ante annualised vol per pair
    pair_gross_cap: float = 1.0
    gross_cap: float = 3.0
    kalman_q: tuple[float, float] = (1e-5, 1e-6)   # (q_β, q_α) state noise
    delay: int = 0                  # extra bars between signal and trade (execution-delay check)
    selection: ci.SelectionParams = field(default_factory=ci.SelectionParams)


def windows(n: int, formation: int, trading: int) -> list[tuple[int, int, int]]:
    """(formation_start, trading_start, trading_end_exclusive) integer positions."""
    out, i = [], 0
    while i + formation < n:
        out.append((i, i + formation, min(i + formation + trading, n)))
        i += trading
    return out


def spread_z(logpx: pd.DataFrame, pair: pd.Series, f0: int, t0: int, t1: int, method: str, q: tuple[float, float] = (1e-5, 1e-6)):
    """z-score and hedge ratio on trading bars [t0, t1) for one selected pair."""
    y = logpx[pair["y"]].to_numpy()[f0:t1]
    x = logpx[pair["x"]].to_numpy()[f0:t1]
    k = t0 - f0
    if method == "static":
        s = y - pair["alpha"] - pair["beta"] * x
        z = (s - pair["resid_mean"]) / pair["resid_sd"]
        beta = np.full(len(y), pair["beta"])
    elif method == "kalman":
        xm = float(np.mean(x[:k]))                       # formation mean only
        kf = kalman_hedge(y, x - xm, q_beta=q[0], q_alpha=q[1], ve=pair["resid_sd"] ** 2,
                          theta0=(pair["beta"], pair["alpha"] + pair["beta"] * xm), p0=1e-4)
        z, beta = kf.z, kf.beta
    else:
        raise ValueError(method)
    return z[k:], beta[k:]


@dataclass
class PairsResult:
    weights: pd.DataFrame
    trades: pd.DataFrame
    windows: pd.DataFrame
    selections: dict          # trading_start -> selected pairs DataFrame
    tests: dict               # trading_start -> all tests DataFrame
    signals: dict             # (trading_start, y, x) -> DataFrame(z, beta, pos, event)


def form_all(px: pd.DataFrame, sectors: dict[str, str], params: PairsParams) -> tuple[dict, list]:
    """Run pair tests for every formation window (shared by static and Kalman engines)."""
    logpx = np.log(px)
    tests = {}
    wins = windows(len(px), params.formation, params.trading)
    for f0, t0, t1 in wins:
        form = logpx.iloc[f0:t0]
        live = [c for c in form.columns if form[c].notna().all() and pd.notna(px[c].iloc[t0 - 1])]
        tests[px.index[t0]] = ci.test_pairs(form[live], sectors)
    return tests, wins


def run_pairs(px: pd.DataFrame, sectors: dict[str, str], params: PairsParams = PairsParams(),
              method: str = "static", fdr: bool = False, tests: dict | None = None) -> PairsResult:
    px = px.sort_index()
    logpx = np.log(px)
    rets = px.pct_change(fill_method=None).fillna(0.0)
    if tests is None:
        tests, wins = form_all(px, sectors, params)
    else:
        wins = windows(len(px), params.formation, params.trading)
    cols = list(px.columns)
    col_ix = {c: i for i, c in enumerate(cols)}
    R = rets.to_numpy()
    n = len(px)

    # ---------- signals per window ----------
    plan = {}          # bar -> list of (pid, pos, event)
    last_bar = {}      # window id -> last trading bar
    selections, signals, wrows = {}, {}, []
    pair_info = {}
    for wid, (f0, t0, t1) in enumerate(wins):
        d0 = px.index[t0]
        tst = tests[d0]
        sel = ci.select_pairs(tst, params.selection, fdr=fdr)
        selections[d0] = sel
        wrows.append({"trading_start": d0, "trading_end": px.index[t1 - 1], "tested": len(tst),
                      "raw_p<0.05": int((tst["p"] < 0.05).sum()) if len(tst) else 0,
                      "bh_q10": int(ci.benjamini_hochberg(tst["p"].to_numpy(), params.selection.fdr_q).sum()) if len(tst) else 0,
                      "selected": len(sel)})
        last_bar[wid] = t1 - 1
        for _, pr in sel.iterrows():
            z, beta = spread_z(logpx, pr, f0, t0, t1, method, params.kalman_q)
            max_hold = int(np.ceil(params.time_stop_hl * pr["half_life"]))
            pos, ev = zscore_positions(z, params.entry, params.exit_z, params.stop, max_hold)
            pid = (wid, pr["y"], pr["x"])
            ry = rets[pr["y"]].to_numpy()[f0 + 1:t0]
            rx = rets[pr["x"]].to_numpy()[f0 + 1:t0]
            vol = np.std(ry - rx, ddof=1) * np.sqrt(252)
            g = min(params.pair_gross_cap, 2 * params.risk_per_pair / vol) if vol > 0 else 0.0
            pair_info[pid] = {"g": g, "sector": pr["sector"], "half_life": pr["half_life"], "p": pr["p"]}
            idx = px.index[t0:t1]
            signals[(d0, pr["y"], pr["x"])] = pd.DataFrame({"z": z, "beta": beta, "pos": pos, "event": ev}, index=idx)
            for j in range(t1 - t0):
                b = min(t0 + j + params.delay, t1 - 1)
                plan.setdefault(b, []).append((pid, int(pos[j]), int(ev[j])))

    # ---------- drifted book ----------
    W = np.zeros((n, len(cols)))
    active: dict = {}   # pid -> dict(iy, ix, wy, wx, entry, side, pnl, wid)
    trades = []
    ends = set(last_bar.values())
    reason_name = {EXIT: "mean reversion", STOP: "stop-loss", TIME: "time stop"}
    for t in range(n):
        if active:
            r = R[t]
            port = sum(a["wy"] * r[a["iy"]] + a["wx"] * r[a["ix"]] for a in active.values())
            for a in active.values():
                a["pnl"] += a["wy"] * r[a["iy"]] + a["wx"] * r[a["ix"]]
                a["wy"] *= (1 + r[a["iy"]]) / (1 + port)
                a["wx"] *= (1 + r[a["ix"]]) / (1 + port)
        for pid, p, ev in plan.get(t, []):
            if pid in active and (p == 0 or ev in (EXIT, STOP, TIME)):
                a = active.pop(pid)
                trades.append({"window": px.index[wins[pid[0]][1]], "y": pid[1], "x": pid[2], "sector": pair_info[pid]["sector"],
                               "side": a["side"], "entry": px.index[a["entry"]], "exit": px.index[t], "days": t - a["entry"],
                               "reason": reason_name.get(ev, "data gap"), "pnl": a["pnl"]})
            elif ev == ENTER and pid not in active and t != last_bar[pid[0]]:
                gross_now = sum(abs(a["wy"]) + abs(a["wx"]) for a in active.values())
                g = min(pair_info[pid]["g"], max(0.0, params.gross_cap - gross_now))
                if g >= 0.01:
                    active[pid] = {"iy": col_ix[pid[1]], "ix": col_ix[pid[2]], "wy": p * g / 2, "wx": -p * g / 2,
                                   "entry": t, "side": p, "pnl": 0.0}
        if t in ends:   # force-close everything belonging to a window that ends today
            for pid in [k for k in active if last_bar[k[0]] == t]:
                a = active.pop(pid)
                trades.append({"window": px.index[wins[pid[0]][1]], "y": pid[1], "x": pid[2], "sector": pair_info[pid]["sector"],
                               "side": a["side"], "entry": px.index[a["entry"]], "exit": px.index[t], "days": t - a["entry"],
                               "reason": "window end", "pnl": a["pnl"]})
        for a in active.values():
            W[t, a["iy"]] += a["wy"]
            W[t, a["ix"]] += a["wx"]
    weights = pd.DataFrame(W, index=px.index, columns=cols)
    return PairsResult(weights, pd.DataFrame(trades), pd.DataFrame(wrows), selections, tests, signals)


def dollar_neutral_targets(pos: dict[tuple[str, str], int], gross: dict[tuple[str, str], float]) -> pd.Series:
    """Target weights for a set of pair positions (+g/2 / −g/2 per pair) — sums to zero."""
    w: dict[str, float] = {}
    for (y, x), p in pos.items():
        g = gross[(y, x)]
        w[y] = w.get(y, 0.0) + p * g / 2
        w[x] = w.get(x, 0.0) - p * g / 2
    return pd.Series(w, dtype=float)
