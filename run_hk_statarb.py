"""End-to-end Hong Kong statistical-arbitrage study on real data (Yahoo Finance).

Run:  python run_hk_statarb.py
Writes docs/*.png and docs/results.md. Every number in the README comes from here.
All strategy parameters are fixed in the module defaults (PairsParams, SelectionParams,
HKCosts, A/H constants below); nothing is optimised on the sample.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import quantcore as qc
from hk_statarb import ah, data, plots
from hk_statarb import costs as hc
from hk_statarb import pairs as pr
from hk_statarb.signals import ENTER
from hk_statarb.universe import AH_PAIRS, BENCH, FX, INDEX, SECTOR_OF, SECTORS, TICKERS

DOCS = Path(__file__).resolve().parent / "docs"
START = "2011-01-01"          # first 252 bars are formation / warm-up only
AH_N_LONG = 5                 # names held in the A/H strategy
AH_Z_WINDOW = 252             # premium z-score look-back (bars)
COSTS = hc.HKCosts()          # 3 bp commission, 5 bp slippage, 10 bp stamp (stocks), 150 bp/yr borrow
# Two-leg reference only: A-share stamp 0.05% (sells only since Aug-2023; charged per side here)
# and an 8%/yr A-share borrow fee (securities lending is scarce and expensive on the mainland).
A_STAMP_BPS, A_BORROW_BPS = 5.0, 800.0
ROWS = ["CAGR", "Ann. Vol", "Sharpe", "Max Drawdown", "Calmar", "Hit Rate", "Avg Daily Turnover", "Annual Cost Drag"]


def pct(x, d=2):
    return "—" if pd.isna(x) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame, fmt=lambda c, v: str(v)) -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = ["| " + " | ".join([str(i)] + [fmt(c, v) for c, v in r.items()]) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    h_tk = [h for h, _ in AH_PAIRS.values()]
    a_tk = [a for _, a in AH_PAIRS.values()]
    every = list(dict.fromkeys(TICKERS + [BENCH, INDEX, FX] + h_tk + a_tk))
    px_all = data.total_return_prices(every, start=START)
    # Hong Kong trading calendar = days on which most of the HK universe printed a close
    cal = px_all.index[px_all[TICKERS].notna().sum(axis=1) > len(TICKERS) / 2]
    print(f"HK calendar: {len(cal)} bars, {cal[0].date()} -> {cal[-1].date()}")

    # ====================================================== 1. Pairs trading
    hk = px_all.loc[cal, TICKERS].ffill(limit=5)
    params = pr.PairsParams()
    tests, wins = pr.form_all(hk, SECTOR_OF, params)
    runs = {
        "Pairs Static": pr.run_pairs(hk, SECTOR_OF, params, "static", tests=tests),
        "Pairs Kalman": pr.run_pairs(hk, SECTOR_OF, params, "kalman", tests=tests),
    }
    robust = {
        "Static, BH-FDR only": pr.run_pairs(hk, SECTOR_OF, params, "static", fdr=True, tests=tests),
        "Kalman, BH-FDR only": pr.run_pairs(hk, SECTOR_OF, params, "kalman", fdr=True, tests=tests),
        "Static, +1d delay": pr.run_pairs(hk, SECTOR_OF, replace(params, delay=1), "static", tests=tests),
        "Kalman, +1d delay": pr.run_pairs(hk, SECTOR_OF, replace(params, delay=1), "kalman", tests=tests),
    }
    t_start = runs["Pairs Static"].windows["trading_start"].iloc[0]
    end = cal[-1]
    print(f"Pairs trading period {t_start.date()} -> {end.date()}, {len(wins)} windows")

    def bt_pairs(res, name, k=1.0):
        return hc.trim(hc.run(res.weights, hk, COSTS.scaled(k), name), t_start)

    # ====================================================== 2. A/H premium
    raw = data.raw_closes(list(dict.fromkeys(h_tk + a_tk + [FX])), start=START)
    prem = ah.premium(raw, raw, raw[FX], AH_PAIRS, cal)
    ahx = ah.ah_index(prem)
    z = ah.premium_z(prem, AH_Z_WINDOW)
    fxc = data.lagged_fx(raw[FX], cal)
    h_of = {k: v[0] for k, v in AH_PAIRS.items()}
    a_cols = {k: f"{v[1]}@HKD" for k, v in AH_PAIRS.items()}
    ah_px = px_all.loc[cal, list(dict.fromkeys(h_tk + [BENCH]))].ffill(limit=5)
    for k, (h, a) in AH_PAIRS.items():
        ah_px[a_cols[k]] = data.to_calendar(px_all[a].dropna(), cal, limit=5) * fxc   # A total return in HKD
    rets_h = ah_px[h_tk].pct_change(fill_method=None)
    betas = ah.rolling_beta(rets_h, ah_px[BENCH].pct_change(fill_method=None))
    ah_targets = {
        "A/H H-only (hedged)": ah.h_only_targets(z, h_of, betas, AH_N_LONG, BENCH, hedged=True),
        "A/H H-only (unhedged)": ah.h_only_targets(z, h_of, betas, AH_N_LONG, BENCH, hedged=False),
        "A/H two-leg (reference)": ah.two_leg_targets(z, AH_PAIRS, a_cols, AH_N_LONG),
        "A/H control: all names EW (hedged)": ah.control_targets(z, h_of, betas, BENCH),
    }
    # execution-delay check: rebalance on the bar after the signal month-end
    nxt = lambda tg: tg.set_axis(cal[np.minimum(cal.get_indexer(tg.index) + 1, len(cal) - 1)])
    ah_targets["A/H H-only (hedged), +1d delay"] = nxt(ah_targets["A/H H-only (hedged)"])
    ah_costs = replace(COSTS, stamp_override={c: A_STAMP_BPS for c in a_cols.values()},
                       borrow_override={c: A_BORROW_BPS for c in a_cols.values()})

    def bt_ah(name, k=1.0):
        tg = ah_targets[name]
        w = ah.drift_targets(tg, ah_px[tg.columns])
        return hc.trim(hc.run(w, ah_px[tg.columns], ah_costs.scaled(k), name), t_start)

    # ====================================================== 3. Backtests
    res = {n: bt_pairs(r, n) for n, r in runs.items()}
    res |= {n: bt_ah(n) for n in ["A/H H-only (hedged)", "A/H control: all names EW (hedged)", "A/H two-leg (reference)"]}
    rob = {n: bt_pairs(r, n) for n, r in robust.items()}
    rob |= {n: bt_ah(n) for n in ["A/H H-only (unhedged)", "A/H H-only (hedged), +1d delay"]}
    bench_r = px_all.loc[cal, BENCH].ffill(limit=5).pct_change(fill_method=None).fillna(0.0).loc[t_start:]
    hsi_r = px_all.loc[cal, INDEX].ffill(limit=5).pct_change(fill_method=None).fillna(0.0).loc[t_start:]
    b2800 = qc.BacktestResult(bench_r, bench_r, pd.DataFrame(), pd.Series(0.0, bench_r.index), pd.Series(0.0, bench_r.index), name="2800.HK (Tracker Fund)")
    hsi_bt = qc.BacktestResult(hsi_r, hsi_r, pd.DataFrame(), pd.Series(0.0, hsi_r.index), pd.Series(0.0, hsi_r.index), name="HSI (price index)")

    def stats_table(results):
        t = qc.compare(list(results)).loc[ROWS]
        t.loc["Sharpe (gross)"] = [qc.metrics.sharpe(r.gross_returns) for r in results]
        t.loc["CAGR (gross)"] = [qc.metrics.cagr(r.gross_returns) for r in results]
        t.loc["Stamp duty drag"] = [r.meta["stamp"].mean() * 252 if "stamp" in r.meta else 0.0 for r in results]
        bb = [qc.metrics.beta_alpha(r.returns, hsi_r) for r in results]
        t.loc["Beta vs HSI"] = [b for b, _ in bb]
        t.loc["Correlation vs HSI"] = [corr(r.returns, hsi_r) for r in results]
        return t
    qc.report._PCT.update({"CAGR (gross)", "Stamp duty drag"})
    main_tab = stats_table([*res.values(), b2800, hsi_bt])
    rob_tab = stats_table(list(rob.values()))
    print(main_tab.to_string())

    # cost sensitivity
    sens = {}
    for k in (0.0, 0.5, 1.0, 2.0):
        row = {}
        for n, r in runs.items():
            b = bt_pairs(r, n, k); row[(n, "Sharpe")] = qc.metrics.sharpe(b.returns); row[(n, "CAGR")] = qc.metrics.cagr(b.returns)
        for n in ["A/H H-only (hedged)", "A/H two-leg (reference)"]:
            b = bt_ah(n, k); row[(n, "Sharpe")] = qc.metrics.sharpe(b.returns); row[(n, "CAGR")] = qc.metrics.cagr(b.returns)
        sens[f"{k:g}× costs"] = row
    sens = pd.DataFrame(sens).T
    sens_md = pd.DataFrame({f"{n} {m}": sens[(n, m)] for n, m in sens.columns})
    sens_md.index.name = "Cost scenario"
    # break-even: per-side cost multiple at which net Sharpe = 0 (linear in k for fixed gross)
    breakeven = {}
    for n, r in [*runs.items()]:
        b0, b1 = bt_pairs(r, n, 0.0), bt_pairs(r, n, 1.0)
        g, c = b0.returns.mean(), (b0.returns - b1.returns).mean()
        breakeven[n] = g / c if c > 0 else np.nan
    for n in ["A/H H-only (hedged)", "A/H two-leg (reference)"]:
        b0, b1 = bt_ah(n, 0.0), bt_ah(n, 1.0)
        g, c = b0.returns.mean(), (b0.returns - b1.returns).mean()
        breakeven[n] = g / c if c > 0 else np.nan

    # ====================================================== 4. Diagnostics
    wtab = runs["Pairs Static"].windows.copy()
    wtab["expected_false_pos"] = 0.05 * wtab["tested"]
    sel_all = pd.concat([s.assign(window=d) for d, s in runs["Pairs Static"].selections.items()], ignore_index=True)
    tests_all = pd.concat(tests.values(), ignore_index=True)
    form_summary = pd.Series({
        "Formation windows": len(wtab),
        "Same-sector pairs tested per window (avg)": wtab["tested"].mean(),
        "Raw p < 0.05 per window (avg)": wtab["raw_p<0.05"].mean(),
        "Expected by chance at 5% (avg)": wtab["expected_false_pos"].mean(),
        "Significant after BH-FDR q=0.10 (avg)": wtab["bh_q10"].mean(),
        "Windows with zero BH-significant pairs": int((wtab["bh_q10"] == 0).sum()),
        "Pairs selected per window (avg)": wtab["selected"].mean(),
        "Selected half-life, median (days)": sel_all["half_life"].median(),
        "Selected half-life, IQR (days)": f"{sel_all['half_life'].quantile(.25):.1f}–{sel_all['half_life'].quantile(.75):.1f}",
        "Selected pairs by sector": ", ".join(f"{k} {v}" for k, v in sel_all["sector"].value_counts().items()),
    })
    # does formation cointegration persist? share of selected pairs still cointegrated (p<0.05) in the next window
    persist = []
    for d, s in runs["Pairs Static"].selections.items():
        nxt_d = [k for k in tests if k > d]
        if not nxt_d or s.empty:
            continue
        nt = tests[nxt_d[0]].set_index(["y", "x"])["p"]
        nt2 = tests[nxt_d[0]].set_index(["x", "y"])["p"]
        for _, row in s.iterrows():
            key = (row["y"], row["x"])
            p = nt.get(key, nt2.get(key, np.nan))
            if pd.notna(p):
                persist.append(p < 0.05)
    form_summary["Selected pairs still p<0.05 in next window"] = f"{np.mean(persist)*100:.0f}% (n={len(persist)})"

    def trade_stats(tr: pd.DataFrame) -> pd.Series:
        rc = tr["reason"].value_counts(normalize=True)
        return pd.Series({"Round trips": len(tr), "Avg holding (days)": tr["days"].mean(),
                          "Win rate (gross)": (tr["pnl"] > 0).mean(),
                          "Exit: mean reversion": rc.get("mean reversion", 0.0), "Exit: stop-loss": rc.get("stop-loss", 0.0),
                          "Exit: time stop": rc.get("time stop", 0.0), "Exit: window end / data gap": rc.get("window end", 0.0) + rc.get("data gap", 0.0),
                          "Trades per year": len(tr) / ((end - t_start).days / 365.25)})
    tstats = pd.DataFrame({n: trade_stats(r.trades) for n, r in runs.items()})
    tstats = tstats.astype(object)
    for i in tstats.index:
        tstats.loc[i] = [f"{int(v)}" if i == "Round trips" else (num(v, 1) if i in ("Avg holding (days)", "Trades per year") else pct(v, 1)) for v in tstats.loc[i]]
    tstats.index.name = "Trade statistics (gross P&L per trade)"

    # ====================================================== 5. Charts
    plots.formation_chart(wtab, sel_all["half_life"], tests_all["half_life"], DOCS / "pairs_formation.png")
    # example pair: the selected pair with the most entries in the static run (ties -> lowest p);
    # chosen by activity, not by profit.
    best, best_n = None, -1
    for (d, y, x), sig in runs["Pairs Static"].signals.items():
        n_ent = int((sig["event"] == ENTER).sum())
        p = runs["Pairs Static"].selections[d].set_index(["y", "x"]).loc[(y, x), "p"]
        if n_ent > best_n or (n_ent == best_n and p < best[3]):
            best, best_n = (d, y, x, p), n_ent
    d, y, x, _ = best
    sel_row = runs["Pairs Static"].selections[d].set_index(["y", "x"]).loc[(y, x)]
    f_start = cal[cal.get_loc(d) - params.formation]
    plots.example_pair_chart(np.log(hk), y, x, f_start, runs["Pairs Static"].signals[(d, y, x)],
                             runs["Pairs Kalman"].signals[(d, y, x)], sel_row["beta"], {v[0]: k for k, v in AH_PAIRS.items()}, DOCS / "example_pair.png")
    ex_trades = runs["Pairs Static"].trades.query("y == @y and x == @x and window == @d")
    plots.equity_chart({
        "Pairs Kalman — net": res["Pairs Kalman"].returns, "Pairs Kalman — gross": res["Pairs Kalman"].gross_returns,
        "Pairs Static — net": res["Pairs Static"].returns, "Pairs Static — gross": res["Pairs Static"].gross_returns,
    }, DOCS / "static_vs_kalman.png", "Static OLS vs Kalman hedge ratio — gross vs net of HK costs",
        styles={"Pairs Kalman — gross": {"color": "#0B3C5D", "ls": "--", "lw": 1.2}, "Pairs Static — gross": {"color": "#328CC1", "ls": "--", "lw": 1.2},
                "Pairs Kalman — net": {"color": "#0B3C5D"}, "Pairs Static — net": {"color": "#328CC1"}})
    plots.ah_premium_chart(ahx.loc[t_start - pd.Timedelta(days=365):], prem.ffill(limit=5).iloc[-1], DOCS / "ah_premium.png")
    qc.report.tearsheet({"Pairs Kalman (net)": res["Pairs Kalman"].returns, "Pairs Static (net)": res["Pairs Static"].returns,
                         "A/H H-only hedged (net)": res["A/H H-only (hedged)"].returns, "2800.HK": bench_r},
                        DOCS / "tearsheet.png", title="HCCT Group – HK stat-arb strategies (net of HK costs)")
    qc.report.tearsheet({"A/H H-only hedged (net)": res["A/H H-only (hedged)"].returns,
                         "A/H H-only hedged (gross)": res["A/H H-only (hedged)"].gross_returns,
                         "A/H two-leg — NOT implementable (net)": res["A/H two-leg (reference)"].returns,
                         "2800.HK": bench_r},
                        DOCS / "tearsheet_ah.png", title="HCCT Group – A/H premium mean reversion")
    plots.neutrality_chart({n: r.returns for n, r in res.items() if "control" not in n}, hsi_r, DOCS / "market_neutrality.png")

    # ====================================================== 6. results.md
    ah_now = prem.ffill(limit=5).iloc[-1]
    med = ahx["median"].loc[t_start:]
    ah_summary = pd.Series({
        "Names with data (latest)": int(ah_now.notna().sum()),
        "Median premium, latest": pct(med.iloc[-1], 1),
        "Median premium, average since start": pct(med.mean(), 1),
        "Median premium, min (date)": f"{pct(med.min(), 1)} ({med.idxmin():%Y-%m-%d})",
        "Median premium, max (date)": f"{pct(med.max(), 1)} ({med.idxmax():%Y-%m-%d})",
        "Widest premium, latest": f"{ah_now.idxmax()} {pct(ah_now.max(), 1)}",
        "Narrowest premium, latest": f"{ah_now.idxmin()} {pct(ah_now.min(), 1)}",
    })
    ex = (res["A/H H-only (hedged)"].returns - res["A/H control: all names EW (hedged)"].returns).dropna()
    yrs = len(ex) / 252
    ir = ex.mean() / ex.std() * np.sqrt(252)
    ex_net = f"{pct(ex.mean() * 252)} a year net (tracking error {pct(ex.std() * np.sqrt(252))}, IR {ir:.2f}, t ≈ {ir * np.sqrt(yrs):.2f} over {yrs:.1f} years)"
    exg = (res["A/H H-only (hedged)"].gross_returns - res["A/H control: all names EW (hedged)"].gross_returns).dropna()
    irg = exg.mean() / exg.std() * np.sqrt(252)
    ex_gross = f"{pct(exg.mean() * 252)} a year gross (IR {irg:.2f}, t ≈ {irg * np.sqrt(yrs):.2f})"
    ah_summary["Signal vs control (top-5 minus all-names EW, both hedged)"] = ex_net
    ah_summary["Signal vs control, gross"] = ex_gross
    held_names = ah_targets["A/H H-only (hedged)"].drop(columns=BENCH, errors="ignore")
    inv_h = {v[0]: k for k, v in AH_PAIRS.items()}
    hold_freq = (held_names.loc[t_start:] > 0).mean().sort_values(ascending=False).head(8)
    hold_freq.index = [f"{inv_h[i]} ({i})" for i in hold_freq.index]

    sel_tab = runs["Pairs Static"].windows.set_index("trading_start")[["trading_end", "tested", "raw_p<0.05", "bh_q10", "selected"]]
    sel_tab.index = [d.strftime("%Y-%m-%d") for d in sel_tab.index]; sel_tab.index.name = "Trading window start"
    sel_tab["trading_end"] = [d.strftime("%Y-%m-%d") for d in sel_tab["trading_end"]]
    sel_tab.columns = ["Trading window end", "Pairs tested", "Raw p < 0.05", "BH-FDR q=0.10", "Selected"]

    be_md = md_table(pd.DataFrame({"Break-even cost multiple": pd.Series(breakeven)}).rename_axis("Strategy"),
                     lambda c, v: "gross ≤ 0" if (pd.isna(v) or v <= 0) else f"{v:.2f}×")
    ex_md = md_table(ex_trades[["side", "entry", "exit", "days", "reason", "pnl"]].assign(
        side=lambda t: t["side"].map({1: "long spread", -1: "short spread"}), entry=lambda t: t["entry"].dt.strftime("%Y-%m-%d"),
        exit=lambda t: t["exit"].dt.strftime("%Y-%m-%d"), pnl=lambda t: t["pnl"].map(lambda v: pct(v, 3))).reset_index(drop=True).rename_axis("#"))

    sector_md = "\n".join(f"- **{s}** ({len(v)}): " + ", ".join(v) for s, v in SECTORS.items())
    ah_md = "\n".join(f"| {k} | {h} | {a} |" for k, (h, a) in AH_PAIRS.items())
    md = f"""# HK Stat-Arb — full results

Generated by `python run_hk_statarb.py` on {pd.Timestamp.today():%Y-%m-%d}. Data: Yahoo Finance, {cal[0]:%Y-%m-%d} → {end:%Y-%m-%d} (HK calendar, {len(cal)} bars).
Strategy period (all series): **{t_start:%Y-%m-%d} → {end:%Y-%m-%d}**. Costs: {COSTS.commission_bps:g} bp commission + {COSTS.slippage_bps:g} bp slippage per side on all trades,
{COSTS.stamp_bps:g} bp stamp duty per side on HK stock trades (2800.HK exempt), {COSTS.borrow_bps_annual:g} bp/yr borrow on gross shorts.

## Main results (net of costs)

{qc.report.to_markdown(main_tab)}

## Gross vs net and cost sensitivity

Sharpe ratio and CAGR when every cost component (commission, slippage, stamp duty, borrow) is scaled by the multiple shown.

{md_table(sens_md, lambda c, v: num(v) if "Sharpe" in c else pct(v))}

Break-even cost multiple = gross mean return ÷ mean cost drag at 1× (net return is zero at this multiple of today's HK costs).

{be_md}

## Robustness (not used for headline results)

BH-FDR only: trade only pairs that survive Benjamini-Hochberg at q = 0.10 across all tests in the window.
+1d delay: every trade executes one bar after the signal. Unhedged: same A/H long basket without the 2800.HK hedge.

{qc.report.to_markdown(rob_tab)}

## Pair formation

{md_table(form_summary.to_frame("Value").rename_axis("Statistic"), lambda c, v: num(v, 1) if isinstance(v, (float, np.floating)) else str(v))}

{md_table(sel_tab, lambda c, v: str(v))}

## Trades

{md_table(tstats, lambda c, v: v)}

Example pair **{y} / {x}** (window starting {d:%Y-%m-%d}; formation p = {sel_row['p']:.4f}, β = {sel_row['beta']:.2f}, half-life = {sel_row['half_life']:.1f} days). Chosen as the selected pair with the most entries, not by profit.

{ex_md}

## A/H premium

{md_table(ah_summary.to_frame("Value").rename_axis("Statistic"), lambda c, v: str(v))}

Most frequently held H shares in the implementable strategy (share of months held):

{md_table(hold_freq.to_frame("Share of months").rename_axis("Name"), lambda c, v: pct(v, 0))}

## Universes

HK pairs universe ({len(TICKERS)} names, pairs formed only within a sector):

{sector_md}

A/H dual listings ({len(AH_PAIRS)}):

| Name | H share | A share |
|---|---|---|
{ah_md}
"""
    (DOCS / "results.md").write_text(md)
    print(f"\nExample pair {y}/{x} window {d.date()} entries={best_n}")
    print(sens_md.to_string()); print(pd.Series(breakeven)); print(form_summary.to_string()); print(tstats.to_string())
    print(rob_tab.to_string()); print(ah_summary.to_string()); print(hold_freq)


if __name__ == "__main__":
    main()
