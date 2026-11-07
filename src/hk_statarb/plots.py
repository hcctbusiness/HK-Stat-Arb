"""Charts for the HK stat-arb study (matplotlib, HCCT palette shared with QuantCore)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantcore.report import PALETTE  # noqa: E402

from .signals import ENTER, EXIT, STOP, TIME  # noqa: E402

FOOT = "HCCT Group · HK Stat-Arb"


def _style(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path):
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="#7A7A7A")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=110); plt.close(fig)
    return path


def formation_chart(windows: pd.DataFrame, selected_hl: pd.Series, all_hl: pd.Series, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(windows))
    a1.bar(x - 0.27, windows["raw_p<0.05"], 0.27, color=PALETTE[2], label="Raw p < 0.05")
    a1.bar(x, windows["selected"], 0.27, color=PALETTE[0], label="Selected (p<0.05 + filters, top 20)")
    a1.bar(x + 0.27, windows["bh_q10"], 0.27, color=PALETTE[1], label="Significant after BH-FDR (q = 0.10)")
    a1.plot(x, 0.05 * windows["tested"], color="k", ls="--", lw=1.2, label="Expected false positives (5% of tests)")
    labels = [d.strftime("%Y-%m") for d in windows["trading_start"]]
    a1.set_xticks(x[::2], labels[::2], rotation=45, ha="right", fontsize=8)
    a1.set_ylabel("Number of pairs"); a1.legend(frameon=False, fontsize=8.5, loc="upper left")
    a1.set_title("Cointegrated same-sector pairs per formation window", loc="left", fontweight="bold")
    _style(a1)
    bins = np.arange(0, 66, 3)
    a2.hist(all_hl[np.isfinite(all_hl)].clip(upper=63), bins=bins, density=True, color=PALETTE[3], alpha=0.45, label="All tested pairs")
    a2.hist(selected_hl.clip(upper=63), bins=bins, density=True, color=PALETTE[0], alpha=0.8, label="Selected pairs")
    a2.axvline(selected_hl.median(), color=PALETTE[1], lw=1.5, label=f"Selected median {selected_hl.median():.1f}d")
    a2.set_xlabel("OU half-life (trading days, formation window)"); a2.set_ylabel("Density")
    a2.legend(frameon=False, fontsize=8.5)
    a2.set_title("Half-life distribution", loc="left", fontweight="bold"); _style(a2)
    fig.tight_layout()
    return _save(fig, path)


def _marks(ax, sig: pd.DataFrame):
    ev = sig["event"]
    for code, m, c, lab in [(ENTER, "^", PALETTE[4], "Entry"), (EXIT, "o", PALETTE[0], "Exit at mean"),
                            (STOP, "x", PALETTE[5], "Stop-loss"), (TIME, "s", PALETTE[1], "Time stop")]:
        d = sig.index[ev == code]
        if len(d):
            ax.scatter(d, sig.loc[d, "z"], marker=m, color=c, s=45, zorder=5, label=lab)


def example_pair_chart(logpx: pd.DataFrame, y: str, x: str, form_start, sig_s: pd.DataFrame, sig_k: pd.DataFrame,
                       beta_form: float, names: dict, path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=False, gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 1]})
    win = logpx.loc[form_start:sig_s.index[-1], [y, x]]
    nrm = win - win.iloc[0]
    a = axes[0]
    a.plot(nrm[y], color=PALETTE[0], label=f"{y} {names.get(y, '')}")
    a.plot(nrm[x], color=PALETTE[1], label=f"{x} {names.get(x, '')}")
    a.axvspan(sig_s.index[0], sig_s.index[-1], color=PALETTE[2], alpha=0.08, label="Trading window")
    a.set_title("Log total-return prices (rebased to 0 at formation start)", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=9, ncol=3, loc="upper left"); _style(a)
    for a, sig, ttl in [(axes[1], sig_s, "Static OLS spread z-score (formation μ, σ)"),
                        (axes[2], sig_k, "Kalman z-score  e_t / √Q_t")]:
        a.plot(sig.index, sig["z"], color=PALETTE[3], lw=1.1)
        for lv, ls in [(2, "--"), (-2, "--"), (4, ":"), (-4, ":"), (0, "-")]:
            a.axhline(lv, color="k", lw=0.7, ls=ls, alpha=0.6)
        _marks(a, sig)
        a.set_title(ttl, loc="left", fontweight="bold"); _style(a)
        a.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left")
        lim = max(4.8, float(np.nanmax(np.abs(sig["z"]))) + 0.5)
        a.set_ylim(-lim, lim * 1.35)
    a = axes[3]
    a.plot(sig_k.index, sig_k["beta"], color=PALETTE[0], label="Kalman β (prior, x centred)")
    a.axhline(beta_form, color=PALETTE[1], ls="--", label=f"Static OLS β = {beta_form:.2f}")
    a.set_title("Hedge ratio during the trading window", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=9); _style(a)
    for a in axes:
        a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    fig.suptitle(f"Example pair {y} / {x} — trading window from {sig_s.index[0]:%Y-%m-%d}", x=0.06, ha="left",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save(fig, path)


def equity_chart(series: dict[str, pd.Series], path, title: str, styles: dict | None = None):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    styles = styles or {}
    for i, (n, r) in enumerate(series.items()):
        st = {"color": PALETTE[i % len(PALETTE)], "lw": 1.6} | styles.get(n, {})
        ax.plot((1 + r).cumprod(), label=n, **st)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_ylabel("Growth of 1")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left"); _style(ax)
    fig.tight_layout()
    return _save(fig, path)


def ah_premium_chart(idx: pd.DataFrame, latest: pd.Series, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    a1.fill_between(idx.index, idx["q25"], idx["q75"], color=PALETTE[2], alpha=0.25, label="Inter-quartile range")
    a1.plot(idx["median"], color=PALETTE[0], lw=1.6, label="Median A/H premium")
    a1.axhline(0, color="k", lw=0.7)
    a1.yaxis.set_major_formatter(mt.PercentFormatter(1.0))
    a1.set_title("A-share premium over H-share (A in HKD / H − 1)", loc="left", fontweight="bold")
    a1.legend(frameon=False, loc="upper left"); _style(a1)
    s = latest.dropna().sort_values()
    colors = [PALETTE[5] if v < 0 else PALETTE[0] for v in s]
    a2.barh(range(len(s)), s.values, color=colors)
    a2.set_yticks(range(len(s)), s.index, fontsize=8.5)
    a2.xaxis.set_major_formatter(mt.PercentFormatter(1.0)); a2.axvline(0, color="k", lw=0.7)
    a2.set_title(f"Premium by name, {latest.name:%Y-%m-%d}", loc="left", fontweight="bold"); _style(a2)
    fig.tight_layout()
    return _save(fig, path)


def neutrality_chart(rets: dict[str, pd.Series], hsi: pd.Series, path, window: int = 252):
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for i, (n, r) in enumerate(rets.items()):
        df = pd.concat([r, hsi], axis=1, join="inner").dropna()
        ax.plot(df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1]), color=PALETTE[i % len(PALETTE)], lw=1.4, label=n)
    ax.axhline(0, color="k", lw=0.7); ax.set_ylim(-1, 1)
    ax.set_title("Rolling 1-year correlation of daily returns with the Hang Seng Index", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left"); _style(ax)
    fig.tight_layout()
    return _save(fig, path)
