"""Offline unit tests on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

import quantcore as qc
from hk_statarb import ah, costs as hc, pairs as pr
from hk_statarb import cointegration as ci
from hk_statarb.data import lagged_fx, to_calendar
from hk_statarb.kalman import kalman_hedge
from hk_statarb.signals import ENTER, EXIT, STOP, TIME, zscore_positions


def ou(n, phi, sd, rng, x0=0.0):
    s = np.empty(n); s[0] = x0
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0, sd)
    return s


@pytest.fixture
def coint_pair():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.01, 500)) + 3.0
    y = 0.2 + 1.3 * x + ou(500, 0.9, 0.01, rng)
    return y, x


def test_engle_granger_detects_cointegration(coint_pair):
    y, x = coint_pair
    assert ci.engle_granger(y, x) < 0.01
    rng = np.random.default_rng(1)
    rw1, rw2 = np.cumsum(rng.normal(0, 0.01, 500)), np.cumsum(rng.normal(0, 0.01, 500))
    assert ci.engle_granger(rw1, rw2) > 0.10
    a, b = ci.ols_hedge(y, x)
    assert b == pytest.approx(1.3, abs=0.1)


def test_half_life_matches_ou_parameter():
    rng = np.random.default_rng(2)
    phi = 0.95
    s = ou(20000, phi, 1.0, rng)
    assert ci.half_life(s) == pytest.approx(np.log(2) / -np.log(phi), rel=0.1)   # ≈ 13.5 bars
    assert ci.half_life(np.cumsum(rng.normal(size=500)) * 0 + np.arange(500.0)) == float("inf")


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert ci.benjamini_hochberg(p, 0.05).sum() == 2
    assert not ci.benjamini_hochberg(np.full(5, 0.5), 0.1).any()


def test_kalman_recovers_time_varying_beta():
    rng = np.random.default_rng(3)
    n = 1500
    x = 3 + np.cumsum(rng.normal(0, 0.02, n))
    beta = 1.0 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n))
    y = 0.1 + beta * x + rng.normal(0, 0.01, n)
    kf = kalman_hedge(y, x, q_beta=1e-4, q_alpha=1e-4, ve=1e-4, theta0=(1.0, 0.0), p0=1.0)
    err = np.abs(kf.beta_post[200:] - beta[200:])
    assert err.mean() < 0.08
    assert np.corrcoef(kf.beta_post[200:], beta[200:])[0, 1] > 0.95
    # prior at t equals posterior at t-1 (no same-bar information in the signal)
    assert np.allclose(kf.beta[1:], kf.beta_post[:-1])


def test_kalman_signal_has_no_lookahead():
    rng = np.random.default_rng(4)
    x = 3 + np.cumsum(rng.normal(0, 0.02, 400)); y = 1.2 * x + rng.normal(0, 0.01, 400)
    base = kalman_hedge(y, x, ve=1e-4).z
    y2 = y.copy(); y2[300:] += 5.0
    alt = kalman_hedge(y2, x, ve=1e-4).z
    assert np.allclose(base[:300], alt[:300])


def test_zscore_state_machine_entry_exit_stop_time():
    z = np.array([0.0, 2.5, 1.0, -0.1, 0.0, -2.5, -4.5, -3.0, -2.5, 0.5, -2.2, -2.3, -2.4, -2.5])
    pos, ev = zscore_positions(z, entry=2, exit_z=0, stop=4, max_hold=3)
    assert list(pos[:5]) == [0, -1, -1, 0, 0]                       # short spread, exit at mean
    assert ev[1] == ENTER and ev[3] == EXIT
    assert pos[5] == 1 and ev[6] == STOP and pos[6] == 0            # long, stopped beyond -4
    assert pos[7] == 0 and pos[8] == 0                              # disarmed until z crosses 0
    assert ev[10] == ENTER and pos[10] == 1                         # re-armed after crossing at bar 9
    assert ev[13] == TIME and pos[13] == 0                          # time stop after 3 bars
    p2, e2 = zscore_positions(np.array([0, 2.5, np.nan, 2.5]))
    assert p2[2] == 0 and e2[2] == EXIT                             # missing data closes


def synthetic_universe(n=700, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.012, n))
    cols = {}
    for i in range(6):
        cols[f"A{i}"] = np.exp(4 + common + ou(n, 0.92, 0.01, rng))
    for i in range(4):
        cols[f"B{i}"] = np.exp(4 + np.cumsum(rng.normal(0, 0.015, n)))
    px = pd.DataFrame(cols, index=idx)
    sectors = {c: ("S1" if c.startswith("A") else "S2") for c in px}
    return px, sectors


def test_pair_selection_finds_cointegrated_sector():
    px, sectors = synthetic_universe()
    tests = ci.test_pairs(np.log(px.iloc[:252]), sectors)
    sel = ci.select_pairs(tests)
    assert len(sel) > 0 and (sel["sector"] == "S1").mean() > 0.8


def test_pairs_engine_no_lookahead_and_dollar_neutral():
    px, sectors = synthetic_universe()
    params = pr.PairsParams(formation=252, trading=126)
    base = pr.run_pairs(px, sectors, params, method="kalman")
    shocked = px.copy(); shocked.iloc[500:] *= np.linspace(1, 3, shocked.shape[1])
    alt = pr.run_pairs(shocked, sectors, params, method="kalman")
    pd.testing.assert_frame_equal(base.weights.iloc[:500], alt.weights.iloc[:500])
    assert len(base.trades) > 0                                           # it traded
    gross = base.weights.abs().sum(axis=1)
    assert gross.max() <= params.gross_cap + 1e-9
    # every pair is entered dollar-neutral: on any day where the only change is a new entry
    # (no open book before), the book sums to zero
    first_entry = base.trades["entry"].min()
    assert base.weights.loc[first_entry].sum() == pytest.approx(0.0, abs=1e-12)
    assert gross.loc[first_entry] > 0
    # the drifting book stays close to dollar-neutral (legs move together)
    assert base.weights.sum(axis=1).abs().max() < 0.05 * max(gross.max(), 1e-9) + 0.05
    tg = pr.dollar_neutral_targets({("A0", "A1"): 1, ("A2", "A1"): -1, ("B0", "B1"): 1},
                                   {("A0", "A1"): 0.3, ("A2", "A1"): 0.2, ("B0", "B1"): 0.4})
    assert tg.sum() == pytest.approx(0.0)


def test_engine_charges_costs_only_on_trades():
    px, sectors = synthetic_universe()
    res = pr.run_pairs(px, sectors, pr.PairsParams(), method="static")
    bt = hc.run(res.weights, px, hc.HKCosts(), "t")
    w = res.weights.shift(1).fillna(0.0)
    changed_book = (w.abs().sum(axis=1) > 0) | (w.shift(1).fillna(0).abs().sum(axis=1) > 0)
    quiet = bt.turnover[~changed_book]
    assert (quiet.abs() < 1e-12).all()
    # on days without entries/exits the drifted book means ~zero turnover
    n_events = len(res.trades) * 2
    assert (bt.turnover > 1e-9).sum() <= n_events + 2


def test_stamp_duty_exempts_etf_and_scales():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame({"0005.HK": 100.0, "2800.HK": 20.0}, index=idx)
    w = pd.DataFrame({"0005.HK": [0, 1, 1, 1, 1], "2800.HK": [0, -1, -1, -1, -1]}, index=idx, dtype=float)
    c = hc.HKCosts(commission_bps=0, slippage_bps=0, stamp_bps=10, borrow_bps_annual=0)
    res = hc.run(w, px, c, "x")
    assert res.costs.sum() == pytest.approx(10 / 1e4)          # only the stock leg pays stamp
    res2 = hc.run(w, px, c.scaled(2), "x")
    assert res2.costs.sum() == pytest.approx(20 / 1e4)


def test_ah_premium_fx_conversion_and_alignment():
    hk = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    sh = pd.DatetimeIndex(["2024-01-02", "2024-01-04", "2024-01-05"])          # mainland shut on 01-03
    fxd = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    h = pd.DataFrame({"H": [5.0, 5.0, 5.0, 5.0]}, index=hk)
    a = pd.DataFrame({"A": [6.0, 7.0, 8.0]}, index=sh)
    fx = pd.Series([1.10, 1.10, 1.20, 1.30, 1.40], index=fxd)
    prem = ah.premium(h, a, fx, {"X": ("H", "A")}, hk)["X"]
    # 01-02: A 6 CNY × FX(01-01)=1.10 → 6.6 HKD vs H 5 → +32%
    assert prem.iloc[0] == pytest.approx(6 * 1.10 / 5 - 1)
    # 01-03: A stale (6, as-of), FX of the previous bar (01-02) = 1.10
    assert prem.iloc[1] == pytest.approx(6 * 1.10 / 5 - 1)
    assert prem.iloc[2] == pytest.approx(7 * 1.20 / 5 - 1)
    assert prem.iloc[3] == pytest.approx(8 * 1.30 / 5 - 1)
    # as-of join never uses a future value and respects the staleness limit
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-10"]))
    cal = pd.bdate_range("2024-01-01", "2024-01-10")
    m = to_calendar(s, cal, limit=3)
    assert m.loc["2024-01-09"] != 2.0 and np.isnan(m.loc["2024-01-09"]) and m.loc["2024-01-03"] == 1.0
    assert lagged_fx(fx, hk).loc["2024-01-02"] == 1.10


def test_ah_signal_no_lookahead_and_hedge_neutral():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2018-01-01", periods=600)
    prem = pd.DataFrame(rng.normal(0.3, 0.05, (600, 12)), index=idx, columns=[f"N{i}" for i in range(12)])
    z = ah.premium_z(prem)
    prem2 = prem.copy(); prem2.iloc[400:] += 1.0
    z2 = ah.premium_z(prem2)
    pd.testing.assert_frame_equal(z.iloc[:400], z2.iloc[:400])
    h_of = {c: f"{c}.HK" for c in prem}
    betas = pd.DataFrame(0.8, index=idx, columns=list(h_of.values()))
    tg = ah.h_only_targets(z, h_of, betas, n_long=5)
    assert np.allclose(tg.drop(columns="2800.HK").sum(axis=1), 1.0)
    assert np.allclose(tg["2800.HK"], -0.8)


def test_drift_targets_has_no_turnover_between_rebalances():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=120)
    px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (120, 3)), axis=0), idx, list("abc"))
    me = ah.month_ends(idx)
    tg = pd.DataFrame([[0.5, 0.5, -1.0]] * len(me), index=me, columns=list("abc"))
    w = ah.drift_targets(tg, px)
    res = qc.backtest(w, px, qc.CostModel(0, 0), lag=1)
    pos = px.index.get_indexer(me) + 1                       # held weights change the bar after
    rb_next = set(px.index[pos[pos < len(px)]])
    quiet = res.turnover[[d not in rb_next for d in res.turnover.index]]
    assert quiet.max() < 1e-12
    assert res.turnover[sorted(rb_next)].iloc[1:].gt(0).all()
