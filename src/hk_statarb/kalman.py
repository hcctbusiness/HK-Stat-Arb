"""Kalman-filter dynamic hedge ratio (numpy implementation).

State-space model (log prices):

    observation:  y_t = α_t + β_t · x_t + ε_t,        ε_t ~ N(0, V_e)
    state:        θ_t = [β_t, α_t]' = θ_{t−1} + ω_t,   ω_t ~ N(0, W),  W = diag(q_β, q_α)

i.e. the hedge ratio and intercept follow independent random walks. In the pairs engine
x is centred on its formation-window mean, so a change in β tilts the fit around the
middle of the data instead of shifting the spread level by Δβ·x̄ (log prices are ~2–6). For each bar the
filter forms the **prior** θ_{t|t−1}, the one-step forecast ŷ_t = α_{t|t−1} + β_{t|t−1} x_t,
the innovation e_t = y_t − ŷ_t with variance Q_t = H_t P_{t|t−1} H_t' + V_e, and then
updates with gain K_t = P_{t|t−1} H_t' / Q_t. The trading signal uses only the prior
state and e_t / √Q_t, so nothing observed after the close of t enters the signal for t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanOutput:
    beta: np.ndarray        # prior β_{t|t−1}
    alpha: np.ndarray       # prior α_{t|t−1}
    innovation: np.ndarray  # e_t
    q: np.ndarray           # innovation variance Q_t
    beta_post: np.ndarray   # posterior β_{t|t}

    @property
    def z(self) -> np.ndarray:
        return self.innovation / np.sqrt(self.q)


def kalman_hedge(y: np.ndarray, x: np.ndarray, q_beta: float = 1e-5, q_alpha: float = 1e-6, ve: float = 1e-3,
                 theta0: tuple[float, float] = (1.0, 0.0), p0: float = 1.0) -> KalmanOutput:
    """Run the filter. ``theta0`` = initial (β, α); ``p0`` = initial state variance (diagonal).

    Defaults: q_β = 1e-5 (β random-walk s.d. ≈ 5% a year), q_α = 1e-6 (intercept s.d.
    ≈ 1.6% a year in log-price units). Fixed a priori, not tuned on the data.
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    W = np.diag([q_beta, q_alpha])
    theta = np.array(theta0, float)
    P = p0 * np.eye(2)
    out = {k: np.full(n, np.nan) for k in ("beta", "alpha", "e", "q", "bpost")}
    for t in range(n):
        R = P + W                                 # prior covariance
        if np.isnan(y[t]) or np.isnan(x[t]):      # missing bar: propagate prior only
            P = R
            out["beta"][t], out["alpha"][t] = theta
            continue
        H = np.array([x[t], 1.0])
        yhat = H @ theta
        Q = H @ R @ H + ve
        e = y[t] - yhat
        out["beta"][t], out["alpha"][t], out["e"][t], out["q"][t] = theta[0], theta[1], e, Q
        K = R @ H / Q
        theta = theta + K * e
        P = R - np.outer(K, H) @ R
        out["bpost"][t] = theta[0]
    return KalmanOutput(out["beta"], out["alpha"], out["e"], out["q"], out["bpost"])
