"""Z-score trading state machine for a single spread.

Position convention: +1 = long the spread (long y, short x), −1 = short the spread.

* Flat and armed: enter short if entry < z < stop, long if −stop < z < −entry.
* Long: exit when z ≥ exit_z (mean reached); stop-loss when z ≤ −stop;
  time stop after ``max_hold`` bars.
* Short: mirror image.
* After a stop-loss or time stop the pair is disarmed until z crosses back through
  ``exit_z`` (prevents immediately re-entering a broken spread).
* A missing z closes any open position.

Decisions at bar t use z up to and including t only.
"""
from __future__ import annotations

import numpy as np

FLAT, ENTER, EXIT, STOP, TIME = 0, 1, 2, 3, 4


def zscore_positions(z: np.ndarray, entry: float = 2.0, exit_z: float = 0.0, stop: float = 4.0,
                     max_hold: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (position after the close of each bar, event code per bar)."""
    z = np.asarray(z, float)
    n = len(z)
    pos = np.zeros(n, int)
    ev = np.zeros(n, int)
    p, held, armed, last_side = 0, 0, True, 0.0
    for t in range(n):
        zt = z[t]
        if np.isnan(zt):
            if p != 0:
                ev[t] = EXIT
            p, held = 0, 0
            pos[t] = p
            continue
        if p == 0:
            if not armed and np.sign(zt - exit_z) != last_side:   # crossed (or touched) the mean
                armed = True
            if armed and entry < abs(zt) < stop:
                p, held = (-1 if zt > 0 else 1), 0
                ev[t] = ENTER
        else:
            held += 1
            if (p == 1 and zt >= exit_z) or (p == -1 and zt <= exit_z):
                p, ev[t] = 0, EXIT
            elif (p == 1 and zt <= -stop) or (p == -1 and zt >= stop):
                last_side = np.sign(zt - exit_z)
                p, ev[t], armed = 0, STOP, False
            elif max_hold is not None and held >= max_hold:
                last_side = np.sign(zt - exit_z)
                p, ev[t], armed = 0, TIME, False
        pos[t] = p
    return pos, ev
