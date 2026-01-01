"""Trap density estimator via random 1D slices on the torus."""
from __future__ import annotations

import numpy as np

from ..utils.math import TWOPI, wrap_torus


def trap_density_1d_slices(obj, sigma_list, n_slices: int = 200, n_grid: int = 1024, seed: int = 0):
    rng = np.random.default_rng(seed)
    td = []

    for _ in sigma_list:
        counts = []
        for _ in range(n_slices):
            theta0 = rng.random(2) * TWOPI
            v = rng.standard_normal(2)
            v /= np.linalg.norm(v)
            ts = np.linspace(0.0, TWOPI, n_grid, endpoint=False)
            vals = np.array([obj.value(wrap_torus(theta0 + t * v)) for t in ts])
            prev_vals = np.roll(vals, 1)
            next_vals = np.roll(vals, -1)
            is_min = (vals < prev_vals) & (vals < next_vals)
            counts.append(int(np.sum(is_min)))
        td.append(float(np.mean(counts)))
    return td
