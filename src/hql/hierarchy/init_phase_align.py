"""Phase-aligned initializer for newly activated parameters."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..utils.math import TWOPI, wrap_torus


@dataclass
class PhaseAlignedInitializer:
    n_grid: int = 16
    torus: bool = True

    def init_new_params(
        self, obj, theta: np.ndarray, new_indices: Sequence[int], *, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta = theta.copy()
        grid = np.linspace(0.0, TWOPI, self.n_grid, endpoint=False)

        for j in new_indices:
            best_b = theta[j]
            best_mag = -np.inf
            shift = float(rng.random() * (TWOPI / self.n_grid))
            candidates = wrap_torus(grid + shift) if self.torus else (grid + shift)
            for b in candidates:
                theta_try = theta.copy()
                theta_try[j] = b
                gj = obj.grad(theta_try, indices=[j], seed=None)[0]
                mag = abs(gj)
                if mag > best_mag:
                    best_mag = mag
                    best_b = b
            theta[j] = best_b

        if self.torus:
            theta = wrap_torus(theta)
        return theta
