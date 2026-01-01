"""Monte Carlo Gaussian smoothing wrappers for objectives."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..utils.math import wrap_torus


@dataclass
class GaussianSmoother:
    base: any
    sigma: float
    n_samples: int
    torus: bool = True

    def value(self, theta: np.ndarray, *, seed: int | None = None) -> float:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(self.n_samples):
            z = rng.standard_normal(size=theta.shape)
            th = theta + self.sigma * z
            if self.torus:
                th = wrap_torus(th)
            vals.append(self.base.value(th, seed=None))
        return float(np.mean(vals))

    def base_grad_point(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Gradient of base objective at a single smoothed sample."""

        th = theta + self.sigma * z
        if self.torus:
            th = wrap_torus(th)
        return self.base.grad(th, seed=None)

    def grad(
        self,
        theta: np.ndarray,
        indices: Sequence[int] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        grads = []
        for _ in range(self.n_samples):
            z = rng.standard_normal(size=theta.shape)
            th = theta + self.sigma * z
            if self.torus:
                th = wrap_torus(th)
            grads.append(self.base.grad(th, indices=indices, seed=None))
        return np.mean(np.stack(grads, axis=0), axis=0)
