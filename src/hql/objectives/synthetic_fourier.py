"""Synthetic 2D Fourier-structured objective with analytic gradients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.math import wrap_torus


@dataclass
class SyntheticFourierObjective:
    """Two-parameter objective with controllable Fourier spectrum."""

    ks: tuple[int, ...] = (9, 11, 13)
    a: tuple[float, ...] = (1.0, 0.7, 0.5)
    eps: float = 0.2
    delta: float = 0.15
    dim: int = 2

    def value(self, theta: np.ndarray, *, seed: int | None = None) -> float:
        x, y = wrap_torus(theta)
        f1 = -np.cos(x) + 0.15 * np.cos(2 * x)
        hf = 0.0
        for aj, kj in zip(self.a, self.ks, strict=False):
            hf += aj * (np.cos(kj * y) - 1.0)
        coupling = np.cos(x + y) - np.cos(x)
        return float(f1 + self.eps * hf + self.delta * coupling)

    def grad(
        self,
        theta: np.ndarray,
        indices: tuple[int, ...] | list[int] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        x, y = wrap_torus(theta)
        dfdx = (np.sin(x) - 0.3 * np.sin(2 * x)) + self.delta * (-np.sin(x + y) + np.sin(x))
        dfdy = 0.0
        for aj, kj in zip(self.a, self.ks, strict=False):
            dfdy += aj * (-kj * np.sin(kj * y))
        dfdy = self.eps * dfdy + self.delta * (-np.sin(x + y))

        g = np.array([dfdx, dfdy], dtype=float)
        if indices is None:
            return g
        return g[np.array(indices, dtype=int)]
