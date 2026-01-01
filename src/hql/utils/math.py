"""Utility math helpers for torus-wrapped optimization."""
from __future__ import annotations

import numpy as np

TWOPI = 2.0 * np.pi


def wrap_torus(theta: np.ndarray) -> np.ndarray:
    """Wrap parameters to the [0, 2π) torus coordinate-wise."""
    return np.mod(theta, TWOPI)
