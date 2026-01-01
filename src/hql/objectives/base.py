"""Protocols for objectives usable by DHFC/SGHO trainers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Objective(Protocol):
    """Minimal objective interface."""

    dim: int

    def value(self, theta: np.ndarray, *, seed: int | None = None) -> float:
        ...

    def grad(
        self,
        theta: np.ndarray,
        indices: Sequence[int] | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return gradient with respect to the provided indices or the full vector."""
        ...
