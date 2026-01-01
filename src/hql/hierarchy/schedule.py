"""Stage definitions for hierarchical training."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..optim.lr import LRSchedule


@dataclass(frozen=True)
class Stage:
    active: Sequence[int]
    sigma: float
    steps: int
    lr: LRSchedule
    smooth_samples: int
    seed: int | None = None
    log_theta: bool = False


@dataclass(frozen=True)
class Schedule:
    stages: Sequence[Stage]
