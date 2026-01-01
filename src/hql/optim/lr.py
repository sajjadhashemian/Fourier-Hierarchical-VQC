"""Learning-rate schedules used by trainers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScheduleType = Literal["constant", "sqrt", "linear"]


@dataclass(frozen=True)
class LRSchedule:
    kind: ScheduleType = "constant"
    lr0: float = 1e-2

    def lr(self, t: int) -> float:
        if self.kind == "constant":
            return self.lr0
        if self.kind == "sqrt":
            return self.lr0 / np.sqrt(t + 1.0)
        if self.kind == "linear":
            return self.lr0 / (t + 1.0)
        raise ValueError(f"unknown schedule kind: {self.kind}")
