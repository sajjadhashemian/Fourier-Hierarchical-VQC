"""Group samplers for SGHO."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class GroupSampler:
    groups: Sequence[Sequence[int]]
    probs: np.ndarray

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.groups), p=self.probs))

    @staticmethod
    def uniform(groups: Sequence[Sequence[int]]) -> GroupSampler:
        m = len(groups)
        return GroupSampler(groups=groups, probs=np.ones(m) / m)
