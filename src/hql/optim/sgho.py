"""Stochastic grouped hierarchical optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..smoothing.gaussian import GaussianSmoother
from ..utils.math import wrap_torus


@dataclass
class SGHOResult:
    theta: np.ndarray
    history: dict[str, list]
    meta: dict[str, Any]


@dataclass
class SGHOTrainer:
    torus: bool = True

    def run(
        self,
        base_obj,
        schedule,
        theta0: np.ndarray,
        group_sampler,
        initializer=None,
        *,
        seed: int | None = None,
    ) -> SGHOResult:
        rng = np.random.default_rng(seed)
        theta = theta0.copy()

        hist: dict[str, list] = {
            "iter": [],
            "loss": [],
            "sigma": [],
            "stage": [],
            "group": [],
            "lr": [],
            "method": [],
        }
        meta: dict[str, Any] = {"expansion": []}

        prev_active: set[int] = set()
        it = 0

        for si, st in enumerate(schedule.stages):
            active_set = set(st.active)
            new_params = sorted(active_set - prev_active)

            sm = GaussianSmoother(
                base=base_obj,
                sigma=st.sigma,
                n_samples=st.smooth_samples,
                torus=self.torus,
            )

            if len(new_params) > 0:
                g_new_before = sm.grad(theta, indices=new_params, seed=int(rng.integers(1 << 30)))
                meta["expansion"].append(
                    {
                        "stage": si,
                        "iter": it,
                        "new_params": new_params,
                        "grad_new_before_norm": float(np.linalg.norm(g_new_before)),
                    }
                )
                if initializer is not None:
                    theta = initializer.init_new_params(
                        sm, theta, new_params, seed=int(rng.integers(1 << 30))
                    )
                g_new_after = sm.grad(theta, indices=new_params, seed=int(rng.integers(1 << 30)))
                meta["expansion"][-1]["grad_new_after_norm"] = float(np.linalg.norm(g_new_after))

            for t in range(st.steps):
                lr_t = float(st.lr.lr(t)) if hasattr(st, "lr") else st.lr

                gi = int(group_sampler.sample(rng))
                group = [j for j in group_sampler.groups[gi] if j in active_set]
                if not group:
                    continue

                pi = float(group_sampler.probs[gi])
                if pi <= 0:
                    raise ValueError("group sampling probability must be positive")

                u = sm.grad(theta, indices=group, seed=int(rng.integers(1 << 30)))
                theta[group] -= (lr_t / pi) * u

                if self.torus:
                    theta = wrap_torus(theta)

                loss = sm.value(theta, seed=int(rng.integers(1 << 30)))

                hist["iter"].append(it)
                hist["loss"].append(float(loss))
                hist["sigma"].append(float(st.sigma))
                hist["stage"].append(int(si))
                hist["group"].append(int(gi))
                hist["lr"].append(float(lr_t))
                hist["method"].append("sgho")
                it += 1

            prev_active = set(active_set)

        return SGHOResult(theta=theta, history=hist, meta=meta)
