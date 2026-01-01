"""Main synthetic experiment comparing naive vs phase-aligned DHFC."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hql.analysis.trap_density import trap_density_1d_slices
from hql.hierarchy.init_phase_align import PhaseAlignedInitializer
from hql.hierarchy.schedule import Schedule, Stage
from hql.objectives.synthetic_fourier import SyntheticFourierObjective
from hql.optim.dhfc import DHFCTrainer
from hql.optim.group_samplers import GroupSampler
from hql.optim.lr import LRSchedule
from hql.optim.sgho import SGHOTrainer
from hql.smoothing.gaussian import GaussianSmoother

RESULTS_CSV = Path("results_synth.csv")
SUMMARY_JSON = Path("summary_synth.json")


def build_schedule(smooth_samples: int = 64) -> Schedule:
    return Schedule(
        stages=[
            Stage(
                active=[0],
                sigma=0.6,
                steps=150,
                lr=LRSchedule(lr0=0.10),
                smooth_samples=smooth_samples,
            ),
            Stage(
                active=[0, 1],
                sigma=0.6,
                steps=50,
                lr=LRSchedule(lr0=0.08),
                smooth_samples=smooth_samples,
            ),
            Stage(
                active=[0, 1],
                sigma=0.3,
                steps=80,
                lr=LRSchedule(lr0=0.06),
                smooth_samples=smooth_samples,
            ),
            Stage(
                active=[0, 1],
                sigma=0.1,
                steps=120,
                lr=LRSchedule(lr0=0.04),
                smooth_samples=smooth_samples,
            ),
        ]
    )


def main() -> None:
    obj = SyntheticFourierObjective()
    theta0 = np.array([0.0, 0.0])

    schedule = build_schedule()
    trainer = DHFCTrainer(torus=True)
    phase_init = PhaseAlignedInitializer(n_grid=16, torus=True)

    res_naive = trainer.run(obj, schedule, theta0, initializer=None, seed=1)
    res_phase = trainer.run(obj, schedule, theta0, initializer=phase_init, seed=1)

    # SGHO example with two singleton groups (optional, logged with method tag)
    groups = [[0], [1]]
    sampler = GroupSampler.uniform(groups)
    sgho_trainer = SGHOTrainer(torus=True)
    res_sgho = sgho_trainer.run(obj, schedule, theta0, sampler, initializer=phase_init, seed=2)

    df_naive = pd.DataFrame(res_naive.history)
    df_naive["method"] = "naive"
    df_phase = pd.DataFrame(res_phase.history)
    df_phase["method"] = "phase_align"
    df_sgho = pd.DataFrame(res_sgho.history)
    df_sgho["method"] = "sgho_phase"

    df = pd.concat([df_naive, df_phase, df_sgho], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)

    stage2 = schedule.stages[1]
    sm_stage2 = GaussianSmoother(obj, sigma=stage2.sigma, n_samples=256, torus=True)
    g_new_naive = abs(sm_stage2.grad(np.array([0.0, 0.0]), indices=[1], seed=0)[0])

    summary = {
        "expand_iter": (
            int(res_naive.meta["expansion"][0]["iter"])
            if res_naive.meta["expansion"]
            else 0
        ),
        "grad_new": {
            "naive": g_new_naive,
            "phase_align": (
                float(res_phase.meta["expansion"][0]["grad_new_after_norm"])
                if res_phase.meta["expansion"]
                else 0.0
            ),
        },
    }

    sigma_list = [0.6, 0.3, 0.1, 0.0]
    td_vals = []
    for s in sigma_list:
        sm = GaussianSmoother(obj, sigma=s, n_samples=64, torus=True) if s > 0 else obj
        td_vals.append(
            trap_density_1d_slices(sm, sigma_list=[s], n_slices=50, n_grid=256, seed=0)[0]
        )
    summary["trap_density"] = {"sigma": sigma_list, "value": td_vals}

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved {RESULTS_CSV} and {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
