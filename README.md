# Fourier-Hierarchical-VQC

Prototype code for Fourier-structured hierarchical optimization with deterministic (DHFC) and stochastic grouped (SGHO) training, phase-aligned initialization, and smoothing-based trap analysis.

## Getting started

```bash
python -m pip install -e .[dev]
```

### Installing inside a Conda environment

```bash
# create and activate an isolated env (Python 3.11 matches the lockfile targets)
conda create -n hql python=3.11 -y
conda activate hql

# install the base package with dev tools (ruff/pytest/mypy/black)
python -m pip install -e .[dev]

# optional extras
# pip install -e ".[ml]"            # torch + scikit-learn
# pip install -e ".[pennylane]"      # PennyLane stack
# pip install -e ".[pennylane,ml]"  # common ML + PennyLane
# pip install -e ".[qiskit]"         # Qiskit simulator stack
```

## Run the synthetic experiment

```bash
python -m hql.experiments.exp_synth_one_figure
python -m hql.analysis.plots --csv results_synth.csv --summary summary_synth.json --out figures/synth_composite.png
```

The experiment logs per-iteration losses for naive and phase-aligned DHFC (plus an SGHO baseline), records gradient activation at expansion, estimates trap density versus smoothing, and produces a composite figure.
