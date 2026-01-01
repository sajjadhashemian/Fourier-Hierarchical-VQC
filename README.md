# Fourier-Hierarchical-VQC

Prototype code for Fourier-structured hierarchical optimization with deterministic (DHFC) and stochastic grouped (SGHO) training, phase-aligned initialization, and smoothing-based trap analysis.

## Getting started

```bash
python -m pip install -e .[dev]
```

## Run the synthetic experiment

```bash
python -m hql.experiments.exp_synth_one_figure
python -m hql.analysis.plots --csv results_synth.csv --summary summary_synth.json --out figures/synth_composite.png
```

The experiment logs per-iteration losses for naive and phase-aligned DHFC (plus an SGHO baseline), records gradient activation at expansion, estimates trap density versus smoothing, and produces a composite figure.
