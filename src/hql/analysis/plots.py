"""Composite plotting utilities for the synthetic experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_composite_figure(csv_path: Path, summary_path: Path, out_path: Path | None) -> None:
    df = pd.read_csv(csv_path)
    if "iter" not in df.columns:
        df = df.copy()
        df["iter"] = df.groupby("method").cumcount()

    summ = _load_summary(summary_path)

    if "expand_iter" in summ:
        expand_iter = int(summ["expand_iter"])
    else:
        if "stage" in df.columns:
            stage1 = df[df["stage"] == 1]
            expand_iter = int(stage1["iter"].min()) if len(stage1) else int(df["iter"].min())
        else:
            expand_iter = int(df["iter"].min())

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[2.5, 1],
        height_ratios=[1, 1],
        wspace=0.35,
        hspace=0.4,
    )
    ax = fig.add_subplot(gs[:, 0])

    for method in sorted(df["method"].unique()):
        d = df[df["method"] == method].sort_values("iter")
        ax.plot(d["iter"].to_numpy(), d["loss"].to_numpy(), label=method)

    ax.axvline(expand_iter, linestyle="--")
    ax.set_xlabel("iteration")
    ax.set_ylabel("smoothed loss (as logged)")
    ax.legend(loc="best")

    ax_g = fig.add_subplot(gs[0, 1])
    grad_new = summ.get("grad_new", None)

    if isinstance(grad_new, dict) and ("naive" in grad_new) and ("phase_align" in grad_new):
        labels = ["naive", "phase_align"]
        values = [float(grad_new["naive"]), float(grad_new["phase_align"])]
        ax_g.bar(labels, values)
        ax_g.set_title(r"$\|\nabla_{\mathrm{new}}\|$ at expansion")
        ax_g.set_ylabel("magnitude")
    else:

        def slope_proxy(method: str) -> float:
            d = df[df["method"] == method].sort_values("iter")
            d = d[d["iter"] >= expand_iter].head(3)
            if len(d) < 2:
                return 0.0
            return float(d["loss"].iloc[0] - d["loss"].iloc[1])

        labels = ["naive", "phase_align"]
        values = [slope_proxy("naive"), slope_proxy("phase_align")]
        ax_g.bar(labels, values)
        ax_g.set_title("activation proxy (fallback)")
        ax_g.set_ylabel("Δloss")

    ax_td = fig.add_subplot(gs[1, 1])
    td = summ.get("trap_density", None)

    if isinstance(td, dict) and ("sigma" in td) and ("value" in td):
        sig = [float(x) for x in td["sigma"]]
        val = [float(x) for x in td["value"]]
        ax_td.plot(sig, val, marker="o")
        ax_td.set_xlabel(r"$\sigma$")
        ax_td.set_ylabel("trap density")
        ax_td.set_title("trap density vs smoothing")
    else:
        ax_td.text(0.05, 0.5, "No trap_density in summary_synth.json", transform=ax_td.transAxes)
        ax_td.set_xticks([])
        ax_td.set_yticks([])

    fig.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results_synth.csv")
    parser.add_argument("--summary", type=str, default="summary_synth.json")
    parser.add_argument("--out", type=str, default="figures/synth_composite.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    summary_path = Path(args.summary)
    out_path = Path(args.out) if args.out else None

    make_composite_figure(csv_path, summary_path, out_path)


if __name__ == "__main__":
    main()
