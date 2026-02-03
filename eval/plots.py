"""Generate figures: Pareto frontier, convergence curves, robustness bars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_frontier(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    x_metric: str = "critical_TTC_p95_mean",
    y_metric: str = "throughput_mean",
    label_col: str = "name",
) -> Path:
    """Plot Pareto frontier: x_metric vs y_metric (e.g. critical TTC p95 vs throughput)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if output_path is None:
        output_path = Path(csv_path).parent / "pareto_frontier.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if x_metric not in df.columns or y_metric not in df.columns:
        return output_path
    fig, ax = plt.subplots()
    labels = (
        df[label_col].astype(str)
        if label_col in df.columns
        else [str(i) for i in range(len(df))]
    )
    ax.scatter(df[x_metric], df[y_metric], s=50, alpha=0.8)
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl, (df[x_metric].iloc[i], df[y_metric].iloc[i]), fontsize=8, alpha=0.9
        )
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title("Pareto frontier (critical TTC p95 vs throughput)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_convergence(
    convergence_json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Plot best feasible fitness per generation from convergence JSON."""
    with open(convergence_json_path) as f:
        data = json.load(f)
    best_fitness = data.get("best_fitness_per_gen", [])
    feasible = data.get("feasible_per_gen", [])
    if output_path is None:
        output_path = Path(convergence_json_path).parent / "convergence.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    gens = list(range(len(best_fitness)))
    ax.plot(gens, best_fitness, marker="o", markersize=4, label="best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title("Convergence: best feasible fitness per generation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_robustness_bars(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    metric: str = "throughput_mean",
) -> Path:
    """Bar chart: mechanism x shift, height = metric."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if output_path is None:
        output_path = Path(csv_path).parent / "robustness_bars.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        "mechanism_name" not in df.columns
        or "shift" not in df.columns
        or metric not in df.columns
    ):
        return output_path
    mechanisms = df["mechanism_name"].unique().tolist()
    shifts = df["shift"].unique().tolist()
    pivot = df.pivot(index="mechanism_name", columns="shift", values=metric)

    fig, ax = plt.subplots(figsize=(max(8, len(shifts) * 1.5), 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metric)
    ax.set_title(f"Robustness: {metric} by mechanism and shift")
    ax.legend(title="Shift", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_ablation_bars(
    csv_path: str | Path,
    output_path: str | Path | None = None,
    metrics: list[str] | None = None,
) -> Path:
    """Bar chart: variant x metric(s)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if output_path is None:
        output_path = Path(csv_path).parent / "ablation_bars.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or ["throughput_mean", "critical_TTC_p95_mean", "fitness"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        if m in df.columns:
            df.plot(x="variant", y=m, kind="bar", ax=ax, legend=False)
            ax.set_ylabel(m)
            ax.set_title(m)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle("Ablation: performance by variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def create_convergence_plots(results_dir: str | Path) -> None:
    """Create convergence plots for the convergence suite."""
    results_dir = Path(results_dir)

    # Find convergence JSON files
    convergence_files = list(results_dir.glob("convergence*.json"))

    if not convergence_files:
        print("No convergence files found for plotting")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Read all convergence data
        all_curves = []
        for conv_file in convergence_files:
            with open(conv_file, "r") as f:
                data = json.load(f)

            if "convergence_analysis" in data:
                conv_analysis = data["convergence_analysis"]
                if "fitness_curves" in conv_analysis:
                    all_curves.extend(conv_analysis["fitness_curves"])

        if not all_curves:
            print("No fitness curves found in convergence data")
            return

        # Create multi-run convergence plot
        plt.figure(figsize=(10, 6))
        for i, curve in enumerate(all_curves):
            plt.plot(curve, alpha=0.7, label=f"Run {i + 1}")

        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Convergence Across Multiple Runs")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_file = results_dir / "convergence_multi_run.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Multi-run convergence plot saved to {plot_file}")

    except ImportError:
        print("Matplotlib not available for convergence plotting")
    except Exception as e:
        print(f"Error creating convergence plots: {e}")


def generate_all_plots(results_dir: str | Path = "results") -> None:
    """Generate all plots from results dir if corresponding CSVs/JSONs exist."""
    results_dir = Path(results_dir)
    if (results_dir / "baselines_results.csv").exists():
        plot_pareto_frontier(
            results_dir / "baselines_results.csv",
            results_dir / "pareto_frontier.png",
            label_col="name",
        )
        print("Saved pareto_frontier.png")
    for j in (results_dir / "evolution").glob("convergence_seed_*.json"):
        plot_convergence(j, j.parent / f"convergence_{j.stem}.png")
        print(f"Saved convergence plot for {j.name}")

    # Create multi-run convergence plots
    create_convergence_plots(results_dir)

    if (results_dir / "robustness_results.csv").exists():
        plot_robustness_bars(
            results_dir / "robustness_results.csv",
            results_dir / "robustness_bars.png",
        )
        print("Saved robustness_bars.png")
    if (results_dir / "ablation_results.csv").exists():
        plot_ablation_bars(
            results_dir / "ablation_results.csv",
            results_dir / "ablation_bars.png",
        )
        print("Saved ablation_bars.png")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()
    generate_all_plots(results_dir=args.results_dir)
