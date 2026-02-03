"""Pareto frontier computation and dominance analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval.metrics import pareto_metrics


def compute_pareto_frontier(
    mechanisms: List[Dict[str, Any]],
    metrics_list: List[Dict[str, Any]],
    constraints_list: List[Dict[str, bool]],
) -> Dict[str, Any]:
    """
    Compute Pareto frontier on objectives: minimize adverse_events, critical_TTC_p95, overload_time, maximize throughput.

    Args:
        mechanisms: List of mechanism dictionaries
        metrics_list: Corresponding metrics for each mechanism
        constraints_list: Corresponding constraint violations for each mechanism

    Returns:
        Dictionary with Pareto frontier analysis
    """
    if not mechanisms or len(mechanisms) != len(metrics_list):
        return {"error": "Invalid input: mechanisms and metrics length mismatch"}

    # Extract objective vectors
    objective_vectors = []
    for i, (mechanism, metrics, constraints) in enumerate(
        zip(mechanisms, metrics_list, constraints_list)
    ):
        # Get Pareto vector (safety_indicator, critical_TTC_p95, overload_time, throughput)
        safety, ttc_p95, overload, throughput = pareto_metrics(metrics, constraints)

        objective_vectors.append(
            {
                "index": i,
                "mechanism": mechanism,
                "metrics": metrics,
                "constraints": constraints,
                "objectives": {
                    "safety_indicator": safety,  # minimize
                    "critical_TTC_p95": ttc_p95,  # minimize
                    "overload_time": overload,  # minimize
                    "throughput": throughput,  # maximize
                },
                "feasible": all(not v for v in constraints.values())
                if constraints
                else False,
            }
        )

    # Find Pareto-optimal mechanisms
    pareto_indices = find_pareto_optimal_indices(objective_vectors)
    pareto_mechanisms = [objective_vectors[i] for i in pareto_indices]

    # Sort Pareto frontier by one objective (e.g., throughput)
    pareto_mechanisms.sort(key=lambda x: x["objectives"]["throughput"], reverse=True)

    return {
        "total_mechanisms": len(mechanisms),
        "pareto_frontier_size": len(pareto_mechanisms),
        "pareto_mechanisms": pareto_mechanisms,
        "pareto_indices": pareto_indices,
        "dominance_analysis": analyze_dominance(objective_vectors, pareto_indices),
    }


def find_pareto_optimal_indices(objective_vectors: List[Dict[str, Any]]) -> List[int]:
    """
    Find indices of Pareto-optimal mechanisms.

    A mechanism i is Pareto-optimal if no other mechanism j dominates it.
    Mechanism j dominates i if:
    - j is better or equal on all objectives
    - j is strictly better on at least one objective
    """
    pareto_indices = []
    n = len(objective_vectors)

    for i in range(n):
        dominated = False
        obj_i = objective_vectors[i]["objectives"]

        for j in range(n):
            if i == j:
                continue

            obj_j = objective_vectors[j]["objectives"]

            # Check if j dominates i
            better_on_safety = (
                obj_j["safety_indicator"] < obj_i["safety_indicator"]
            )  # lower is better
            better_on_ttc = (
                obj_j["critical_TTC_p95"] < obj_i["critical_TTC_p95"]
            )  # lower is better
            better_on_overload = (
                obj_j["overload_time"] < obj_i["overload_time"]
            )  # lower is better
            better_on_throughput = (
                obj_j["throughput"] > obj_i["throughput"]
            )  # higher is better

            # j dominates i if it's better or equal on all and strictly better on at least one
            if (
                (
                    better_on_safety
                    or obj_j["safety_indicator"] <= obj_i["safety_indicator"]
                )
                and (
                    better_on_ttc
                    or obj_j["critical_TTC_p95"] <= obj_i["critical_TTC_p95"]
                )
                and (
                    better_on_overload
                    or obj_j["overload_time"] <= obj_i["overload_time"]
                )
                and (better_on_throughput or obj_j["throughput"] >= obj_i["throughput"])
                and (
                    better_on_safety
                    or better_on_ttc
                    or better_on_overload
                    or better_on_throughput
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto_indices.append(i)

    return pareto_indices


def analyze_dominance(
    objective_vectors: List[Dict[str, Any]], pareto_indices: List[int]
) -> Dict[str, Any]:
    """Analyze dominance relationships and provide insights."""

    n_total = len(objective_vectors)
    n_pareto = len(pareto_indices)
    n_dominated = n_total - n_pareto

    # Count how many mechanisms each Pareto-optimal mechanism dominates
    domination_counts = {i: 0 for i in pareto_indices}

    for i in range(n_total):
        if i in pareto_indices:
            continue

        obj_i = objective_vectors[i]["objectives"]

        for j in pareto_indices:
            obj_j = objective_vectors[j]["objectives"]

            # Check if Pareto mechanism j dominates i
            better_or_equal_all = (
                obj_j["safety_indicator"] <= obj_i["safety_indicator"]
                and obj_j["critical_TTC_p95"] <= obj_i["critical_TTC_p95"]
                and obj_j["overload_time"] <= obj_i["overload_time"]
                and obj_j["throughput"] >= obj_i["throughput"]
            )

            strictly_better_some = (
                obj_j["safety_indicator"] < obj_i["safety_indicator"]
                or obj_j["critical_TTC_p95"] < obj_i["critical_TTC_p95"]
                or obj_j["overload_time"] < obj_i["overload_time"]
                or obj_j["throughput"] > obj_i["throughput"]
            )

            if better_or_equal_all and strictly_better_some:
                domination_counts[j] += 1

    # Find most "powerful" Pareto mechanisms
    sorted_by_dominance = sorted(
        domination_counts.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "total_mechanisms": n_total,
        "pareto_optimal": n_pareto,
        "dominated": n_dominated,
        "pareto_percentage": n_pareto / n_total if n_total > 0 else 0,
        "domination_counts": domination_counts,
        "most_dominating_pareto": sorted_by_dominance[:5]
        if sorted_by_dominance
        else [],
    }


def compare_with_baselines(
    pareto_results: Dict[str, Any], baseline_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare Pareto-optimal mechanisms against baseline mechanisms.

    Args:
        pareto_results: Results from compute_pareto_frontier for evolved mechanisms
        baseline_results: List of baseline evaluation results

    Returns:
        Comparison analysis
    """
    if "pareto_mechanisms" not in pareto_results:
        return {"error": "Invalid Pareto results"}

    pareto_mechanisms = pareto_results["pareto_mechanisms"]

    # Check if any Pareto mechanism dominates baselines
    dominating_count = 0
    dominated_baselines = []
    baseline_comparisons = []

    for pareto_mech in pareto_mechanisms:
        pareto_obj = pareto_mech["objectives"]
        pareto_feasible = pareto_mech["feasible"]

        for baseline_idx, baseline in enumerate(baseline_results):
            # Extract baseline objectives (approximate)
            baseline_obj = {
                "safety_indicator": 0.0 if baseline.get("feasible", True) else 1000.0,
                "critical_TTC_p95": baseline.get("critical_TTC_p95_mean", 0),
                "overload_time": baseline.get("overload_time_mean", 0),
                "throughput": baseline.get("throughput_mean", 0),
            }

            # Check if Pareto mechanism dominates baseline
            better_or_equal_all = (
                pareto_obj["safety_indicator"] <= baseline_obj["safety_indicator"]
                and pareto_obj["critical_TTC_p95"] <= baseline_obj["critical_TTC_p95"]
                and pareto_obj["overload_time"] <= baseline_obj["overload_time"]
                and pareto_obj["throughput"] >= baseline_obj["throughput"]
            )

            strictly_better_some = (
                pareto_obj["safety_indicator"] < baseline_obj["safety_indicator"]
                or pareto_obj["critical_TTC_p95"] < baseline_obj["critical_TTC_p95"]
                or pareto_obj["overload_time"] < baseline_obj["overload_time"]
                or pareto_obj["throughput"] > baseline_obj["throughput"]
            )

            dominates = better_or_equal_all and strictly_better_some and pareto_feasible

            baseline_comparisons.append(
                {
                    "pareto_index": pareto_mech["index"],
                    "baseline_index": baseline_idx,
                    "dominates": dominates,
                    "pareto_feasible": pareto_feasible,
                    "baseline_feasible": baseline.get("feasible", True),
                }
            )

            if dominates and baseline_idx not in dominated_baselines:
                dominated_baselines.append(baseline_idx)
                dominating_count += 1

    return {
        "pareto_mechanisms_count": len(pareto_mechanisms),
        "baseline_mechanisms_count": len(baseline_results),
        "dominated_baselines": len(dominated_baselines),
        "dominated_baselines_indices": dominated_baselines,
        "domination_percentage": len(dominated_baselines) / len(baseline_results)
        if baseline_results
        else 0,
        "detailed_comparisons": baseline_comparisons,
    }


def create_pareto_plot(pareto_results: Dict[str, Any], output_path: Path) -> None:
    """Create Pareto frontier visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if "pareto_mechanisms" not in pareto_results:
            return

        pareto_mechs = pareto_results["pareto_mechanisms"]

        if not pareto_mechs:
            return

        # Extract objectives
        safety = [m["objectives"]["safety_indicator"] for m in pareto_mechs]
        ttc_p95 = [m["objectives"]["critical_TTC_p95"] for m in pareto_mechs]
        overload = [m["objectives"]["overload_time"] for m in pareto_mechs]
        throughput = [m["objectives"]["throughput"] for m in pareto_mechs]
        feasible = [m["feasible"] for m in pareto_mechs]

        # Create 2x2 subplot for different objective pairs
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Throughput vs TTC_p95
        feasible_mask = [f for f in feasible]
        infeasible_mask = [not f for f in feasible]

        ax1.scatter(
            [throughput[i] for i in range(len(throughput)) if feasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if feasible_mask[i]],
            c="green",
            label="Feasible",
            alpha=0.7,
        )
        ax1.scatter(
            [throughput[i] for i in range(len(throughput)) if infeasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if infeasible_mask[i]],
            c="red",
            label="Infeasible",
            alpha=0.7,
        )
        ax1.set_xlabel("Throughput")
        ax1.set_ylabel("Critical TTC p95")
        ax1.set_title("Throughput vs Critical TTC")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Throughput vs Safety
        ax2.scatter(
            [throughput[i] for i in range(len(throughput)) if feasible_mask[i]],
            [safety[i] for i in range(len(safety)) if feasible_mask[i]],
            c="green",
            label="Feasible",
            alpha=0.7,
        )
        ax2.scatter(
            [throughput[i] for i in range(len(throughput)) if infeasible_mask[i]],
            [safety[i] for i in range(len(safety)) if infeasible_mask[i]],
            c="red",
            label="Infeasible",
            alpha=0.7,
        )
        ax2.set_xlabel("Throughput")
        ax2.set_ylabel("Safety Indicator")
        ax2.set_title("Throughput vs Safety")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Safety vs TTC_p95
        ax3.scatter(
            [safety[i] for i in range(len(safety)) if feasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if feasible_mask[i]],
            c="green",
            label="Feasible",
            alpha=0.7,
        )
        ax3.scatter(
            [safety[i] for i in range(len(safety)) if infeasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if infeasible_mask[i]],
            c="red",
            label="Infeasible",
            alpha=0.7,
        )
        ax3.set_xlabel("Safety Indicator")
        ax3.set_ylabel("Critical TTC p95")
        ax3.set_title("Safety vs Critical TTC")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Overload vs TTC_p95
        ax4.scatter(
            [overload[i] for i in range(len(overload)) if feasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if feasible_mask[i]],
            c="green",
            label="Feasible",
            alpha=0.7,
        )
        ax4.scatter(
            [overload[i] for i in range(len(overload)) if infeasible_mask[i]],
            [ttc_p95[i] for i in range(len(ttc_p95)) if infeasible_mask[i]],
            c="red",
            label="Infeasible",
            alpha=0.7,
        )
        ax4.set_xlabel("Overload Time")
        ax4.set_ylabel("Critical TTC p95")
        ax4.set_title("Overload vs Critical TTC")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Pareto frontier plot saved to {output_path}")

    except ImportError:
        print("Matplotlib not available for plotting")


def save_pareto_results(pareto_results: Dict[str, Any], output_path: Path) -> None:
    """Save Pareto frontier results to JSON file."""

    # Make results more serializable
    serializable_results = dict(pareto_results)

    # Handle any non-serializable objects
    if "pareto_mechanisms" in serializable_results:
        for mech in serializable_results["pareto_mechanisms"]:
            # Ensure mechanism is serializable
            if "mechanism" in mech:
                # This should already be JSON-serializable, but double-check
                pass

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Pareto frontier results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanisms_file", type=str, required=True, help="JSON file with mechanisms"
    )
    parser.add_argument(
        "--metrics_file", type=str, required=True, help="JSON file with metrics"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )

    args = parser.parse_args()

    # Load data
    with open(args.mechanisms_file, "r") as f:
        mechanisms = json.load(f)

    with open(args.metrics_file, "r") as f:
        metrics_data = json.load(f)

    # Extract metrics and constraints
    metrics_list = metrics_data.get("metrics", [])
    constraints_list = metrics_data.get("constraints", [])

    # Compute Pareto frontier
    results = compute_pareto_frontier(mechanisms, metrics_list, constraints_list)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pareto_results(results, output_dir / "pareto_frontier.json")
    create_pareto_plot(results, output_dir / "pareto_frontier.png")
