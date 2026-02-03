"""Robustness evaluation under distribution shifts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.run_episodes import evaluate_mechanism, load_config
from baselines.definitions import BASELINES


def run_robustness_suite(
    mechanisms: List[Dict[str, Any]],
    base_params: Dict[str, Any],
    episodes_per_shift: int = 200,
    base_seed: int = 0,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Evaluate mechanisms under various distribution shifts.

    Args:
        mechanisms: List of mechanisms to test (including baselines)
        base_params: Base simulation parameters
        episodes_per_shift: Episodes to run per shift condition
        base_seed: Base random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with robustness analysis
    """
    print(f"Starting robustness suite with {len(mechanisms)} mechanisms")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define shift scenarios
    shift_scenarios = [
        {"name": "nominal", "description": "Original parameters", "params": {}},
        {
            "name": "lambda_up_25",
            "description": "Arrival rate +25%",
            "params": {"lambda": base_params.get("lambda", 4.0) * 1.25},
        },
        {
            "name": "lambda_down_25",
            "description": "Arrival rate -25%",
            "params": {"lambda": base_params.get("lambda", 4.0) * 0.75},
        },
        {
            "name": "heavier_service_tail",
            "description": "Heavier-tailed service times",
            "params": {
                "service": {"default": {"dist": "lognormal", "mean": 25, "std": 20}}
            },
        },
        {
            "name": "more_high_risk",
            "description": "More high-risk patients",
            "params": {"risk_mix": {"critical": 0.2, "urgent": 0.4, "low": 0.4}},
        },
        {
            "name": "reduced_patience",
            "description": "Reduced patient patience",
            "params": {"patience": {"critical": 60, "urgent": 30, "low": 15}},
        },
    ]

    # Evaluate each mechanism under each shift
    mechanism_results = {}

    for mech_idx, mechanism in enumerate(mechanisms):
        print(f"Testing mechanism {mech_idx + 1}/{len(mechanisms)}")

        mechanism_id = mechanism.get("meta", {}).get("id", f"mech_{mech_idx}")
        mechanism_results[mechanism_id] = {"mechanism": mechanism, "shift_results": {}}

        for scenario in shift_scenarios:
            print(f"  Running scenario: {scenario['name']}")

            # Create shifted parameters
            shifted_params = dict(base_params)
            shifted_params.update(scenario["params"])

            try:
                # Evaluate mechanism under shifted conditions
                eval_result = evaluate_mechanism(
                    mechanism=mechanism,
                    params=shifted_params,
                    K=episodes_per_shift,
                    base_seed=base_seed + mech_idx * 100 + len(shift_scenarios) * 10,
                    fitness_weights=base_params.get("fitness_weights"),
                )

                mechanism_results[mechanism_id]["shift_results"][scenario["name"]] = {
                    "scenario": scenario,
                    "evaluation": eval_result,
                    "success": True,
                }

            except Exception as e:
                print(f"    Error in {scenario['name']}: {e}")
                mechanism_results[mechanism_id]["shift_results"][scenario["name"]] = {
                    "scenario": scenario,
                    "error": str(e),
                    "success": False,
                }

    # Analyze robustness
    robustness_analysis = analyze_robustness(mechanism_results)

    # Save results
    suite_results = {
        "config": {
            "mechanisms_count": len(mechanisms),
            "episodes_per_shift": episodes_per_shift,
            "base_seed": base_seed,
            "shift_scenarios": [s["name"] for s in shift_scenarios],
        },
        "mechanism_results": mechanism_results,
        "robustness_analysis": robustness_analysis,
    }

    results_file = results_dir / "robustness_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(suite_results, f, indent=2)

    print(f"\n=== Robustness Suite Completed ===")
    print(f"Results saved to {results_file}")

    return suite_results


def analyze_robustness(mechanism_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze robustness of mechanisms across shifts."""

    robustness_scores = {}
    mechanism_rankings = {}

    # Key metrics to track for robustness
    key_metrics = [
        "throughput",
        "critical_TTC_p95",
        "adverse_events_rate",
        "missed_critical_rate",
        "overload_time",
    ]

    for mechanism_id, results in mechanism_results.items():
        shift_results = results["shift_results"]

        # Skip if no successful evaluations
        successful_shifts = [
            name
            for name, result in shift_results.items()
            if result.get("success", False)
        ]

        if len(successful_shifts) < 3:  # Need minimum data for analysis
            robustness_scores[mechanism_id] = -1.0  # Poor robustness
            continue

        # Calculate performance deltas relative to nominal
        nominal_result = shift_results.get("nominal", {})
        if not nominal_result.get("success", False):
            robustness_scores[mechanism_id] = -1.0
            continue

        nominal_metrics = nominal_result["evaluation"]["mean_metrics"]

        # Calculate relative performance changes
        performance_changes = {}
        for metric in key_metrics:
            if metric not in nominal_metrics:
                continue

            nominal_value = nominal_metrics[metric]
            changes = []

            for shift_name in successful_shifts:
                if shift_name == "nominal":
                    continue

                shift_result = shift_results[shift_name]["evaluation"]["mean_metrics"]
                if metric in shift_result:
                    shift_value = shift_result[metric]
                    # Relative change (positive = worse for metrics we want to minimize)
                    if metric in [
                        "critical_TTC_p95",
                        "adverse_events_rate",
                        "missed_critical_rate",
                        "overload_time",
                    ]:
                        change = (shift_value - nominal_value) / max(
                            abs(nominal_value), 0.001
                        )
                    else:  # throughput (higher is better)
                        change = (nominal_value - shift_value) / max(
                            abs(nominal_value), 0.001
                        )
                    changes.append(change)

            if changes:
                performance_changes[metric] = {
                    "mean_change": sum(changes) / len(changes),
                    "max_change": max(changes),
                    "min_change": min(changes),
                    "std_change": (
                        sum((c - sum(changes) / len(changes)) ** 2 for c in changes)
                        / len(changes)
                    )
                    ** 0.5,
                }

        # Calculate overall robustness score (lower is better)
        # Penalize large negative impacts on key constraints and objectives
        robustness_score = 0.0

        # Heavy penalty for constraint violations
        if "missed_critical_rate" in performance_changes:
            robustness_score += (
                performance_changes["missed_critical_rate"]["mean_change"] * 10
            )

        if "critical_TTC_p95" in performance_changes:
            robustness_score += (
                performance_changes["critical_TTC_p95"]["mean_change"] * 5
            )

        # Moderate penalty for performance degradation
        if "throughput" in performance_changes:
            robustness_score += performance_changes["throughput"]["mean_change"] * 2

        if "adverse_events_rate" in performance_changes:
            robustness_score += (
                performance_changes["adverse_events_rate"]["mean_change"] * 3
            )

        if "overload_time" in performance_changes:
            robustness_score += performance_changes["overload_time"]["mean_change"] * 1

        robustness_scores[mechanism_id] = robustness_score
        mechanism_rankings[mechanism_id] = performance_changes

    # Rank mechanisms by robustness
    sorted_by_robustness = sorted(robustness_scores.items(), key=lambda x: x[1])

    return {
        "robustness_scores": robustness_scores,
        "mechanism_rankings": mechanism_rankings,
        "sorted_mechanisms": sorted_by_robustness,
        "most_robust": sorted_by_robustness[0] if sorted_by_robustness else None,
        "least_robust": sorted_by_robustness[-1] if sorted_by_robustness else None,
    }


def create_robustness_comparison_table(
    mechanism_results: Dict[str, Any], output_path: Path
) -> None:
    """Create a comparison table of mechanism performance across shifts."""

    import csv

    # Prepare CSV data
    csv_data = []

    for mechanism_id, results in mechanism_results.items():
        base_row = {"mechanism_id": mechanism_id}

        # Add nominal performance as baseline
        nominal_result = results["shift_results"].get("nominal", {})
        if nominal_result.get("success", False):
            nominal_metrics = nominal_result["evaluation"]["mean_metrics"]
            for metric, value in nominal_metrics.items():
                base_row[f"nominal_{metric}"] = value

        # Add shift performance deltas
        for shift_name, shift_result in results["shift_results"].items():
            if shift_name == "nominal" or not shift_result.get("success", False):
                continue

            shift_metrics = shift_result["evaluation"]["mean_metrics"]

            for metric in [
                "throughput",
                "critical_TTC_p95",
                "adverse_events_rate",
                "missed_critical_rate",
            ]:
                if metric in nominal_metrics and metric in shift_metrics:
                    nominal_val = nominal_metrics[metric]
                    shift_val = shift_metrics[metric]

                    if metric in [
                        "critical_TTC_p95",
                        "adverse_events_rate",
                        "missed_critical_rate",
                    ]:
                        delta_pct = (
                            (shift_val - nominal_val) / max(abs(nominal_val), 0.001)
                        ) * 100
                    else:  # throughput
                        delta_pct = (
                            (nominal_val - shift_val) / max(abs(nominal_val), 0.001)
                        ) * 100

                    base_row[f"{shift_name}_{metric}_delta_pct"] = round(delta_pct, 2)

        csv_data.append(base_row)

    # Write CSV
    if csv_data:
        fieldnames = list(csv_data[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"Robustness comparison table saved to {output_path}")


def evaluate_baselines_and_evolved_mechanisms(
    evolution_results_path: str,
    config_path: Optional[str] = None,
    results_dir: str = "results",
    top_k_evolved: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate top evolved mechanisms and all baselines under robustness testing.

    Args:
        evolution_results_path: Path to evolution results JSON
        config_path: Configuration file path
        results_dir: Results directory
        top_k_evolved: Number of top evolved mechanisms to test

    Returns:
        Robustness evaluation results
    """
    # Load evolution results to get top mechanisms
    with open(evolution_results_path, "r") as f:
        evolution_data = json.load(f)

    # Extract top evolved mechanisms
    best_mechanism = evolution_data.get("best_mechanism")
    evolved_mechanisms = [best_mechanism] if best_mechanism else []

    # For now, just use the best mechanism (could extend to get top-K)
    if len(evolved_mechanisms) < top_k_evolved:
        print(
            f"Warning: Only {len(evolved_mechanisms)} evolved mechanisms found, requested {top_k_evolved}"
        )

    # Combine with baselines
    all_mechanisms = evolved_mechanisms + BASELINES

    # Load configuration
    params = load_config(config_path)

    # Run robustness suite
    return run_robustness_suite(
        mechanisms=all_mechanisms,
        base_params=params,
        episodes_per_shift=200,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evolution_results", type=str, help="Path to evolution results JSON"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--results", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--top_evolved",
        type=int,
        default=5,
        help="Number of top evolved mechanisms to test",
    )

    args = parser.parse_args()

    if args.evolution_results:
        # Evaluate evolved mechanisms + baselines
        evaluate_baselines_and_evolved_mechanisms(
            evolution_results_path=args.evolution_results,
            config_path=args.config,
            results_dir=args.results,
            top_k_evolved=args.top_evolved,
        )
    else:
        print("Please provide --evolution_results to run robustness evaluation")
