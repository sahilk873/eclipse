"""Ablations study to analyze component importance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.run_episodes import evaluate_mechanism, load_config
from mechanisms.schema import create_mechanism_id, validate_mechanism


def run_ablation_study(
    best_mechanism: Dict[str, Any],
    base_params: Dict[str, Any],
    episodes_per_ablation: int = 200,
    base_seed: int = 0,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Run ablation study on best mechanism by removing/changing each component.

    Args:
        best_mechanism: The best mechanism found by evolution
        base_params: Base simulation parameters
        episodes_per_ablation: Episodes to run for each ablated version
        base_seed: Base random seed
        results_dir: Directory to save results

    Returns:
        Dictionary with ablation study results
    """
    print(f"Starting ablation study on best mechanism")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # First, evaluate the original mechanism
    print("Evaluating original mechanism...")
    original_result = evaluate_mechanism(
        mechanism=best_mechanism,
        params=base_params,
        K=episodes_per_ablation,
        base_seed=base_seed,
        fitness_weights=base_params.get("fitness_weights"),
    )

    # Generate ablated mechanisms
    ablation_variants = generate_ablation_variants(best_mechanism)

    # Evaluate each ablated variant
    ablation_results = {
        "original": {
            "mechanism": best_mechanism,
            "evaluation": original_result,
            "description": "Original best mechanism",
        }
    }

    for variant_name, (variant_mechanism, description) in ablation_variants.items():
        print(f"Evaluating ablation: {variant_name}")

        try:
            # Validate ablated mechanism
            valid, errors = validate_mechanism(variant_mechanism)
            if not valid:
                print(f"  Invalid ablated mechanism: {errors}")
                continue

            # Evaluate ablated version
            eval_result = evaluate_mechanism(
                mechanism=variant_mechanism,
                params=base_params,
                K=episodes_per_ablation,
                base_seed=base_seed + hash(variant_name) % 10000,
                fitness_weights=base_params.get("fitness_weights"),
            )

            ablation_results[variant_name] = {
                "mechanism": variant_mechanism,
                "evaluation": eval_result,
                "description": description,
            }

            # Calculate performance impact
            fitness_impact = eval_result["fitness"] - original_result["fitness"]
            print(f"  Fitness impact: {fitness_impact:+.4f}")

        except Exception as e:
            print(f"  Error evaluating {variant_name}: {e}")
            ablation_results[variant_name] = {
                "mechanism": variant_mechanism,
                "error": str(e),
                "description": description,
            }

    # Analyze component importance
    component_analysis = analyze_component_importance(ablation_results)

    # Save results
    study_results = {
        "config": {
            "episodes_per_ablation": episodes_per_ablation,
            "base_seed": base_seed,
            "ablation_variants": list(ablation_variants.keys()),
        },
        "original_mechanism": best_mechanism,
        "ablation_results": ablation_results,
        "component_analysis": component_analysis,
    }

    results_file = results_dir / "ablation_study_results.json"
    with open(results_file, "w") as f:
        json.dump(study_results, f, indent=2)

    print(f"\n=== Ablation Study Completed ===")
    print(f"Results saved to {results_file}")

    return study_results


def generate_ablation_variants(
    mechanism: Dict[str, Any],
) -> Dict[str, Tuple[Dict[str, Any], str]]:
    """
    Generate ablated versions of the mechanism by systematically changing components.

    Returns:
        Dictionary mapping variant_name to (ablated_mechanism, description)
    """
    variants = {}
    base_mechanism = copy.deepcopy(mechanism)

    # Ensure we have the expected structure
    info_policy = base_mechanism.get("info_policy", {})
    service_policy = base_mechanism.get("service_policy", {})
    redirect_policy = base_mechanism.get("redirect_exit_policy", {})
    meta = base_mechanism.get("meta", {})

    # 1. Remove information (set to none)
    variant1 = copy.deepcopy(base_mechanism)
    variant1["info_policy"]["info_mode"] = "none"
    if "bins" in variant1["info_policy"]:
        del variant1["info_policy"]["bins"]
    variant1["meta"]["id"] = create_mechanism_id()
    variant1["meta"]["seed_tag"] = "ablation_no_info"
    variants["no_info"] = (variant1, "No information provided to patients")

    # 2. Use FIFO service (remove prioritization)
    variant2 = copy.deepcopy(base_mechanism)
    variant2["service_policy"]["service_rule"] = "fifo"
    if "params" in variant2["service_policy"]:
        del variant2["service_policy"]["params"]
    variant2["meta"]["id"] = create_mechanism_id()
    variant2["meta"]["seed_tag"] = "ablation_fifo"
    variants["fifo_service"] = (variant2, "FIFO service rule (no prioritization)")

    # 3. No redirect policy
    variant3 = copy.deepcopy(base_mechanism)
    variant3["redirect_exit_policy"]["redirect_low_risk"] = False
    variant3["redirect_exit_policy"]["redirect_mode"] = "none"
    if "params" in variant3["redirect_exit_policy"]:
        del variant3["redirect_exit_policy"]["params"]
    variant3["meta"]["id"] = create_mechanism_id()
    variant3["meta"]["seed_tag"] = "ablation_no_redirect"
    variants["no_redirect"] = (variant3, "No patient redirection")

    # 4. No reneging (perfect patience)
    variant4 = copy.deepcopy(base_mechanism)
    variant4["redirect_exit_policy"]["reneging_enabled"] = False
    variant4["meta"]["id"] = create_mechanism_id()
    variant4["meta"]["seed_tag"] = "ablation_no_reneging"
    variants["no_reneging"] = (variant4, "No patient reneging")

    # 5. Exact information (if not already)
    if info_policy.get("info_mode") != "exact":
        variant5 = copy.deepcopy(base_mechanism)
        variant5["info_policy"]["info_mode"] = "exact"
        if "bins" in variant5["info_policy"]:
            del variant5["info_policy"]["bins"]
        variant5["meta"]["id"] = create_mechanism_id()
        variant5["meta"]["seed_tag"] = "ablation_exact_info"
        variants["exact_info"] = (variant5, "Exact wait time information")

    # 6. Severity priority service (if not already)
    if service_policy.get("service_rule") != "severity_priority":
        variant6 = copy.deepcopy(base_mechanism)
        variant6["service_policy"]["service_rule"] = "severity_priority"
        if "params" in variant6["service_policy"]:
            del variant6["service_policy"]["params"]
        variant6["meta"]["id"] = create_mechanism_id()
        variant6["meta"]["seed_tag"] = "ablation_severity_priority"
        variants["severity_priority"] = (variant6, "Severity-based prioritization")

    # 7. Combined redirect (if not already)
    if redirect_policy.get("redirect_mode") != "combined":
        variant7 = copy.deepcopy(base_mechanism)
        variant7["redirect_exit_policy"]["redirect_low_risk"] = True
        variant7["redirect_exit_policy"]["redirect_mode"] = "combined"
        variant7["redirect_exit_policy"]["params"] = {
            "risk_threshold": 0.3,
            "congestion_threshold": 15,
        }
        variant7["meta"]["id"] = create_mechanism_id()
        variant7["meta"]["seed_tag"] = "ablation_combined_redirect"
        variants["combined_redirect"] = (
            variant7,
            "Combined risk and congestion-based redirection",
        )

    return variants


def analyze_component_importance(ablation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the importance of each component based on fitness impact.

    Args:
        ablation_results: Results from ablation study

    Returns:
        Component importance analysis
    """
    original_result = ablation_results.get("original", {}).get("evaluation", {})
    original_fitness = original_result.get("fitness", 0.0)
    original_metrics = original_result.get("mean_metrics", {})

    component_importance = {}
    component_effects = {}

    for variant_name, variant_data in ablation_results.items():
        if variant_name == "original" or "error" in variant_data:
            continue

        eval_result = variant_data.get("evaluation", {})
        if not eval_result:
            continue

        variant_fitness = eval_result.get("fitness", 0.0)
        variant_metrics = eval_result.get("mean_metrics", {})

        # Calculate fitness impact
        fitness_impact = variant_fitness - original_fitness

        # Calculate metric impacts
        metric_impacts = {}
        for metric in [
            "throughput",
            "critical_TTC_p95",
            "adverse_events_rate",
            "missed_critical_rate",
            "overload_time",
        ]:
            if metric in original_metrics and metric in variant_metrics:
                original_val = original_metrics[metric]
                variant_val = variant_metrics[metric]

                if metric in [
                    "critical_TTC_p95",
                    "adverse_events_rate",
                    "missed_critical_rate",
                    "overload_time",
                ]:
                    # Lower is better, so positive impact = reduction
                    impact = original_val - variant_val
                else:  # throughput
                    # Higher is better, so positive impact = increase
                    impact = variant_val - original_val

                metric_impacts[metric] = impact

        component_importance[variant_name] = {
            "fitness_impact": fitness_impact,
            "metric_impacts": metric_impacts,
            "description": variant_data.get("description", ""),
            "relative_fitness": variant_fitness / max(abs(original_fitness), 0.001),
        }

        component_effects[variant_name] = metric_impacts

    # Rank components by importance (absolute fitness impact)
    ranked_components = sorted(
        component_importance.items(),
        key=lambda x: abs(x[1]["fitness_impact"]),
        reverse=True,
    )

    # Categorize components
    critical_components = [
        name
        for name, data in component_importance.items()
        if abs(data["fitness_impact"]) > 0.1
    ]
    moderate_components = [
        name
        for name, data in component_importance.items()
        if 0.05 <= abs(data["fitness_impact"]) <= 0.1
    ]
    minor_components = [
        name
        for name, data in component_importance.items()
        if abs(data["fitness_impact"]) < 0.05
    ]

    return {
        "component_importance": component_importance,
        "component_effects": component_effects,
        "ranked_components": ranked_components,
        "critical_components": critical_components,
        "moderate_components": moderate_components,
        "minor_components": minor_components,
        "most_important": ranked_components[0] if ranked_components else None,
        "least_important": ranked_components[-1] if ranked_components else None,
    }


def create_ablation_summary_table(
    ablation_results: Dict[str, Any], output_path: Path
) -> None:
    """Create a summary table of ablation results."""

    import csv

    original_result = ablation_results.get("original", {}).get("evaluation", {})
    original_fitness = original_result.get("fitness", 0.0)
    original_metrics = original_result.get("mean_metrics", {})

    csv_data = []

    for variant_name, variant_data in ablation_results.items():
        if "error" in variant_data:
            continue

        eval_result = variant_data.get("evaluation", {})
        if not eval_result:
            continue

        variant_fitness = eval_result.get("fitness", 0.0)
        variant_metrics = eval_result.get("mean_metrics", {})

        row = {
            "variant": variant_name,
            "description": variant_data.get("description", ""),
            "fitness": variant_fitness,
            "fitness_change": variant_fitness - original_fitness,
            "fitness_change_pct": (
                (variant_fitness - original_fitness) / max(abs(original_fitness), 0.001)
            )
            * 100,
        }

        # Add metric changes
        for metric in [
            "throughput",
            "critical_TTC_p95",
            "adverse_events_rate",
            "missed_critical_rate",
            "overload_time",
        ]:
            if metric in original_metrics and metric in variant_metrics:
                original_val = original_metrics[metric]
                variant_val = variant_metrics[metric]
                row[f"{metric}_original"] = original_val
                row[f"{metric}_variant"] = variant_val
                row[f"{metric}_change"] = variant_val - original_val
                row[f"{metric}_change_pct"] = (
                    (variant_val - original_val) / max(abs(original_val), 0.001)
                ) * 100

        csv_data.append(row)

    # Write CSV
    if csv_data:
        fieldnames = list(csv_data[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"Ablation summary table saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mechanism", type=str, required=True, help="JSON file with best mechanism"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--results", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--episodes", type=int, default=200, help="Episodes per ablation"
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed")

    args = parser.parse_args()

    # Load mechanism
    with open(args.mechanism, "r") as f:
        best_mechanism = json.load(f)

    # Load configuration
    params = load_config(args.config)

    # Run ablation study
    run_ablation_study(
        best_mechanism=best_mechanism,
        base_params=params,
        episodes_per_ablation=args.episodes,
        base_seed=args.seed,
        results_dir=args.results,
    )
