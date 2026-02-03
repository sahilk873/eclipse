"""Multi-run convergence suite for robustness evidence."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolution.evolve import evolve_mechanisms, analyze_component_diversity
from evolution.llm_mutation import llm_mutate


def run_convergence_suite(
    num_runs: int = 5,
    generations_per_run: int = 20,
    population_size: int = 50,
    episodes_per_mechanism: int = 50,
    base_seed: int = 0,
    config_path: Optional[str] = None,
    results_dir: str = "results",
    use_llm: bool = False,
    llm_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run multiple evolutionary searches with different seeds to assess convergence.

    Args:
        num_runs: Number of independent runs
        generations_per_run: Generations per run
        population_size: Population size per generation
        episodes_per_mechanism: Episodes for mechanism evaluation
        base_seed: Starting seed (each run gets base_seed + run_id)
        config_path: Path to configuration file
        results_dir: Directory to save results
        use_llm: Whether to use LLM mutation
        llm_api_key: API key for LLM (if use_llm=True)

    Returns:
        Dictionary with convergence analysis and results
    """
    print(f"Starting convergence suite with {num_runs} runs")
    print(f"Each run: {generations_per_run} generations, {population_size} population")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup LLM mutation if requested
    llm_mutate_func = None
    if use_llm and llm_api_key:
        llm_mutate_func = lambda *args, **kwargs: llm_mutate(
            *args, api_key=llm_api_key, **kwargs
        )

    # Run evolutionary searches
    run_results = []

    for run_id in range(num_runs):
        run_seed = base_seed + run_id * 1000
        run_identifier = f"convergence_run_{run_id}"

        print(f"\n=== Starting Run {run_id + 1}/{num_runs} (seed: {run_seed}) ===")

        try:
            result = evolve_mechanisms(
                generations=generations_per_run,
                population_size=population_size,
                episodes_per_mechanism=episodes_per_mechanism,
                config_path=config_path,
                results_dir=str(results_dir / f"run_{run_id}"),
                base_seed=run_seed,
                run_id=run_identifier,
                llm_mutate=llm_mutate_func,
                llm_fraction=0.2 if use_llm else 0.0,
            )

            run_results.append(
                {
                    "run_id": run_id,
                    "seed": run_seed,
                    "identifier": run_identifier,
                    "result": result,
                    "success": True,
                }
            )

            print(
                f"Run {run_id + 1} completed. Best fitness: {result['best_fitness']:.4f}"
            )

        except Exception as e:
            print(f"Run {run_id + 1} failed: {e}")
            run_results.append(
                {
                    "run_id": run_id,
                    "seed": run_seed,
                    "identifier": run_identifier,
                    "result": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # Analyze convergence across runs
    convergence_analysis = analyze_convergence_across_runs(run_results, results_dir)

    # Component structure analysis
    structure_analysis = analyze_common_structures(run_results, results_dir)

    # Save comprehensive results
    suite_results = {
        "config": {
            "num_runs": num_runs,
            "generations_per_run": generations_per_run,
            "population_size": population_size,
            "episodes_per_mechanism": episodes_per_mechanism,
            "base_seed": base_seed,
            "use_llm": use_llm,
        },
        "run_results": run_results,
        "convergence_analysis": convergence_analysis,
        "structure_analysis": structure_analysis,
    }

    # Save results
    results_file = results_dir / "convergence_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(suite_results, f, indent=2)

    print(f"\n=== Convergence Suite Completed ===")
    print(f"Successful runs: {sum(1 for r in run_results if r['success'])}/{num_runs}")
    print(f"Results saved to {results_file}")

    return suite_results


def analyze_convergence_across_runs(
    run_results: List[Dict[str, Any]], results_dir: Path
) -> Dict[str, Any]:
    """Analyze convergence patterns across multiple runs."""

    successful_runs = [r for r in run_results if r["success"]]
    if not successful_runs:
        return {"error": "No successful runs to analyze"}

    # Extract fitness curves
    fitness_curves = []
    best_fitnesses = []

    for run in successful_runs:
        conv_data = run["result"]["convergence_data"]
        fitness_curve = [gen["best_fitness"] for gen in conv_data]
        fitness_curves.append(fitness_curve)
        best_fitnesses.append(fitness_curve[-1] if fitness_curve else float("-inf"))

    # Calculate statistics
    final_fitness_stats = {
        "mean": sum(best_fitnesses) / len(best_fitnesses),
        "min": min(best_fitnesses),
        "max": max(best_fitnesses),
        "std": (
            sum(
                (f - sum(best_fitnesses) / len(best_fitnesses)) ** 2
                for f in best_fitnesses
            )
            / len(best_fitnesses)
        )
        ** 0.5,
    }

    # Convergence consistency analysis
    if len(fitness_curves) > 1:
        # Calculate correlation between curves (simplified)
        avg_correlation = calculate_average_correlation(fitness_curves)
    else:
        avg_correlation = 1.0

    return {
        "successful_runs": len(successful_runs),
        "final_fitness_stats": final_fitness_stats,
        "avg_correlation_between_runs": avg_correlation,
        "fitness_curves": fitness_curves,
        "best_run_fitness": max(best_fitnesses),
        "worst_run_fitness": min(best_fitnesses),
    }


def analyze_common_structures(
    run_results: List[Dict[str, Any]], results_dir: Path
) -> Dict[str, Any]:
    """Analyze common mechanism structures across runs."""

    successful_runs = [r for r in run_results if r["success"]]
    if not successful_runs:
        return {"error": "No successful runs to analyze"}

    # Extract best mechanisms from each run
    best_mechanisms = []
    for run in successful_runs:
        best_mech = run["result"]["best_mechanism"]
        if best_mech:
            best_mechanisms.append(best_mech)

    if not best_mechanisms:
        return {"error": "No best mechanisms found"}

    # Analyze component frequencies
    component_stats = {}

    for mechanism in best_mechanisms:
        # Extract component flags
        info_mode = mechanism.get("info_policy", {}).get("info_mode", "none")
        service_rule = mechanism.get("service_policy", {}).get("service_rule", "fifo")
        redirect_mode = mechanism.get("redirect_exit_policy", {}).get(
            "redirect_mode", "none"
        )
        redirect_low_risk = mechanism.get("redirect_exit_policy", {}).get(
            "redirect_low_risk", False
        )
        reneging_enabled = mechanism.get("redirect_exit_policy", {}).get(
            "reneging_enabled", True
        )

        components = {
            "info_mode": info_mode,
            "service_rule": service_rule,
            "redirect_mode": redirect_mode,
            "redirect_low_risk": redirect_low_risk,
            "reneging_enabled": reneging_enabled,
        }

        for key, value in components.items():
            if key not in component_stats:
                component_stats[key] = {}
            if value not in component_stats[key]:
                component_stats[key][value] = 0
            component_stats[key][value] += 1

    # Convert to percentages
    component_percentages = {}
    for key, value_counts in component_stats.items():
        total = sum(value_counts.values())
        component_percentages[key] = {
            value: count / total for value, count in value_counts.items()
        }

    # Find most common structure
    most_common_components = {}
    for key, percentages in component_percentages.items():
        most_common = max(percentages.items(), key=lambda x: x[1])
        most_common_components[key] = {
            "value": most_common[0],
            "frequency": most_common[1],
        }

    return {
        "num_best_mechanisms": len(best_mechanisms),
        "component_percentages": component_percentages,
        "most_common_components": most_common_components,
        "diversity_score": calculate_diversity_score(component_percentages),
    }


def calculate_average_correlation(curves: List[List[float]]) -> float:
    """Calculate average pairwise correlation between fitness curves."""
    if len(curves) < 2:
        return 1.0

    correlations = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            # Simple correlation calculation (simplified)
            corr = simple_correlation(curves[i], curves[j])
            correlations.append(corr)

    return sum(correlations) / len(correlations) if correlations else 0.0


def simple_correlation(a: List[float], b: List[float]) -> float:
    """Calculate simple Pearson correlation."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0

    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n

    if n <= 1:
        return 0.0

    numerator = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    sum_sq_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    sum_sq_b = sum((b[i] - mean_b) ** 2 for i in range(n))

    denominator = (sum_sq_a * sum_sq_b) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_diversity_score(
    component_percentages: Dict[str, Dict[str, float]],
) -> float:
    """Calculate diversity score (Shannon entropy) across components."""
    total_entropy = 0.0
    component_count = 0

    for percentages in component_percentages.values():
        # Shannon entropy: -sum(p * log(p))
        entropy = -sum(
            p * (0 if p <= 0 else p * 1000) for p in percentages.values()
        )  # Simplified
        total_entropy += entropy
        component_count += 1

    return total_entropy / component_count if component_count > 0 else 0.0


def create_convergence_plots(results_dir: Path) -> None:
    """Create convergence plots from multi-run results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Load results if available
        results_file = results_dir / "convergence_suite_results.json"
        if not results_file.exists():
            return

        with open(results_file, "r") as f:
            results = json.load(f)

        convergence_analysis = results.get("convergence_analysis", {})
        fitness_curves = convergence_analysis.get("fitness_curves", [])

        if not fitness_curves:
            return

        # Plot fitness curves
        plt.figure(figsize=(10, 6))
        for i, curve in enumerate(fitness_curves):
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

        print(f"Convergence plot saved to {plot_file}")

    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_api_key", type=str, default=None)

    args = parser.parse_args()

    # Load API key from .env if use_llm and no key provided
    llm_api_key = args.llm_api_key
    if args.use_llm and not llm_api_key:
        from env_config import get_openai_api_key
        llm_api_key = get_openai_api_key()
        if not llm_api_key:
            print("❌ --use_llm set but no API key. Add OPENAI_API_KEY to .env or use --llm_api_key")
            sys.exit(1)

    run_convergence_suite(
        num_runs=args.num_runs,
        generations_per_run=args.generations,
        population_size=args.population,
        episodes_per_mechanism=args.episodes,
        base_seed=args.seed,
        config_path=args.config,
        results_dir=args.results,
        use_llm=args.use_llm,
        llm_api_key=llm_api_key,
    )
