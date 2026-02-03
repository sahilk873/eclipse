"""
Transfer evaluation: Option A (train on default, evaluate on alternate configs)
and Option B (train on alternate config, evaluate on default).

Writes results/transfer_results.json for use by make_report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_episodes import evaluate_mechanism, load_config

# Config name -> path (relative to project root)
TRANSFER_CONFIG_NAMES = ["default", "high_volume", "higher_acuity", "low_resource"]
CONFIG_PATHS = {
    "default": PROJECT_ROOT / "config" / "default.yaml",
    "high_volume": PROJECT_ROOT / "config" / "high_volume.yaml",
    "higher_acuity": PROJECT_ROOT / "config" / "higher_acuity.yaml",
    "low_resource": PROJECT_ROOT / "config" / "low_resource.yaml",
}


def _config_path(name: str) -> Path:
    p = CONFIG_PATHS.get(name)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Config {name} not found: {p}")
    return p


def run_transfer_evaluation(
    results_dir: str | Path = "results",
    evolution_results_path: Optional[str] = None,
    episodes: int = 100,
    base_seed: int = 0,
    run_evolution_on: Optional[List[str]] = None,
    evolution_generations: int = 15,
    evolution_population: int = 30,
    evolution_episodes: int = 30,
    use_llm: bool = False,
    llm_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Option A: Load best mechanism from evolution_result.json; evaluate on default,
    high_volume, higher_acuity, low_resource. Option B: For each name in
    run_evolution_on, run evolution with that config and evaluate best on default.

    Returns dict for transfer_results.json.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if evolution_results_path is None:
        evolution_results_path = results_dir / "evolution_result.json"
    evolution_results_path = Path(evolution_results_path)

    evaluations: List[Dict[str, Any]] = []
    run_evolution_on = run_evolution_on or []

    # Option A: best from default -> evaluate on all test configs
    if evolution_results_path.exists():
        with open(evolution_results_path, "r") as f:
            main_evo = json.load(f)
        best_mechanism = main_evo.get("best_mechanism")
        if best_mechanism:
            mech_id = best_mechanism.get("meta", {}).get("id", "best_default")
            for test_config in TRANSFER_CONFIG_NAMES:
                params = load_config(_config_path(test_config))
                try:
                    eval_result = evaluate_mechanism(
                        best_mechanism,
                        params,
                        K=episodes,
                        base_seed=base_seed,
                        fitness_weights=params.get("fitness_weights"),
                    )
                    evaluations.append({
                        "mechanism_id": mech_id,
                        "train_config": "default",
                        "test_config": test_config,
                        "mechanism_source": "evolution_result",
                        "fitness": eval_result.get("fitness", 0),
                        "feasible": eval_result.get("feasible", False),
                        "mean_metrics": eval_result.get("mean_metrics", {}),
                    })
                except Exception as e:
                    evaluations.append({
                        "mechanism_id": mech_id,
                        "train_config": "default",
                        "test_config": test_config,
                        "mechanism_source": "evolution_result",
                        "error": str(e),
                        "fitness": None,
                        "feasible": False,
                    })
    else:
        print(f"Warning: {evolution_results_path} not found; skipping Option A.")

    # Option B: run evolution on alternate config(s), then evaluate best on default
    if run_evolution_on:
        default_params = load_config(_config_path("default"))
    for train_config in run_evolution_on:
        if train_config not in CONFIG_PATHS or not _config_path(train_config).exists():
            print(f"Warning: config {train_config} not found; skipping.")
            continue
        print(f"Running evolution on {train_config} (Option B)...")
        try:
            from evolution.evolve import evolve_mechanisms
            from evolution.llm_mutation import llm_mutate

            llm_mutate_func = None
            if use_llm and llm_api_key:
                llm_mutate_func = lambda *args, **kwargs: llm_mutate(*args, api_key=llm_api_key, **kwargs)

            evo_result = evolve_mechanisms(
                generations=evolution_generations,
                population_size=evolution_population,
                episodes_per_mechanism=evolution_episodes,
                elite_size=max(1, evolution_population // 10),
                config_path=str(_config_path(train_config)),
                results_dir=str(results_dir),
                base_seed=base_seed + 1000,
                llm_mutate=llm_mutate_func,
                llm_fraction=0.4 if use_llm else 0.0,
                run_id=train_config,
                adaptive_params={
                    "base_strength": 1.0,
                    "stagnation_generations": 3,
                    "improvement_threshold": 0.01,
                    "max_strength": 4.0,
                },
                mutation_jitter_range=0.35,
            )
            # Save for reproducibility
            alt_evo_file = results_dir / f"evolution_result_{train_config}.json"
            with open(alt_evo_file, "w") as f:
                json.dump(evo_result, f, indent=2)

            best_alt = evo_result.get("best_mechanism")
            if best_alt:
                mech_id = best_alt.get("meta", {}).get("id", f"best_{train_config}")
                try:
                    eval_result = evaluate_mechanism(
                        best_alt,
                        default_params,
                        K=episodes,
                        base_seed=base_seed + 2000,
                        fitness_weights=default_params.get("fitness_weights"),
                    )
                    evaluations.append({
                        "mechanism_id": mech_id,
                        "train_config": train_config,
                        "test_config": "default",
                        "mechanism_source": f"evolution_result_{train_config}",
                        "fitness": eval_result.get("fitness", 0),
                        "feasible": eval_result.get("feasible", False),
                        "mean_metrics": eval_result.get("mean_metrics", {}),
                    })
                except Exception as e:
                    evaluations.append({
                        "mechanism_id": mech_id,
                        "train_config": train_config,
                        "test_config": "default",
                        "mechanism_source": f"evolution_result_{train_config}",
                        "error": str(e),
                        "fitness": None,
                        "feasible": False,
                    })
        except Exception as e:
            print(f"Evolution on {train_config} failed: {e}")
            evaluations.append({
                "train_config": train_config,
                "test_config": "default",
                "mechanism_source": f"evolution_result_{train_config}",
                "error": str(e),
                "fitness": None,
                "feasible": False,
            })

    # Build summary for report
    fitness_default_on_default = None
    for e in evaluations:
        if e.get("train_config") == "default" and e.get("test_config") == "default" and e.get("fitness") is not None:
            fitness_default_on_default = e["fitness"]
            break
    transfer_table = [
        {"train_config": e["train_config"], "test_config": e["test_config"], "fitness": e.get("fitness"), "feasible": e.get("feasible")}
        for e in evaluations
    ]
    summary = {}
    if fitness_default_on_default is not None:
        by_test = {e["test_config"]: e.get("fitness") for e in evaluations if e.get("train_config") == "default" and e.get("fitness") is not None}
        summary["best_from_default_on_alternates"] = by_test
        if "high_volume" in by_test and by_test["high_volume"] is not None:
            pct = (1 - by_test["high_volume"] / fitness_default_on_default) * 100 if fitness_default_on_default != 0 else 0
            summary["degradation_on_high_volume_pct"] = round(pct, 1)

    out = {
        "evaluations": evaluations,
        "transfer_table": transfer_table,
        "summary": summary,
        "config": {
            "episodes": episodes,
            "base_seed": base_seed,
            "run_evolution_on": run_evolution_on,
        },
    }
    out_path = results_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transfer evaluation (Option A and B)")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--evolution_results", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--run_evolution_on", type=str, nargs="*", default=[], help="e.g. high_volume for Option B")
    parser.add_argument("--evolution_generations", type=int, default=15)
    parser.add_argument("--evolution_population", type=int, default=30)
    parser.add_argument("--evolution_episodes", type=int, default=30)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_api_key", type=str, default=None)
    args = parser.parse_args()

    try:
        from env_config import get_openai_api_key
        api_key = args.llm_api_key or get_openai_api_key()
    except Exception:
        api_key = None

    run_transfer_evaluation(
        results_dir=args.results_dir,
        evolution_results_path=args.evolution_results,
        episodes=args.episodes,
        base_seed=args.base_seed,
        run_evolution_on=args.run_evolution_on if args.run_evolution_on else None,
        evolution_generations=args.evolution_generations,
        evolution_population=args.evolution_population,
        evolution_episodes=args.evolution_episodes,
        use_llm=args.use_llm,
        llm_api_key=api_key,
    )
