"""Run evolution for G generations with N population, K episodes per evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from eval.run_episodes import load_config, run_episodes
from eval.metrics import compute_fitness, is_feasible, aggregate_metrics
from evolution.selection import select_top_m
from evolution.reproduction import create_offspring, create_initial_population
from mechanisms.schema import validate_mechanism
from baselines.definitions import BASELINES


def load_evolution_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load full config; return params for sim + evolution settings."""
    path = config_path or (Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    if not Path(path).exists():
        return {
            "params": _default_params(),
            "N": 30,
            "M": 10,
            "K": 20,
            "G": 30,
            "use_llm": False,
            "results_dir": "results",
        }
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    from eval.run_episodes import _config_to_params
    params = _config_to_params(cfg)
    evo = cfg.get("evolution", {})
    return {
        "params": params,
        "N": evo.get("N", 30),
        "M": evo.get("M", 10),
        "K": evo.get("K", 20),
        "G": evo.get("G", 30),
        "use_llm": evo.get("use_llm", False),
        "results_dir": cfg.get("results_dir", "results"),
        "fitness_weights": cfg.get("fitness", {}),
    }


def _default_params() -> dict[str, Any]:
    from eval.run_episodes import _default_params
    return _default_params()


def run_evolution(
    config_path: str | Path | None = None,
    master_seed: int = 0,
    save_every: int = 1,
) -> tuple[list[dict], list[float], list[bool], dict]:
    """
    Run full evolution: G generations, N population, K episodes per eval.
    Returns (best_mechanisms_per_gen, best_fitness_per_gen, feasible_per_gen, config_used).
    """
    cfg = load_evolution_config(config_path)
    params = cfg["params"]
    N = cfg["N"]
    M = cfg["M"]
    K = cfg["K"]
    G = cfg["G"]
    use_llm = cfg["use_llm"]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fitness_weights = cfg.get("fitness_weights") or params.get("fitness_weights") or {}

    # Initial population
    population = create_initial_population(N, seed=master_seed, baselines=BASELINES[:3])
    best_per_gen: list[dict] = []
    best_fitness_per_gen: list[float] = []
    feasible_per_gen: list[bool] = []

    for gen in range(G):
        # Evaluate each mechanism with K episodes
        fitness_list: list[float] = []
        feasible_list: list[bool] = []
        metrics_list: list[dict] = []
        constraints_list: list[dict] = []

        for i, mechanism in enumerate(population):
            mean_m, std_m, constraint_rates, _, all_constraints = run_episodes(
                mechanism, params, K, base_seed=master_seed + gen * 10000 + i * 100
            )
            constraints_violated = {k: (v > 0.5) for k, v in constraint_rates.items()}
            feasible = is_feasible(constraints_violated)
            fitness = compute_fitness(mean_m, fitness_weights)
            fitness_list.append(fitness)
            feasible_list.append(feasible)
            metrics_list.append(mean_m)
            constraints_list.append(constraints_violated)

        # Select top M
        indices = select_top_m(population, fitness_list, feasible_list, M)
        parents = [population[i] for i in indices]
        best_idx = indices[0]
        best_mechanism = population[best_idx]
        best_fitness = fitness_list[best_idx]
        best_feasible = feasible_list[best_idx]

        best_per_gen.append(dict(best_mechanism))
        best_fitness_per_gen.append(best_fitness)
        feasible_per_gen.append(best_feasible)

        # Log
        print(f"Gen {gen}: best_fitness={best_fitness:.2f} feasible={best_feasible} n_feasible={sum(feasible_list)}")

        # Save checkpoint
        if save_every and (gen % save_every == 0 or gen == G - 1):
            (results_dir / "evolution").mkdir(parents=True, exist_ok=True)
            with open(results_dir / "evolution" / f"best_gen_{gen}.json", "w") as f:
                json.dump(best_mechanism, f, indent=2)
            with open(results_dir / "evolution" / f"convergence_seed_{master_seed}.json", "w") as f:
                json.dump({
                    "best_fitness_per_gen": best_fitness_per_gen,
                    "feasible_per_gen": feasible_per_gen,
                }, f, indent=2)

        # Reproduction: N offspring from parents (LLM plug-in when use_llm and llm_mutation available)
        llm_mutate = None
        failure_bullets: list[str] = []
        if use_llm:
            try:
                from evolution.llm_mutation import llm_mutate as _llm, failure_analysis_from_metrics
                llm_mutate = _llm
                parent_metrics_for_llm = [dict(metrics_list[i], fitness=fitness_list[i]) for i in indices]
                parent_constraints_for_llm = [constraints_list[i] for i in indices]
                failure_bullets = failure_analysis_from_metrics(parent_metrics_for_llm, parent_constraints_for_llm)
            except ImportError:
                pass
        parent_metrics = [dict(metrics_list[i], fitness=fitness_list[i]) for i in indices]
        parent_constraints = [constraints_list[i] for i in indices] if use_llm else []
        offspring = create_offspring(
            parents,
            target_size=N,
            seed=master_seed + gen * 20000,
            llm_mutate=llm_mutate,
            llm_fraction=0.5 if (use_llm and llm_mutate) else 0.0,
            parent_metrics=parent_metrics,
            parent_failures=failure_bullets if use_llm else [],
            parent_constraints=parent_constraints,
        )
        population = offspring

    return best_per_gen, best_fitness_per_gen, feasible_per_gen, cfg


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1)
    args = p.parse_args()
    run_evolution(config_path=args.config, master_seed=args.seed, save_every=args.save_every)
