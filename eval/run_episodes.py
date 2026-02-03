"""Run K episodes for a mechanism and aggregate metrics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mechanisms.genome import mechanism_to_dict
from sim.runner import run_episode
from eval.metrics import (
    aggregate_metrics,
    compute_fitness,
    check_constraints,
    is_feasible,
    pareto_metrics,
)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config YAML; merge sim, constraints, fitness into flat params for run_episode."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    path = Path(config_path)
    if not path.exists():
        return _default_params()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _config_to_params(cfg)


def _default_params() -> dict[str, Any]:
    return {
        "T": 480,
        "lambda": 4.0,
        "n_servers": 3,
        "Qmax": 20,
        "service": {},
        "patience": {},
        "risk_mix": {"critical": 0.1, "urgent": 0.3, "low": 0.6},
        "benefit_per_class": {"critical": 100, "urgent": 50, "low": 20},
        "c_wait": 0.5,
        "deterioration": {"enabled": False},
        "constraints": {"missed_critical_epsilon": 0.02, "critical_TTC_minutes": 30, "critical_TTC_exceed_pct": 0.05},
    }


def _config_to_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten config into params dict for run_episode."""
    sim = cfg.get("sim", {})
    constraints = cfg.get("constraints", {})
    fitness = cfg.get("fitness", {})
    params: dict[str, Any] = {
        "T": sim.get("T", 480),
        "lambda": sim.get("lambda", 4.0),
        "n_servers": sim.get("n_servers", 3),
        "Qmax": sim.get("Qmax", 20),
        "service": sim.get("service", {}),
        "patience": sim.get("patience", {}),
        "risk_mix": sim.get("risk_mix", {"critical": 0.1, "urgent": 0.3, "low": 0.6}),
        "benefit_per_class": sim.get("benefit_per_class", {"critical": 100, "urgent": 50, "low": 20}),
        "c_wait": sim.get("c_wait", 0.5),
        "deterioration": sim.get("deterioration", {}),
        "constraints": {
            "missed_critical_epsilon": constraints.get("missed_critical_epsilon", 0.02),
            "critical_TTC_minutes": constraints.get("critical_TTC_minutes", 30),
            "critical_TTC_exceed_pct": constraints.get("critical_TTC_exceed_pct", 0.05),
        },
        "fitness_weights": fitness,
    }
    return params


def run_episodes(
    mechanism: dict[str, Any],
    params: dict[str, Any],
    K: int,
    base_seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict], list[dict]]:
    """
    Run K episodes with seeds base_seed .. base_seed+K-1.
    Returns (mean_metrics, std_metrics, constraint_rates, all_metrics_list, all_constraints_list).
    """
    metrics_list: list[dict] = []
    constraints_list: list[dict] = []
    for i in range(K):
        seed = base_seed + i
        metrics, constraints_violated = run_episode(mechanism, params, seed)
        metrics_list.append(metrics)
        constraints_list.append(constraints_violated)

    mean_metrics, std_metrics, constraint_rates = aggregate_metrics(metrics_list, constraints_list)
    return mean_metrics, std_metrics, constraint_rates, metrics_list, constraints_list


def evaluate_mechanism(
    mechanism: dict[str, Any],
    params: dict[str, Any],
    K: int,
    base_seed: int,
    fitness_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Run K episodes and return summary: mean_metrics, fitness, feasible, pareto_vector.
    """
    # Normalize mechanism (JSON/LLM bins are lists; sim expects tuples)
    mechanism = mechanism_to_dict(mechanism)
    mean_m, std_m, constraint_rates, _, _ = run_episodes(mechanism, params, K, base_seed)
    # Build synthetic constraints_violated from rates (for feasibility: any rate > 0.5?)
    constraints_violated = {k: (v > 0.5) for k, v in constraint_rates.items()}
    fitness = compute_fitness(mean_m, fitness_weights or params.get("fitness_weights"))
    feasible = not any(constraints_violated.values())
    pvec = pareto_metrics(mean_m, constraints_violated)
    return {
        "mean_metrics": mean_m,
        "std_metrics": std_m,
        "constraint_rates": constraint_rates,
        "fitness": fitness,
        "feasible": feasible,
        "pareto_vector": pvec,
        "mechanism": mechanism,
    }
