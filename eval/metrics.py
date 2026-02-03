"""Fitness, constraints, and Pareto metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def check_constraints(
    metrics: dict[str, Any],
    constraints_violated: dict[str, bool],
    config: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Return dict of constraint names -> True if violated."""
    cfg = config or {}
    epsilon = cfg.get("missed_critical_epsilon", 0.02)
    T_crit = cfg.get("critical_TTC_minutes", 30)
    exceed_pct = cfg.get("critical_TTC_exceed_pct", 0.05)

    out = dict(constraints_violated)
    if metrics.get("missed_critical_rate", 0) > epsilon:
        out["missed_critical_rate"] = True
    if metrics.get("critical_TTC_exceeded", False):
        out["critical_TTC_exceeded"] = True
    # Also derive from metrics if not in constraints_violated
    n_critical = (
        metrics.get("critical_served", 0)
        + metrics.get("critical_redirected", 0)
        + metrics.get("critical_left", 0)
    )
    if n_critical > 0:
        missed = (metrics.get("critical_redirected", 0) + metrics.get("critical_left", 0)) / n_critical
        if missed > epsilon:
            out["missed_critical_rate"] = True
    return out


def is_feasible(constraints_violated: dict[str, bool]) -> bool:
    """True if no hard constraint is violated."""
    return not any(constraints_violated.values())


def compute_fitness(
    metrics: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Maximize: fitness = -A*adverse_events - B*critical_TTC_p95 - C*overload_time + D*throughput - E*mean_wait
    """
    w = weights or {}
    A = w.get("A", 100)
    B = w.get("B", 1.0)
    C = w.get("C", 0.1)
    D = w.get("D", 2.0)
    E = w.get("E", 0.5)

    adverse = metrics.get("adverse_events_rate", 0)
    ttc_p95 = metrics.get("critical_TTC_p95", 0)
    overload = metrics.get("overload_time", 0)
    throughput = metrics.get("throughput", 0)
    mean_wait = metrics.get("mean_wait", 0)

    return float(
        -A * adverse - B * ttc_p95 - C * overload + D * throughput - E * mean_wait
    )


def pareto_metrics(metrics: dict[str, Any], constraints_violated: dict[str, bool]) -> tuple[float, float, float, float]:
    """
    Return (safety_indicator, critical_TTC_p95, overload_time, throughput) for Pareto frontier.
    safety_indicator: 0 if feasible, else penalty (e.g. missed_critical_rate * 1000).
    """
    safety = 0.0 if is_feasible(constraints_violated) else 1000.0
    if constraints_violated.get("missed_critical_rate"):
        safety += metrics.get("missed_critical_rate", 0) * 100
    return (
        safety,
        metrics.get("critical_TTC_p95", 0),
        metrics.get("overload_time", 0),
        metrics.get("throughput", 0),
    )


def aggregate_metrics(metrics_list: list[dict], constraints_list: list[dict]) -> tuple[dict, dict, dict]:
    """
    Aggregate K episodes: mean metrics, mean constraints_violated (as rates), std for key metrics.
    """
    if not metrics_list:
        return {}, {}, {}

    keys = list(metrics_list[0].keys())
    means = {}
    stds = {}
    for k in keys:
        vals = [m.get(k, 0) for m in metrics_list]
        if isinstance(vals[0], (int, float)):
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals)) if len(vals) > 1 else 0.0
        else:
            means[k] = vals[0]

    # Constraint satisfaction: fraction of episodes where each constraint was violated
    constraint_rates = {}
    if constraints_list:
        for c in constraints_list[0].keys():
            constraint_rates[c] = float(np.mean([1.0 if d.get(c) else 0.0 for d in constraints_list]))

    return means, stds, constraint_rates


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for mean."""
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if values else (0.0, 0.0)
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / (n ** 0.5) if n > 1 else 0
    # Approximate 1.96 for 95%
    margin = 1.96 * se
    return (float(mean - margin), float(mean + margin))
