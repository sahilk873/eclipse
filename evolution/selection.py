"""Constrained fitness and top-M selection."""

from __future__ import annotations

from typing import Any

from eval.metrics import is_feasible


def select_top_m(
    population: list[dict[str, Any]],
    fitness_list: list[float],
    feasible_list: list[bool],
    M: int,
) -> list[int]:
    """
    Select top M indices. Prefer feasible over infeasible; among feasible, rank by fitness (higher better).
    Among infeasible, rank by fitness. Returns list of indices into population.
    """
    indices = list(range(len(population)))
    # Sort: feasible first, then by fitness descending
    def key(i: int) -> tuple[int, float]:
        # feasible -> 0, infeasible -> 1 so feasible come first
        f = 0 if feasible_list[i] else 1
        return (f, -fitness_list[i])

    indices.sort(key=key)
    return indices[:M]


def select_top_m_with_metrics(
    population: list[dict[str, Any]],
    fitness_list: list[float],
    feasible_list: list[bool],
    metrics_list: list[dict],
    constraints_list: list[dict],
    M: int,
) -> tuple[list[int], list[dict], list[dict], list[dict], list[dict]]:
    """
    Select top M; return (indices, mechanisms, fitness, metrics, constraints) for selected.
    """
    indices = select_top_m(population, fitness_list, feasible_list, M)
    return (
        indices,
        [population[i] for i in indices],
        [fitness_list[i] for i in indices],
        [metrics_list[i] for i in indices],
        [constraints_list[i] for i in indices],
    )
