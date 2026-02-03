"""Create offspring from selected parents via random mutation (and optional LLM)."""

from __future__ import annotations

import random
from typing import Any, Callable

from mechanisms.mutation import mutate_mechanism
from mechanisms.genome import random_mechanism
from mechanisms.schema import validate_mechanism


def create_offspring(
    parents: list[dict[str, Any]],
    target_size: int,
    seed: int | None = None,
    llm_mutate: Callable[..., list[dict]] | None = None,
    llm_fraction: float = 0.0,
    parent_metrics: list[dict[str, Any]] | None = None,
    parent_failures: list[str] | None = None,
    parent_constraints: list[dict[str, Any]] | None = None,
    mutation_strength: float = 1.0,
    mutation_jitter_range: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Create target_size offspring. With llm_fraction=0, all via random mutation from parents.
    If llm_mutate is provided and llm_fraction > 0, that fraction of offspring come from LLM.
    parent_metrics and parent_failures are passed to llm_mutate when present.
    """
    rng = random.Random(seed)
    offspring: list[dict[str, Any]] = []
    n_llm = int(target_size * llm_fraction) if llm_mutate else 0
    n_random = target_size - n_llm

    # Random mutations from parents
    for _ in range(n_random):
        parent = rng.choice(parents)
        child = mutate_mechanism(
            parent,
            seed=rng.randint(0, 2**31 - 1),
            mutation_strength=mutation_strength,
            mutation_jitter_range=mutation_jitter_range,
        )
        valid, _ = validate_mechanism(child)
        if not valid:
            child = mutate_mechanism(
                parent,
                seed=rng.randint(0, 2**31 - 1),
                mutation_strength=mutation_strength,
                mutation_jitter_range=mutation_jitter_range,
            )
        offspring.append(child)

    # LLM-generated offspring (if requested)
    parent_metrics = parent_metrics or []
    parent_failures = parent_failures or []
    parent_constraints = parent_constraints or []
    if llm_mutate and n_llm > 0 and parents:
        try:
            # Pass constraints so LLM prompt shows feasible yes/no per mechanism
            kwargs: dict = {}
            if parent_constraints:
                kwargs["constraints_list"] = parent_constraints[:5]
            proposals = llm_mutate(parents[:5], parent_metrics[:5], parent_failures, **kwargs)
            for p in proposals[:n_llm]:
                valid, _ = validate_mechanism(p)
                if valid:
                    offspring.append(p)
                else:
                    offspring.append(
                        mutate_mechanism(
                            rng.choice(parents),
                            seed=rng.randint(0, 2**31 - 1),
                            mutation_strength=mutation_strength,
                            mutation_jitter_range=mutation_jitter_range,
                        )
                    )
            while len(offspring) < target_size:
                offspring.append(
                    mutate_mechanism(
                        rng.choice(parents),
                        seed=rng.randint(0, 2**31 - 1),
                        mutation_strength=mutation_strength,
                        mutation_jitter_range=mutation_jitter_range,
                    )
                )
        except Exception:
            for _ in range(n_llm):
                offspring.append(
                    mutate_mechanism(
                        rng.choice(parents),
                        seed=rng.randint(0, 2**31 - 1),
                        mutation_strength=mutation_strength,
                        mutation_jitter_range=mutation_jitter_range,
                    )
                )

    # Ensure we have exactly target_size
    while len(offspring) < target_size:
        offspring.append(
            mutate_mechanism(
                rng.choice(parents),
                seed=rng.randint(0, 2**31 - 1),
                mutation_strength=mutation_strength,
                mutation_jitter_range=mutation_jitter_range,
            )
        )
    return offspring[:target_size]


def create_initial_population(
    size: int,
    seed: int | None = None,
    baselines: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create initial population: mix of baselines (if provided) and random mechanisms."""
    rng = random.Random(seed)
    pop: list[dict[str, Any]] = []
    if baselines:
        for m in baselines[:size]:
            pop.append(dict(m))
    while len(pop) < size:
        pop.append(random_mechanism(seed=rng.randint(0, 2**31 - 1)))
    return pop[:size]
