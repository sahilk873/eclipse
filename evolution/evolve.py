"""Main evolutionary loop with adaptive mutation and population logging.

Runs population-based search over mechanism designs: evaluates each mechanism
via discrete-event simulation, selects elites, creates offspring via random
mutation and optional LLM-guided proposals, and adapts mutation strength
based on convergence stagnation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evolution.adaptive_mutation import AdaptiveMutation
from evolution.population_db import PopulationDB
from evolution.selection import select_top_m_with_metrics
from evolution.reproduction import create_initial_population, create_offspring
from eval.run_episodes import evaluate_mechanism, load_config
from baselines.definitions import BASELINES


def evolve_mechanisms(
    generations: int = 20,
    population_size: int = 50,
    episodes_per_mechanism: int = 50,
    elite_size: int = 5,
    config_path: Optional[str] = None,
    results_dir: str = "results",
    base_seed: int = 0,
    llm_mutate: Optional[Callable] = None,
    llm_fraction: float = 0.2,
    run_id: Optional[str] = None,
    adaptive_params: Optional[Dict[str, Any]] = None,
    mutation_jitter_range: float = 0.35,
) -> Dict[str, Any]:
    """
    Run evolutionary search for optimal healthcare mechanisms.

    Args:
        generations: Number of generations to evolve
        population_size: Size of population each generation
        episodes_per_mechanism: Number of simulation episodes per mechanism evaluation
        elite_size: Number of top mechanisms to preserve (elitism)
        config_path: Path to configuration file
        results_dir: Directory to save results
        base_seed: Base random seed
        llm_mutate: Optional LLM mutation function
        llm_fraction: Fraction of offspring created by LLM
        run_id: Unique identifier for this run
        adaptive_params: Parameters for adaptive mutation

    Returns:
        Dictionary with evolution results and best mechanisms
    """
    # Setup
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    params = load_config(config_path)

    # Setup population database
    db_path = results_dir / f"population_{run_id or 'default'}.db"
    db = PopulationDB(db_path, run_id)

    # Initialize adaptive mutation
    adaptive_params = adaptive_params or {}
    adaptive_mutation = AdaptiveMutation(**adaptive_params)

    # Create initial population
    rng = random.Random(base_seed)
    population = create_initial_population(
        size=population_size,
        seed=rng.randint(0, 2**31 - 1),
        baselines=BASELINES[:3],  # Include some baselines
    )

    # Evolution state
    best_mechanism = None
    best_fitness = float("-inf")
    convergence_data = []

    print(
        f"Starting evolution with {len(population)} mechanisms for {generations} generations"
    )

    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")

        # Evaluate population
        population_data = []
        for i, mechanism in enumerate(population):
            # Store mechanism in database
            mech_id = db.store_mechanism(
                mechanism=mechanism,
                generation=gen,
                seed_tag=f"gen{gen}_ind{i}",
                parent_ids=mechanism.get("meta", {}).get("parent_ids", []),
            )

            # Evaluate mechanism
            eval_result = evaluate_mechanism(
                mechanism=mechanism,
                params=params,
                K=episodes_per_mechanism,
                base_seed=base_seed + gen * population_size + i,
                fitness_weights=params.get("fitness_weights"),
            )

            # Store evaluation
            db.store_evaluation(
                mechanism_id=mech_id,
                generation=gen,
                seed=base_seed + gen * population_size + i,
                metrics=eval_result["mean_metrics"],
                constraints={
                    k: v for k, v in eval_result["constraint_rates"].items() if v > 0.5
                },
                fitness=eval_result["fitness"],
                feasible=eval_result["feasible"],
            )

            population_data.append(
                {
                    "mechanism": mechanism,
                    "mechanism_id": mech_id,
                    "fitness": eval_result["fitness"],
                    "feasible": eval_result["feasible"],
                    "metrics": eval_result["mean_metrics"],
                    "constraints": eval_result["constraint_rates"],
                }
            )

        # Sort by fitness
        population_data.sort(key=lambda x: (not x["feasible"], -x["fitness"]))

        # Track best mechanism
        current_best = population_data[0]
        if current_best["fitness"] > best_fitness:
            best_fitness = current_best["fitness"]
            best_mechanism = current_best["mechanism"]
            print(
                f"  New best fitness: {best_fitness:.4f} (feasible: {current_best['feasible']})"
            )

        # Update adaptive mutation
        current_best_fitness = current_best["fitness"]
        mutation_strength = adaptive_mutation.update_strength(current_best_fitness)

        # Calculate convergence statistics
        feasible_count = sum(1 for x in population_data if x["feasible"])
        avg_fitness = sum(x["fitness"] for x in population_data) / len(population_data)

        convergence_data.append(
            {
                "generation": gen,
                "best_fitness": current_best_fitness,
                "best_feasible": current_best["feasible"],
                "avg_fitness": avg_fitness,
                "n_feasible": feasible_count,
                "population_size": len(population_data),
                "mutation_strength": mutation_strength,
            }
        )

        # Store convergence data
        db.store_convergence(
            generation=gen,
            best_fitness=current_best_fitness,
            best_feasible=current_best["feasible"],
            avg_fitness=avg_fitness,
            n_feasible=feasible_count,
            population_size=len(population_data),
        )

        # Selection - get elite mechanisms
        (
            elite_indices,
            elite_mechanisms,
            elite_fitness,
            elite_metrics,
            elite_constraints,
        ) = select_top_m_with_metrics(
            population=[x["mechanism"] for x in population_data],
            fitness_list=[x["fitness"] for x in population_data],
            feasible_list=[x["feasible"] for x in population_data],
            metrics_list=[x["metrics"] for x in population_data],
            constraints_list=[x["constraints"] for x in population_data],
            M=elite_size,
        )

        # Create next generation
        if gen < generations - 1:  # Don't create offspring for last generation
            # Prepare parent data for LLM mutation
            parent_failures = []
            for i, x in enumerate(population_data[:5]):  # Top 5 mechanisms
                failures = [k for k, v in x["constraints"].items() if v > 0.5]
                parent_failures.append(
                    f"Mechanism {i}: {', '.join(failures) if failures else 'No violations'}"
                )

            # Create offspring (use adaptive mutation strength for random mutations)
            mutation_strength = adaptive_mutation.get_state_summary()["current_strength"]
            offspring = create_offspring(
                parents=elite_mechanisms,
                target_size=population_size - elite_size,
                seed=rng.randint(0, 2**31 - 1),
                llm_mutate=llm_mutate,
                llm_fraction=llm_fraction,
                parent_metrics=elite_metrics,
                parent_failures=parent_failures,
                parent_constraints=elite_constraints,
                mutation_strength=mutation_strength,
                mutation_jitter_range=mutation_jitter_range,
            )

            # Combine elite and offspring for next generation
            population = elite_mechanisms + offspring

            # Log adaptive mutation state
            adaptive_state = adaptive_mutation.get_state_summary()
            print(f"  Mutation strength: {adaptive_state['current_strength']:.3f}")
            print(
                f"  Generations stagnant: {adaptive_state['generations_without_improvement']}"
            )
            print(f"  Feasible mechanisms: {feasible_count}/{len(population_data)}")

    # Save final results
    final_results = {
        "best_mechanism": best_mechanism,
        "best_fitness": best_fitness,
        "convergence_data": convergence_data,
        "final_population_size": len(population_data),
        "generations": generations,
        "adaptive_state": adaptive_mutation.get_state_summary(),
        "run_id": db.run_id,
    }

    # Save best mechanism
    if best_mechanism:
        best_mech_path = results_dir / f"best_mechanism_{run_id or 'default'}.json"
        import json

        with open(best_mech_path, "w") as f:
            json.dump(best_mechanism, f, indent=2)
        print(f"Saved best mechanism to {best_mech_path}")

    # Save convergence data
    conv_path = results_dir / f"convergence_{run_id or 'default'}.json"
    import json

    with open(conv_path, "w") as f:
        json.dump(convergence_data, f, indent=2)

    print(f"Evolution completed. Best fitness: {best_fitness:.4f}")
    print(f"Results saved to {results_dir}")

    return final_results


def analyze_component_diversity(
    db: PopulationDB, generation: int
) -> Dict[str, Dict[str, float]]:
    """
    Analyze component diversity in the population.

    Args:
        db: Population database
        generation: Generation to analyze

    Returns:
        Dictionary with component frequencies
    """
    return db.get_component_frequencies(generation, top_percent=0.2)


def get_evolution_summary(db: PopulationDB) -> Dict[str, Any]:
    """
    Get summary of evolution results.

    Args:
        db: Population database

    Returns:
        Dictionary with evolution summary
    """
    convergence_data = db.get_convergence_data()

    if not convergence_data:
        return {"error": "No convergence data found"}

    # Calculate summary statistics
    best_fitnesses = [d["best_fitness"] for d in convergence_data]
    feasible_counts = [d["n_feasible"] for d in convergence_data]

    return {
        "total_generations": len(convergence_data),
        "initial_best_fitness": best_fitnesses[0] if best_fitnesses else 0,
        "final_best_fitness": best_fitnesses[-1] if best_fitnesses else 0,
        "fitness_improvement": best_fitnesses[-1] - best_fitnesses[0]
        if len(best_fitnesses) > 1
        else 0,
        "max_feasible_count": max(feasible_counts) if feasible_counts else 0,
        "final_feasible_count": feasible_counts[-1] if feasible_counts else 0,
        "convergence_curve": convergence_data,
    }
