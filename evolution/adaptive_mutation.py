"""Adaptive mutation strength based on convergence stagnation."""

from __future__ import annotations

import math
from typing import Any, List

from mechanisms.mutation import mutate_mechanism
from mechanisms.genome import random_mechanism


class AdaptiveMutation:
    """Manages adaptive mutation strength for evolutionary search."""

    def __init__(
        self,
        base_strength: float = 1.0,
        min_strength: float = 0.1,
        max_strength: float = 3.0,
        stagnation_generations: int = 5,
        improvement_threshold: float = 0.01,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.9,
        exploration_boost: float = 2.0,
    ):
        """
        Initialize adaptive mutation parameters.

        Args:
            base_strength: Initial mutation strength
            min_strength: Minimum allowed mutation strength
            max_strength: Maximum allowed mutation strength
            stagnation_generations: Number of generations without improvement before increasing strength
            improvement_threshold: Minimum relative improvement to consider as progress
            increase_factor: Factor to multiply strength when stagnating
            decrease_factor: Factor to multiply strength when improving
            exploration_boost: Additional strength boost when major stagnation detected
        """
        self.base_strength = base_strength
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.stagnation_generations = stagnation_generations
        self.improvement_threshold = improvement_threshold
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.exploration_boost = exploration_boost

        # State tracking
        self.current_strength = base_strength
        self.generations_without_improvement = 0
        self.last_best_fitness = float("-inf")
        self.fitness_history: List[float] = []
        self.major_stagnation_detected = False

    def update_strength(self, current_best_fitness: float) -> float:
        """
        Update mutation strength based on current best fitness.

        Args:
            current_best_fitness: Best fitness from current generation

        Returns:
            Updated mutation strength
        """
        # Calculate improvement
        improvement = current_best_fitness - self.last_best_fitness
        relative_improvement = improvement / max(abs(self.last_best_fitness), 1.0)

        # Check if we have meaningful improvement
        if relative_improvement > self.improvement_threshold:
            # We're improving - decay mutation strength
            self.current_strength = max(
                self.min_strength, self.current_strength * self.decrease_factor
            )
            self.generations_without_improvement = 0
            self.major_stagnation_detected = False
        else:
            # No improvement
            self.generations_without_improvement += 1

            if self.generations_without_improvement >= self.stagnation_generations:
                # Stagnation detected - increase mutation strength
                old_strength = self.current_strength

                # Check for major stagnation (very long no improvement)
                if (
                    self.generations_without_improvement
                    >= 2 * self.stagnation_generations
                ):
                    self.major_stagnation_detected = True
                    self.current_strength = min(
                        self.max_strength,
                        self.current_strength
                        * self.increase_factor
                        * self.exploration_boost,
                    )
                else:
                    self.current_strength = min(
                        self.max_strength, self.current_strength * self.increase_factor
                    )

                # Increase probability of mode swaps during stagnation
                # This is handled by the strength parameter in mutation functions

                # Reset counter after adjustment
                if self.current_strength != old_strength:
                    self.generations_without_improvement = 0

        # Update tracking
        self.last_best_fitness = current_best_fitness
        self.fitness_history.append(current_best_fitness)

        # Keep history manageable
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-50:]

        return self.current_strength

    def mutate_population(
        self,
        parents: List[dict[str, Any]],
        offspring_size: int,
        seed: int = 0,
        elitism_rate: float = 0.1,
    ) -> List[dict[str, Any]]:
        """
        Create offspring using current mutation strength.

        Args:
            parents: Parent mechanisms
            offspring_size: Number of offspring to create
            seed: Random seed
            elitism_rate: Fraction of offspring that should be small mutations (elite preservation)

        Returns:
            List of offspring mechanisms
        """
        import random

        rng = random.Random(seed)
        offspring: List[dict[str, Any]] = []

        # Elite offspring with small mutations
        n_elite = int(offspring_size * elitism_rate)
        for i in range(n_elite):
            parent = rng.choice(parents)
            # Use reduced strength for elite offspring
            elite_strength = max(self.min_strength, self.current_strength * 0.5)
            child = mutate_mechanism(
                parent, seed=rng.randint(0, 2**31 - 1), mutation_strength=elite_strength
            )
            offspring.append(child)

        # Regular offspring with current strength
        for i in range(offspring_size - n_elite):
            parent = rng.choice(parents)

            # During major stagnation, occasionally use random mechanisms
            if self.major_stagnation_detected and rng.random() < 0.1:
                child = random_mechanism(seed=rng.randint(0, 2**31 - 1))
            else:
                child = mutate_mechanism(
                    parent,
                    seed=rng.randint(0, 2**31 - 1),
                    mutation_strength=self.current_strength,
                )

            offspring.append(child)

        return offspring

    def get_mode_swap_probability(self) -> float:
        """
        Get current probability of performing mode swaps vs parameter tweaks.
        Higher during stagnation to encourage exploration.

        Returns:
            Probability of mode swap (0.0 to 1.0)
        """
        if self.major_stagnation_detected:
            return 0.4  # High exploration
        elif self.generations_without_improvement >= self.stagnation_generations:
            return 0.3  # Medium exploration
        else:
            return 0.2  # Low exploration (mostly parameter tuning)

    def should_increase_exploration(self) -> bool:
        """
        Check if we should increase exploration (reduce elitism, increase randomness).

        Returns:
            True if exploration should be increased
        """
        return (
            self.generations_without_improvement >= self.stagnation_generations
            or self.major_stagnation_detected
        )

    def get_exploration_settings(self) -> dict[str, Any]:
        """
        Get current exploration settings for evolution loop.

        Returns:
            Dictionary with exploration parameters
        """
        return {
            "mutation_strength": self.current_strength,
            "mode_swap_probability": self.get_mode_swap_probability(),
            "exploration_mode": self.should_increase_exploration(),
            "major_stagnation": self.major_stagnation_detected,
            "generations_stagnant": self.generations_without_improvement,
        }

    def reset(self) -> None:
        """Reset adaptive mutation state."""
        self.current_strength = self.base_strength
        self.generations_without_improvement = 0
        self.last_best_fitness = float("-inf")
        self.fitness_history = []
        self.major_stagnation_detected = False

    def get_state_summary(self) -> dict[str, Any]:
        """
        Get summary of current adaptive mutation state.

        Returns:
            Dictionary with state information
        """
        return {
            "current_strength": self.current_strength,
            "generations_without_improvement": self.generations_without_improvement,
            "last_best_fitness": self.last_best_fitness,
            "fitness_trend": "improving"
            if len(self.fitness_history) < 2
            else "improving"
            if self.fitness_history[-1] > self.fitness_history[-2]
            else "stagnant",
            "major_stagnation_detected": self.major_stagnation_detected,
            "mode_swap_probability": self.get_mode_swap_probability(),
        }
