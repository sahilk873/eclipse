"""Population database with SQLite storage for mechanisms, metrics, and component logging."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mechanisms.schema import get_component_flags, validate_mechanism


class PopulationDB:
    """SQLite database for storing evaluated mechanisms and their performance."""

    def __init__(self, db_path: str | Path, run_id: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            run_id = (
                f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
        self.run_id = run_id

        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mechanisms (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    mechanism_json TEXT NOT NULL,
                    component_flags TEXT NOT NULL,
                    parent_ids TEXT,
                    seed_tag TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, generation, id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    eval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    mechanism_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    fitness REAL NOT NULL,
                    feasible BOOLEAN NOT NULL,
                    reasoning_trace TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (mechanism_id) REFERENCES mechanisms (id),
                    UNIQUE(run_id, mechanism_id, seed)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS convergence (
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    best_fitness REAL NOT NULL,
                    best_feasible BOOLEAN NOT NULL,
                    avg_fitness REAL NOT NULL,
                    n_feasible INTEGER NOT NULL,
                    population_size INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, generation)
                )
            """)

            # Migration: add reasoning_trace to existing evaluations tables
            try:
                cursor = conn.execute("PRAGMA table_info(evaluations)")
                columns = [row[1] for row in cursor.fetchall()]
                if columns and "reasoning_trace" not in columns:
                    conn.execute(
                        "ALTER TABLE evaluations ADD COLUMN reasoning_trace TEXT"
                    )
            except sqlite3.OperationalError:
                pass  # Table may not exist yet

            # Indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mechanisms_run_gen ON mechanisms(run_id, generation)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evaluations_run_gen ON evaluations(run_id, generation)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_evaluations_mechanism ON evaluations(mechanism_id)"
            )

    def store_mechanism(
        self,
        mechanism: Dict[str, Any],
        generation: int,
        seed_tag: str,
        parent_ids: Optional[List[str]] = None,
    ) -> str:
        """Store a mechanism and return its ID."""
        # Validate mechanism
        valid, errors = validate_mechanism(mechanism)
        if not valid:
            raise ValueError(f"Invalid mechanism: {errors}")

        # Get or generate mechanism ID
        mechanism_id = mechanism.get("meta", {}).get("id")
        if not mechanism_id:
            mechanism_id = str(uuid.uuid4())
            if "meta" not in mechanism:
                mechanism["meta"] = {}
            mechanism["meta"]["id"] = mechanism_id

        # Extract component flags for structure analysis
        component_flags = get_component_flags(mechanism)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO mechanisms 
                (id, run_id, generation, mechanism_json, component_flags, parent_ids, seed_tag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    mechanism_id,
                    self.run_id,
                    generation,
                    json.dumps(mechanism),
                    json.dumps(component_flags),
                    json.dumps(parent_ids or []),
                    seed_tag,
                ),
            )

        return mechanism_id

    def store_evaluation(
        self,
        mechanism_id: str,
        generation: int,
        seed: int,
        metrics: Dict[str, Any],
        constraints: Dict[str, bool],
        fitness: float,
        feasible: bool,
        reasoning_trace: str = None,
    ) -> str:
        """Store evaluation results in database."""
        eval_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluations 
                (eval_id, run_id, mechanism_id, generation, seed, metrics_json, constraints_json, fitness, feasible, reasoning_trace, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    eval_id,
                    self.run_id,
                    mechanism_id,
                    generation,
                    seed,
                    json.dumps(metrics),
                    json.dumps(constraints),
                    fitness,
                    feasible,
                    reasoning_trace,
                ),
            )
        return eval_id

    def store_convergence(
        self,
        generation: int,
        best_fitness: float,
        best_feasible: bool,
        avg_fitness: float,
        n_feasible: int,
        population_size: int,
    ) -> None:
        """Store convergence statistics for a generation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO convergence
                (run_id, generation, best_fitness, best_feasible, avg_fitness, n_feasible, population_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.run_id,
                    generation,
                    best_fitness,
                    best_feasible,
                    avg_fitness,
                    n_feasible,
                    population_size,
                ),
            )

    def get_mechanism(self, mechanism_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a mechanism by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT mechanism_json FROM mechanisms WHERE id = ? AND run_id = ?
            """,
                (mechanism_id, self.run_id),
            ).fetchone()

            if row:
                return json.loads(row[0])
            return None

    def get_best_mechanisms(
        self, generation: int, limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float, bool]]:
        """Get top mechanisms from a generation by fitness."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT m.mechanism_json, e.fitness, e.feasible
                FROM mechanisms m
                JOIN evaluations e ON m.id = e.mechanism_id
                WHERE m.run_id = ? AND m.generation = ? AND e.seed = 0  -- Use first seed as representative
                ORDER BY e.fitness DESC, e.feasible DESC
                LIMIT ?
            """,
                (self.run_id, generation, limit),
            ).fetchall()

            return [(json.loads(row[0]), row[1], bool(row[2])) for row in rows]

    def get_convergence_data(self) -> List[Dict[str, Any]]:
        """Get convergence data for all generations."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT generation, best_fitness, best_feasible, avg_fitness, n_feasible, population_size
                FROM convergence
                WHERE run_id = ?
                ORDER BY generation
            """,
                (self.run_id,),
            ).fetchall()

            return [
                {
                    "generation": row[0],
                    "best_fitness": row[1],
                    "best_feasible": bool(row[2]),
                    "avg_fitness": row[3],
                    "n_feasible": row[4],
                    "population_size": row[5],
                }
                for row in rows
            ]

    def get_component_frequencies(
        self, generation: int, top_percent: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """Get frequency of component flags among top percentage of mechanisms."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT m.component_flags, e.fitness
                FROM mechanisms m
                JOIN evaluations e ON m.id = e.mechanism_id
                WHERE m.run_id = ? AND m.generation = ? AND e.seed = 0
                ORDER BY e.fitness DESC
            """,
                (self.run_id, generation),
            ).fetchall()

            if not rows:
                return {}

            # Take top percentage
            n_top = max(1, int(len(rows) * top_percent))
            top_rows = rows[:n_top]

            # Count component frequencies
            component_counts = {}
            for component_flags_json, _ in top_rows:
                component_flags = json.loads(component_flags_json)
                for key, value in component_flags.items():
                    if key not in component_counts:
                        component_counts[key] = {}
                    if value not in component_counts[key]:
                        component_counts[key][value] = 0
                    component_counts[key][value] += 1

            # Convert to percentages
            frequencies = {}
            for key, value_counts in component_counts.items():
                total = sum(value_counts.values())
                frequencies[key] = {k: v / total for k, v in value_counts.items()}

            return frequencies

    def get_generation_population(self, generation: int) -> List[Dict[str, Any]]:
        """Get all mechanisms from a generation."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT m.mechanism_json
                FROM mechanisms m
                WHERE m.run_id = ? AND m.generation = ?
                ORDER BY m.id
            """,
                (self.run_id, generation),
            ).fetchall()

            return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        """Close database connection (placeholder for interface consistency)."""
        pass
