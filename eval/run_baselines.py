"""Run all baselines for 200 episodes and save CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from eval.run_episodes import load_config, run_episodes
from eval.metrics import compute_fitness, is_feasible, confidence_interval_95
from baselines.definitions import BASELINES, BASELINE_NAMES, get_baseline_name


def main(
    n_episodes: int = 200,
    config_path: str | Path | None = None,
    results_dir: str | Path = "results",
    base_seed: int = 0,
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = load_config(config_path)
    fitness_weights = params.get("fitness_weights") or {}
    # Load from config if present
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                fitness_weights = cfg.get("fitness", fitness_weights)
    except Exception:
        pass

    rows: list[dict] = []
    for i, mechanism in enumerate(BASELINES):
        name = get_baseline_name(i)
        mean_m, std_m, constraint_rates, all_metrics, all_constraints = run_episodes(
            mechanism, params, n_episodes, base_seed
        )
        feasible = not any(constraint_rates.get(k, 0) > 0.5 for k in ("missed_critical_rate", "critical_TTC_exceeded"))
        fitness = compute_fitness(mean_m, fitness_weights)
        # 95% CI for key metrics
        throughput_ci = confidence_interval_95([m["throughput"] for m in all_metrics])
        ttc_ci = confidence_interval_95([m["critical_TTC_p95"] for m in all_metrics]) if all(m.get("critical_TTC_p95") is not None for m in all_metrics) else (mean_m.get("critical_TTC_p95", 0), mean_m.get("critical_TTC_p95", 0))
        row = {
            "name": name,
            "fitness": fitness,
            "feasible": feasible,
            "throughput_mean": mean_m.get("throughput", 0),
            "throughput_ci_lower": throughput_ci[0],
            "throughput_ci_upper": throughput_ci[1],
            "critical_TTC_p95_mean": mean_m.get("critical_TTC_p95", 0),
            "critical_TTC_p95_ci_lower": ttc_ci[0],
            "critical_TTC_p95_ci_upper": ttc_ci[1],
            "adverse_events_rate_mean": mean_m.get("adverse_events_rate", 0),
            "overload_time_mean": mean_m.get("overload_time", 0),
            "missed_critical_rate_mean": mean_m.get("missed_critical_rate", 0),
            "mean_wait_mean": mean_m.get("mean_wait", 0),
            "constraint_missed_critical_rate": constraint_rates.get("missed_critical_rate", 0),
            "constraint_critical_TTC_exceeded": constraint_rates.get("critical_TTC_exceeded", 0),
        }
        rows.append(row)
        # Save each baseline mechanism JSON
        (results_dir / "baselines").mkdir(parents=True, exist_ok=True)
        with open(results_dir / "baselines" / f"{name}.json", "w") as f:
            json.dump(mechanism, f, indent=2)

    # Write CSV
    out_csv = results_dir / "baselines_results.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} baselines.")
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(n_episodes=args.n_episodes, config_path=args.config, results_dir=args.results_dir, base_seed=args.seed)
