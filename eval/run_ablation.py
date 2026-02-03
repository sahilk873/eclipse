"""Ablation: flip one component of best mechanism (info, gating, redirect) and re-run."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from eval.run_episodes import load_config, run_episodes
from eval.metrics import compute_fitness


def ablation_variants(mechanism: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """
    Create variants: (1) info policy none vs exact, (2) remove gating (always_admit),
    (3) remove redirect (redirect_low_risk=False). Return list of (variant_name, mechanism_dict).
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    # 1) Change info policy
    for info_mode in ("none", "exact"):
        m = copy.deepcopy(mechanism)
        m["info_mode"] = info_mode
        variants.append((f"info_{info_mode}", m))
    # 2) Remove gating -> always_admit
    m2 = copy.deepcopy(mechanism)
    m2["gating_mode"] = "always_admit"
    if "gating_threshold" in m2:
        del m2["gating_threshold"]
    if "capacity_load" in m2:
        del m2["capacity_load"]
    variants.append(("no_gating_always_admit", m2))
    # 3) Remove redirect
    m3 = copy.deepcopy(mechanism)
    m3["redirect_low_risk"] = False
    variants.append(("no_redirect", m3))
    return variants


def run_ablation(
    mechanism: dict[str, Any],
    params: dict[str, Any],
    n_episodes: int = 200,
    base_seed: int = 0,
) -> list[dict[str, Any]]:
    """Run baseline mechanism and each ablation variant; return rows with metrics."""
    base_name = "best_baseline"
    rows: list[dict[str, Any]] = []
    fitness_weights = params.get("fitness_weights") or {}

    # Baseline (original)
    mean_m, _, _, _, _ = run_episodes(mechanism, params, n_episodes, base_seed=base_seed)
    rows.append({
        "variant": base_name,
        "throughput_mean": mean_m.get("throughput", 0),
        "critical_TTC_p95_mean": mean_m.get("critical_TTC_p95", 0),
        "fitness": compute_fitness(mean_m, fitness_weights),
        "adverse_events_rate_mean": mean_m.get("adverse_events_rate", 0),
        "overload_time_mean": mean_m.get("overload_time", 0),
    })

    for name, m in ablation_variants(mechanism):
        mean_m, _, _, _, _ = run_episodes(m, params, n_episodes, base_seed=base_seed + 1000)
        rows.append({
            "variant": name,
            "throughput_mean": mean_m.get("throughput", 0),
            "critical_TTC_p95_mean": mean_m.get("critical_TTC_p95", 0),
            "fitness": compute_fitness(mean_m, fitness_weights),
            "adverse_events_rate_mean": mean_m.get("adverse_events_rate", 0),
            "overload_time_mean": mean_m.get("overload_time", 0),
        })
    return rows


def main(
    mechanism_path: str | Path,
    config_path: str | Path | None = None,
    results_dir: str | Path = "results",
    n_episodes: int = 200,
    base_seed: int = 0,
) -> None:
    with open(mechanism_path) as f:
        mechanism = json.load(f)
    params = load_config(config_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = run_ablation(mechanism, params, n_episodes=n_episodes, base_seed=base_seed)
    out_csv = results_dir / "ablation_results.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    from eval.run_episodes import load_config as _load
    return _load(config_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mechanism_path", type=str, help="Path to best mechanism JSON")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(
        mechanism_path=args.mechanism_path,
        config_path=args.config,
        results_dir=args.results_dir,
        n_episodes=args.n_episodes,
        base_seed=args.seed,
    )
