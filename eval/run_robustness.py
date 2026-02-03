"""Re-evaluate top mechanisms under distribution shift (lambda, service, risk mix, patience)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from eval.run_episodes import load_config, run_episodes, _config_to_params
from eval.metrics import compute_fitness
from baselines.definitions import BASELINES, BASELINE_NAMES


def load_base_params(config_path: str | Path | None = None) -> dict[str, Any]:
    path = config_path or (Path(__file__).resolve().parent.parent / "config" / "default.yaml")
    if not Path(path).exists():
        return _default_params()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return _config_to_params(cfg)


def _default_params() -> dict[str, Any]:
    from eval.run_episodes import _default_params
    return _default_params()


def shift_lambda(params: dict[str, Any], factor: float) -> dict[str, Any]:
    """Scale arrival rate by factor (e.g. 0.75 or 1.25)."""
    p = dict(params)
    p["lambda"] = p.get("lambda", 4.0) * factor
    return p


def shift_service_heavier(params: dict[str, Any]) -> dict[str, Any]:
    """Heavier-tail service times: increase mean and std."""
    p = dict(params)
    service = dict(p.get("service", {}))
    for k in list(service.keys()):
        if isinstance(service[k], dict):
            s = dict(service[k])
            s["mean"] = s.get("mean", 20) * 1.25
            s["std"] = s.get("std", 10) * 1.5
            service[k] = s
    p["service"] = service
    return p


def shift_risk_mix_more_high(params: dict[str, Any]) -> dict[str, Any]:
    """Increase fraction of critical/urgent arrivals."""
    p = dict(params)
    risk = dict(p.get("risk_mix", {"critical": 0.1, "urgent": 0.3, "low": 0.6}))
    risk["critical"] = min(0.25, risk.get("critical", 0.1) * 1.5)
    risk["urgent"] = min(0.5, risk.get("urgent", 0.3) * 1.3)
    risk["low"] = 1.0 - risk.get("critical", 0.1) - risk.get("urgent", 0.3)
    risk["low"] = max(0.25, risk["low"])
    p["risk_mix"] = risk
    return p


def shift_patience_stricter(params: dict[str, Any]) -> dict[str, Any]:
    """Lower patience (patients leave sooner)."""
    p = dict(params)
    pat = dict(p.get("patience", {"critical": 120, "urgent": 60, "low": 30}))
    for k in pat:
        if isinstance(pat[k], (int, float)):
            pat[k] = pat[k] * 0.6
    p["patience"] = pat
    return p


SHIFTS = {
    "lambda_75": lambda p: shift_lambda(p, 0.75),
    "lambda_125": lambda p: shift_lambda(p, 1.25),
    "service_heavier": shift_service_heavier,
    "risk_more_high": shift_risk_mix_more_high,
    "patience_stricter": shift_patience_stricter,
}


def run_robustness(
    mechanisms: list[tuple[str, dict[str, Any]]],
    params: dict[str, Any],
    n_episodes: int = 200,
    base_seed: int = 0,
    shifts: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run each (name, mechanism) under baseline params and under each shift.
    Returns list of rows: name, shift (or "baseline"), throughput_mean, critical_TTC_p95_mean, fitness, ...
    """
    shifts = shifts or SHIFTS
    rows: list[dict[str, Any]] = []
    fitness_weights = params.get("fitness_weights") or {}

    for name, mechanism in mechanisms:
        for shift_name, shift_fn in [("baseline", (lambda p: p))] + list(shifts.items()):
            p = shift_fn(dict(params))
            mean_m, std_m, constraint_rates, _, _ = run_episodes(
                mechanism, p, n_episodes, base_seed=base_seed
            )
            fitness = compute_fitness(mean_m, fitness_weights)
            rows.append({
                "mechanism_name": name,
                "shift": shift_name,
                "throughput_mean": mean_m.get("throughput", 0),
                "critical_TTC_p95_mean": mean_m.get("critical_TTC_p95", 0),
                "adverse_events_rate_mean": mean_m.get("adverse_events_rate", 0),
                "overload_time_mean": mean_m.get("overload_time", 0),
                "fitness": fitness,
                "missed_critical_rate_mean": mean_m.get("missed_critical_rate", 0),
            })
    return rows


def main(
    config_path: str | Path | None = None,
    results_dir: str | Path = "results",
    n_episodes: int = 200,
    base_seed: int = 0,
    evolved_path: str | Path | None = None,
) -> None:
    params = load_base_params(config_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Top baselines + optionally evolved
    mechanisms: list[tuple[str, dict]] = [
        (BASELINE_NAMES[i], BASELINES[i]) for i in range(min(3, len(BASELINES)))
    ]
    if evolved_path and Path(evolved_path).exists():
        with open(evolved_path) as f:
            mechanisms.append(("evolved_best", json.load(f)))

    rows = run_robustness(mechanisms, params, n_episodes=n_episodes, base_seed=base_seed)
    out_csv = results_dir / "robustness_results.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evolved", type=str, default=None, help="Path to best evolved mechanism JSON")
    args = p.parse_args()
    main(
        config_path=args.config,
        results_dir=args.results_dir,
        n_episodes=args.n_episodes,
        base_seed=args.seed,
        evolved_path=args.evolved,
    )
