"""Mechanism genome: typed structure and JSON conversion for new schema."""

from __future__ import annotations

import random
from typing import Any

from mechanisms.schema import (
    INFO_MODES,
    SERVICE_RULES,
    REDIRECT_MODES,
    DEFAULT_BINS,
    create_mechanism_id,
    validate_mechanism,
)


def mechanism_to_dict(mechanism: dict[str, Any]) -> dict[str, Any]:
    """Export mechanism to JSON-serializable dict (for LLM / saving)."""
    if not mechanism:
        return _default_mechanism()

    out = dict(mechanism)

    # Ensure nested structure exists
    if "info_policy" not in out:
        out["info_policy"] = {}
    if "service_policy" not in out:
        out["service_policy"] = {}
    if "redirect_exit_policy" not in out:
        out["redirect_exit_policy"] = {}
    if "meta" not in out:
        out["meta"] = {}

    # Set defaults for info_policy
    info_policy = out["info_policy"]
    info_policy.setdefault("info_mode", "none")
    if info_policy["info_mode"] == "coarse_bins" and "bins" not in info_policy:
        info_policy["bins"] = DEFAULT_BINS
    # Normalize bins to tuples (JSON/LLM returns lists; sim expects tuples)
    if "bins" in info_policy and info_policy["bins"]:
        info_policy["bins"] = [
            (b[0], b[1]) if isinstance(b, (list, tuple)) and len(b) >= 2 else b
            for b in info_policy["bins"]
        ]

    # Set defaults for service_policy
    service_policy = out["service_policy"]
    service_policy.setdefault("service_rule", "fifo")
    if service_policy["service_rule"] == "hybrid" and "params" not in service_policy:
        service_policy["params"] = {"a": 1.0, "b": 0.1}

    # Set defaults for redirect_exit_policy
    redirect_policy = out["redirect_exit_policy"]
    redirect_policy.setdefault("redirect_low_risk", False)
    redirect_policy.setdefault("redirect_mode", "none")
    redirect_policy.setdefault("reneging_enabled", True)
    if (
        redirect_policy["redirect_mode"] not in ["none"]
        and "params" not in redirect_policy
    ):
        redirect_policy["params"] = {}

    # Set defaults for meta
    meta = out["meta"]
    meta.setdefault("id", create_mechanism_id())
    meta.setdefault("parent_ids", [])
    meta.setdefault("generation", 0)
    meta.setdefault("seed_tag", "generated")

    return out


def dict_to_mechanism(data: dict[str, Any]) -> dict[str, Any]:
    """Build mechanism from JSON/dict; fill defaults and normalize keys."""
    m = dict(data)
    valid, errs = validate_mechanism(m)
    if not valid:
        # Apply defaults for missing/invalid so sim can still run
        return _default_mechanism()
    return mechanism_to_dict(m)


def _default_mechanism() -> dict[str, Any]:
    """Create a default mechanism with all required fields."""
    return {
        "info_policy": {"info_mode": "none"},
        "service_policy": {"service_rule": "fifo"},
        "redirect_exit_policy": {
            "redirect_low_risk": False,
            "redirect_mode": "none",
            "reneging_enabled": True,
        },
        "meta": {
            "id": create_mechanism_id(),
            "parent_ids": [],
            "generation": 0,
            "seed_tag": "default",
        },
    }


def random_mechanism(seed: int | None = None) -> dict[str, Any]:
    """Generate a random valid mechanism (for evolution init)."""
    if seed is not None:
        random.seed(seed)

    # Random info policy
    info_mode = random.choice(INFO_MODES)
    info_policy = {"info_mode": info_mode}
    if info_mode == "coarse_bins":
        # Random bins
        n_bins = random.randint(2, 4)
        max_wait = random.uniform(60, 180)
        bins = []
        for i in range(n_bins):
            upper_bound = (i + 1) * (max_wait / n_bins)
            label = f"bin_{i + 1}"
            bins.append((round(upper_bound, 1), label))
        info_policy["bins"] = bins

    # Random service policy
    service_rule = random.choice(SERVICE_RULES)
    service_policy = {"service_rule": service_rule}
    if service_rule == "hybrid":
        service_policy["params"] = {
            "a": round(random.uniform(0.5, 3.0), 2),
            "b": round(random.uniform(0.01, 0.5), 2),
        }

    # Random redirect policy
    redirect_low_risk = random.choice([True, False])
    redirect_mode = random.choice(REDIRECT_MODES)
    redirect_policy = {
        "redirect_low_risk": redirect_low_risk,
        "redirect_mode": redirect_mode,
        "reneging_enabled": random.choice([True, False]),
    }

    if redirect_mode in ["risk_cutoff", "combined"]:
        redirect_policy["params"] = {
            "risk_threshold": round(random.uniform(0.1, 0.8), 2)
        }

    if redirect_mode in ["congestion_cutoff", "combined"]:
        if "params" not in redirect_policy:
            redirect_policy["params"] = {}
        redirect_policy["params"]["congestion_threshold"] = round(
            random.uniform(5, 25), 1
        )

    mechanism = {
        "info_policy": info_policy,
        "service_policy": service_policy,
        "redirect_exit_policy": redirect_policy,
        "meta": {
            "id": create_mechanism_id(),
            "parent_ids": [],
            "generation": 0,
            "seed_tag": "random",
        },
    }

    return mechanism_to_dict(mechanism)
