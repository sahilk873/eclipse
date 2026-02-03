"""Random mutation for mechanism genome; all outputs validated."""

from __future__ import annotations

import copy
import random
from typing import Any

from mechanisms.schema import (
    validate_mechanism,
    INFO_MODES,
    SERVICE_RULES,
    REDIRECT_MODES,
    DEFAULT_BINS,
)
from mechanisms.genome import mechanism_to_dict, create_mechanism_id


def mutate_mechanism(
    mechanism: dict[str, Any],
    seed: int | None = None,
    mutation_strength: float = 1.0,
    mutation_jitter_range: float = 0.35,
) -> dict[str, Any]:
    """
    Apply one random mutation: parameter jitter, mode swap, or add/remove redirect.
    mutation_strength controls the magnitude of parameter changes.
    Returns a new validated mechanism dict.
    """
    rng = random.Random(seed)
    m = copy.deepcopy(mechanism)
    m = mechanism_to_dict(m)

    choices = [
        lambda: _jitter_params(m, rng, mutation_strength, mutation_jitter_range),
        lambda: _swap_info_mode(m, rng, mutation_strength),
        lambda: _swap_service_rule(m, rng, mutation_strength),
        lambda: _swap_redirect_mode(m, rng, mutation_strength),
        lambda: _flip_redirect_low_risk(m, rng, mutation_strength),
        lambda: _flip_reneging(m, rng, mutation_strength),
        lambda: _mutate_bins(m, rng, mutation_strength, mutation_jitter_range),
    ]
    rng.choice(choices)()

    # Update generation
    m["meta"]["generation"] = m["meta"].get("generation", 0) + 1
    m["meta"]["parent_ids"] = [mechanism["meta"].get("id", "unknown")]
    m["meta"]["id"] = create_mechanism_id()
    m["meta"]["seed_tag"] = f"mutated_{rng.randint(1000, 9999)}"

    # Ensure valid
    valid, errors = validate_mechanism(m)
    if not valid:
        # Fallback: jitter only
        _jitter_params(m, rng, min(mutation_strength, 0.5), mutation_jitter_range)
        valid, errors = validate_mechanism(m)
        if not valid:
            # Last resort: return parent with updated meta
            m = copy.deepcopy(mechanism)
            m["meta"]["generation"] = m["meta"].get("generation", 0) + 1
            m["meta"]["parent_ids"] = [mechanism["meta"].get("id", "unknown")]
            m["meta"]["id"] = create_mechanism_id()
            m["meta"]["seed_tag"] = f"fallback_{rng.randint(1000, 9999)}"

    return mechanism_to_dict(m)


def _jitter_params(
    m: dict[str, Any],
    rng: random.Random,
    strength: float = 1.0,
    jitter_range: float = 0.35,
) -> None:
    """Jitter one numeric parameter within bounds."""
    params = []

    # Service policy parameters
    service_policy = m.get("service_policy", {})
    if service_policy.get("service_rule") == "hybrid":
        params.append(("service_policy.params.a", 0.0, 5.0))
        params.append(("service_policy.params.b", 0.0, 1.0))

    # Redirect policy parameters
    redirect_policy = m.get("redirect_exit_policy", {})
    redirect_mode = redirect_policy.get("redirect_mode", "none")
    if redirect_mode in ["risk_cutoff", "combined"]:
        params.append(("redirect_exit_policy.params.risk_threshold", 0.0, 1.0))
    if redirect_mode in ["congestion_cutoff", "combined"]:
        params.append(("redirect_exit_policy.params.congestion_threshold", 0.0, 50.0))

    if not params:
        return

    param_path, lo, hi = rng.choice(params)
    keys = param_path.split(".")

    # Navigate to the parameter
    current = m
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    param_name = keys[-1]

    # Get current value or set random
    current_val = current.get(param_name, rng.uniform(lo, hi))

    # Apply jitter scaled by strength and jitter_range
    delta = rng.uniform(-jitter_range, jitter_range) * strength * (hi - lo)
    new_val = max(lo, min(hi, current_val + delta))
    current[param_name] = round(new_val, 3)


def _swap_info_mode(
    m: dict[str, Any], rng: random.Random, strength: float = 1.0
) -> None:
    """Swap information mode."""
    current_mode = m.get("info_policy", {}).get("info_mode", "none")
    modes = [x for x in INFO_MODES if x != current_mode]
    if not modes:
        return
    m["info_policy"]["info_mode"] = rng.choice(modes)

    if m["info_policy"]["info_mode"] == "coarse_bins":
        # Random bins
        n_bins = rng.randint(2, 4)
        max_wait = rng.uniform(60, 180)
        bins = []
        for i in range(n_bins):
            upper_bound = (i + 1) * (max_wait / n_bins)
            label = f"bin_{i + 1}"
            bins.append((round(upper_bound, 1), label))
        m["info_policy"]["bins"] = bins


def _swap_service_rule(
    m: dict[str, Any], rng: random.Random, strength: float = 1.0
) -> None:
    """Swap service rule."""
    current_rule = m.get("service_policy", {}).get("service_rule", "fifo")
    rules = [x for x in SERVICE_RULES if x != current_rule]
    if not rules:
        return

    m["service_policy"]["service_rule"] = rng.choice(rules)

    if m["service_policy"]["service_rule"] == "hybrid":
        m["service_policy"]["params"] = {
            "a": round(rng.uniform(0.5, 3.0), 2),
            "b": round(rng.uniform(0.01, 0.5), 2),
        }


def _swap_redirect_mode(
    m: dict[str, Any], rng: random.Random, strength: float = 1.0
) -> None:
    """Swap redirect mode."""
    current_mode = m.get("redirect_exit_policy", {}).get("redirect_mode", "none")
    modes = [x for x in REDIRECT_MODES if x != current_mode]
    if not modes:
        return

    m["redirect_exit_policy"]["redirect_mode"] = rng.choice(modes)

    # Set default params for new mode if needed
    redirect_mode = m["redirect_exit_policy"]["redirect_mode"]
    params = {}
    if redirect_mode in ["risk_cutoff", "combined"]:
        params["risk_threshold"] = round(rng.uniform(0.1, 0.8), 2)
    if redirect_mode in ["congestion_cutoff", "combined"]:
        params["congestion_threshold"] = round(rng.uniform(5, 25), 1)

    if params:
        m["redirect_exit_policy"]["params"] = params


def _flip_redirect_low_risk(
    m: dict[str, Any], rng: random.Random, strength: float = 1.0
) -> None:
    """Toggle redirect low risk setting."""
    current = m.get("redirect_exit_policy", {}).get("redirect_low_risk", False)
    m["redirect_exit_policy"]["redirect_low_risk"] = not current


def _flip_reneging(
    m: dict[str, Any], rng: random.Random, strength: float = 1.0
) -> None:
    """Toggle reneging enabled setting."""
    current = m.get("redirect_exit_policy", {}).get("reneging_enabled", True)
    m["redirect_exit_policy"]["reneging_enabled"] = not current


def _mutate_bins(
    m: dict[str, Any],
    rng: random.Random,
    strength: float = 1.0,
    jitter_range: float = 0.35,
) -> None:
    """Mutate bin definitions if coarse_bins mode."""
    if m.get("info_policy", {}).get("info_mode") != "coarse_bins":
        return

    bins = m["info_policy"].get("bins", [])
    if not bins:
        return

    # Either add/remove bin or modify existing bin
    if rng.random() < 0.3 and len(bins) > 2:
        # Remove a bin
        idx = rng.randint(0, len(bins) - 1)
        bins.pop(idx)
    elif rng.random() < 0.3 and len(bins) < 5:
        # Add a bin
        max_bound = max(b[0] for b in bins)
        new_bound = max_bound + rng.uniform(20, 60)
        bins.append((round(new_bound, 1), f"bin_{len(bins) + 1}"))
        bins.sort(key=lambda x: x[0])
    else:
        # Modify existing bin
        idx = rng.randint(0, len(bins) - 1)
        upper_bound, label = bins[idx]

        # Jitter the bound
        lower_bound = 0.0 if idx == 0 else bins[idx - 1][0]
        upper_limit = bins[idx + 1][0] if idx < len(bins) - 1 else upper_bound + 60

        delta = rng.uniform(-jitter_range, jitter_range) * strength * (upper_limit - lower_bound)
        new_bound = max(lower_bound + 1, min(upper_limit - 1, upper_bound + delta))
        bins[idx] = (round(new_bound, 1), label)

        # Sort bins
        bins.sort(key=lambda x: x[0])

    m["info_policy"]["bins"] = bins
