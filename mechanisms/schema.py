"""JSON schema and validation for mechanism genome."""

from __future__ import annotations

from typing import Any
import uuid

# Bounds for parameters (for validation and mutation)
INFO_MODES = ("none", "coarse_bins", "exact")
SERVICE_RULES = ("fifo", "severity_priority", "hybrid")
REDIRECT_MODES = ("none", "risk_cutoff", "congestion_cutoff", "combined")

# Default coarse bins (minutes)
DEFAULT_BINS = [(30, "low_wait"), (90, "medium_wait"), (180, "high_wait")]

# JSON Schema (structural) - we validate manually for flexibility
MECHANISM_SCHEMA = {
    "info_policy": {
        "info_mode": INFO_MODES,
        "bins": None,  # optional list of (upper_bound_minutes, label) when coarse_bins
    },
    "service_policy": {
        "service_rule": SERVICE_RULES,
        "params": None,  # if hybrid => weights a,b
    },
    "redirect_exit_policy": {
        "redirect_low_risk": bool,
        "redirect_mode": REDIRECT_MODES,
        "params": None,  # thresholds
        "reneging_enabled": bool,
        "patience_model": None,  # simple distribution parameters
    },
    "meta": {
        "id": str,
        "parent_ids": list,
        "generation": int,
        "seed_tag": str,
    },
}


def validate_mechanism(mechanism: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate mechanism against schema. Returns (valid, list of error messages).
    """
    errors: list[str] = []
    if not isinstance(mechanism, dict):
        return False, ["mechanism must be a dict"]

    # Info policy validation
    info_policy = mechanism.get("info_policy", {})
    if not isinstance(info_policy, dict):
        errors.append("info_policy must be a dict")
    else:
        info_mode = info_policy.get("info_mode", "none")
        if info_mode not in INFO_MODES:
            errors.append(f"info_mode must be one of {INFO_MODES}")
        if info_mode == "coarse_bins":
            bins = info_policy.get("bins")
            if not bins or not isinstance(bins, list):
                errors.append("bins required when info_mode is coarse_bins")
            else:
                # Validate bins are sorted and valid
                if len(bins) == 0:
                    errors.append("bins cannot be empty")
                else:
                    prev_upper = 0
                    for bin_def in bins:
                        if not isinstance(bin_def, (list, tuple)) or len(bin_def) != 2:
                            errors.append(
                                "each bin must be (upper_bound_minutes, label)"
                            )
                            continue
                        upper_bound, label = bin_def
                        if (
                            not isinstance(upper_bound, (int, float))
                            or upper_bound <= prev_upper
                        ):
                            errors.append(
                                "bin upper bounds must be strictly increasing"
                            )
                            break
                        prev_upper = upper_bound

    # Service policy validation
    service_policy = mechanism.get("service_policy", {})
    if not isinstance(service_policy, dict):
        errors.append("service_policy must be a dict")
    else:
        service_rule = service_policy.get("service_rule", "fifo")
        if service_rule not in SERVICE_RULES:
            errors.append(f"service_rule must be one of {SERVICE_RULES}")
        if service_rule == "hybrid":
            params = service_policy.get("params", {})
            if not isinstance(params, dict):
                errors.append("params must be a dict for hybrid service_rule")
            else:
                a = params.get("a", 1.0)
                b = params.get("b", 0.1)
                if not isinstance(a, (int, float)) or a < 0 or a > 5:
                    errors.append("hybrid weight 'a' must be in [0, 5]")
                if not isinstance(b, (int, float)) or b < 0 or b > 1:
                    errors.append("hybrid weight 'b' must be in [0, 1]")

    # Redirect/exit policy validation
    redirect_policy = mechanism.get("redirect_exit_policy", {})
    if not isinstance(redirect_policy, dict):
        errors.append("redirect_exit_policy must be a dict")
    else:
        if "redirect_low_risk" in redirect_policy and not isinstance(
            redirect_policy["redirect_low_risk"], bool
        ):
            errors.append("redirect_low_risk must be bool")

        redirect_mode = redirect_policy.get("redirect_mode", "none")
        if redirect_mode not in REDIRECT_MODES:
            errors.append(f"redirect_mode must be one of {REDIRECT_MODES}")

        if redirect_mode in ["risk_cutoff", "combined"]:
            params = redirect_policy.get("params", {})
            if not isinstance(params, dict) or "risk_threshold" not in params:
                errors.append(
                    "risk_threshold required in params for risk_cutoff/combined modes"
                )
            else:
                risk_thresh = params["risk_threshold"]
                if (
                    not isinstance(risk_thresh, (int, float))
                    or risk_thresh < 0
                    or risk_thresh > 1
                ):
                    errors.append("risk_threshold must be in [0, 1]")

        if redirect_mode in ["congestion_cutoff", "combined"]:
            params = redirect_policy.get("params", {})
            if not isinstance(params, dict) or "congestion_threshold" not in params:
                errors.append(
                    "congestion_threshold required in params for congestion_cutoff/combined modes"
                )
            else:
                cong_thresh = params["congestion_threshold"]
                if not isinstance(cong_thresh, (int, float)) or cong_thresh < 0:
                    errors.append("congestion_threshold must be >= 0")

        if "reneging_enabled" in redirect_policy and not isinstance(
            redirect_policy["reneging_enabled"], bool
        ):
            errors.append("reneging_enabled must be bool")

    # Meta validation
    meta = mechanism.get("meta", {})
    if not isinstance(meta, dict):
        errors.append("meta must be a dict")
    else:
        if "id" in meta:
            try:
                uuid.UUID(meta["id"])
            except ValueError:
                errors.append("meta.id must be a valid UUID")
        if "parent_ids" in meta and not isinstance(meta["parent_ids"], list):
            errors.append("meta.parent_ids must be a list")
        if "generation" in meta and not isinstance(meta["generation"], int):
            errors.append("meta.generation must be an int")
        if "seed_tag" in meta and not isinstance(meta["seed_tag"], str):
            errors.append("meta.seed_tag must be a string")

    return len(errors) == 0, errors


def get_bounds(field: str) -> tuple[float, float] | None:
    """Return (min, max) for numeric fields that have bounds."""
    # Define specific bounds for nested fields
    bounds_map = {
        ("info_policy", "bins", "upper_bound"): (1, 300),  # minutes
        ("service_policy", "params", "a"): (0.0, 5.0),
        ("service_policy", "params", "b"): (0.0, 1.0),
        ("redirect_exit_policy", "params", "risk_threshold"): (0.0, 1.0),
        ("redirect_exit_policy", "params", "congestion_threshold"): (0.0, 50.0),
    }

    for key_path, bounds in bounds_map.items():
        if field == ".".join(key_path):
            return bounds
    return None


def create_mechanism_id() -> str:
    """Create a new UUID for mechanism ID."""
    return str(uuid.uuid4())


def get_component_flags(mechanism: dict[str, Any]) -> dict[str, Any]:
    """Extract component flags for structure logging."""
    flags = {}

    info_policy = mechanism.get("info_policy", {})
    flags["info_mode"] = info_policy.get("info_mode", "none")

    service_policy = mechanism.get("service_policy", {})
    flags["service_rule"] = service_policy.get("service_rule", "fifo")

    redirect_policy = mechanism.get("redirect_exit_policy", {})
    flags["redirect_mode"] = redirect_policy.get("redirect_mode", "none")
    flags["reneging_enabled"] = redirect_policy.get("reneging_enabled", False)

    return flags
