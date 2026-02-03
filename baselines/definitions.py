"""Seven baseline mechanism definitions (valid against new schema)."""

from typing import Any

from mechanisms.schema import create_mechanism_id, validate_mechanism

# 1. FIFO + always admit + exact wait info
BASELINE_1: dict[str, Any] = {
    "info_policy": {"info_mode": "exact"},
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
        "seed_tag": "baseline",
    },
}

# 2. FIFO + always admit + no wait info
BASELINE_2: dict[str, Any] = {
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
        "seed_tag": "baseline",
    },
}

# 3. Severity priority + always admit + exact info
BASELINE_3: dict[str, Any] = {
    "info_policy": {"info_mode": "exact"},
    "service_policy": {"service_rule": "severity_priority"},
    "redirect_exit_policy": {
        "redirect_low_risk": False,
        "redirect_mode": "none",
        "reneging_enabled": True,
    },
    "meta": {
        "id": create_mechanism_id(),
        "parent_ids": [],
        "generation": 0,
        "seed_tag": "baseline",
    },
}

# 4. Severity priority + always admit + no info
BASELINE_4: dict[str, Any] = {
    "info_policy": {"info_mode": "none"},
    "service_policy": {"service_rule": "severity_priority"},
    "redirect_exit_policy": {
        "redirect_low_risk": False,
        "redirect_mode": "none",
        "reneging_enabled": True,
    },
    "meta": {
        "id": create_mechanism_id(),
        "parent_ids": [],
        "generation": 0,
        "seed_tag": "baseline",
    },
}

# 5. Hybrid priority (fixed weights) + always admit + coarse info
BASELINE_5: dict[str, Any] = {
    "info_policy": {
        "info_mode": "coarse_bins",
        "bins": [(30, "low_wait"), (90, "medium_wait"), (180, "high_wait")],
    },
    "service_policy": {"service_rule": "hybrid", "params": {"a": 1.0, "b": 0.1}},
    "redirect_exit_policy": {
        "redirect_low_risk": False,
        "redirect_mode": "none",
        "reneging_enabled": True,
    },
    "meta": {
        "id": create_mechanism_id(),
        "parent_ids": [],
        "generation": 0,
        "seed_tag": "baseline",
    },
}

# 6. Risk-threshold gating (ESI-like): redirect low risk when load high, no exact wait info
BASELINE_6: dict[str, Any] = {
    "info_policy": {"info_mode": "none"},
    "service_policy": {"service_rule": "severity_priority"},
    "redirect_exit_policy": {
        "redirect_low_risk": True,
        "redirect_mode": "combined",
        "params": {"risk_threshold": 0.3, "congestion_threshold": 15},
        "reneging_enabled": True,
    },
    "meta": {
        "id": create_mechanism_id(),
        "parent_ids": [],
        "generation": 0,
        "seed_tag": "baseline",
    },
}

# 7. FAHP-MAUT-inspired: composite utility of acuity + wait (Ashour & Kremer 2012).
# Approximates their multi-attribute utility with hybrid rule: score = a*risk + b*wait.
BASELINE_7: dict[str, Any] = {
    "info_policy": {"info_mode": "exact"},
    "service_policy": {"service_rule": "hybrid", "params": {"a": 2.0, "b": 0.15}},
    "redirect_exit_policy": {
        "redirect_low_risk": False,
        "redirect_mode": "none",
        "reneging_enabled": True,
    },
    "meta": {
        "id": create_mechanism_id(),
        "parent_ids": [],
        "generation": 0,
        "seed_tag": "baseline",
    },
}

BASELINES: list[dict[str, Any]] = [
    BASELINE_1,
    BASELINE_2,
    BASELINE_3,
    BASELINE_4,
    BASELINE_5,
    BASELINE_6,
    BASELINE_7,
]

# Validate all baselines
for i, baseline in enumerate(BASELINES):
    valid, errors = validate_mechanism(baseline)
    if not valid:
        raise ValueError(f"Baseline {i} invalid: {errors}")

BASELINE_NAMES = [
    "FIFO_always_admit_exact_info",
    "FIFO_always_admit_no_info",
    "Severity_always_admit_exact_info",
    "Severity_always_admit_no_info",
    "Hybrid_always_admit_coarse_info",
    "Risk_threshold_gating_no_exact_info",
    "FAHP_MAUT_inspired",
]


def get_baseline(index: int) -> dict[str, Any]:
    """Return baseline mechanism by index (0..6)."""
    if 0 <= index < len(BASELINES):
        return BASELINES[index]
    return BASELINES[0]


def get_baseline_name(index: int) -> str:
    if 0 <= index < len(BASELINE_NAMES):
        return BASELINE_NAMES[index]
    return f"baseline_{index}"
