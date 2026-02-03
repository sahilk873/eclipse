"""Processes: arrivals, service time/patience sampling, join/redirect, deterioration, reneging."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.entities import Patient, SystemState, RISK_CRITICAL, RISK_URGENT, RISK_LOW

# Default risk score for priority (higher = more urgent)
RISK_SCORE = {RISK_CRITICAL: 3.0, RISK_URGENT: 2.0, RISK_LOW: 1.0}


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def sample_interarrival(rate_per_min: float) -> float:
    """Exponential interarrival time (minutes)."""
    if rate_per_min <= 0:
        return float("inf")
    return float(np.random.exponential(1.0 / rate_per_min))


def sample_risk_class(risk_mix: dict[str, float]) -> str:
    """Sample risk class from mix. Keys: critical, urgent, low."""
    classes = list(risk_mix.keys())
    probs = [risk_mix.get(c, 0) for c in classes]
    s = sum(probs)
    if s <= 0:
        probs = [1.0 / len(classes)] * len(classes)
    else:
        probs = [p / s for p in probs]
    return str(np.random.choice(classes, p=probs))


def sample_service_time(risk_class: str, service_params: dict[str, Any]) -> float:
    """Sample service time (minutes). Params per class: dist='lognormal', mean=..., std=... or dist='exp', rate=..."""
    params = service_params.get(risk_class, service_params.get("default", {}))
    dist = params.get("dist", "lognormal")
    if dist == "lognormal":
        mean = params.get("mean", 20)
        std = params.get("std", 10)
        # lognormal: mean = exp(mu + sigma^2/2), var = (exp(sigma^2)-1)*exp(2*mu+sigma^2)
        sigma2 = math.log(1 + (std**2) / (mean**2))
        mu = math.log(mean) - 0.5 * sigma2
        return max(0.1, float(np.random.lognormal(mu, math.sqrt(sigma2))))
    if dist == "exp":
        rate = params.get("rate", 1.0 / 20)
        return max(0.1, float(np.random.exponential(1.0 / rate)))
    return max(0.1, float(params.get("mean", 20)))


def sample_patience(risk_class: str, patience_params: dict[str, float]) -> float:
    """Exponential patience (minutes). Params: critical: 120, urgent: 60, low: 30."""
    mean = patience_params.get(risk_class, patience_params.get("default", 60))
    return max(1.0, float(np.random.exponential(mean)))


def sample_group(groups: list[str] | None = None) -> str:
    if not groups:
        return "A" if np.random.rand() < 0.5 else "B"
    return str(np.random.choice(groups))


def create_patient(
    patient_id: int,
    arrival_time: float,
    risk_mix: dict[str, float],
    service_params: dict[str, Any],
    patience_params: dict[str, float],
) -> Patient:
    """Create one patient with sampled attributes."""
    risk_class = sample_risk_class(risk_mix)
    service_time = sample_service_time(risk_class, service_params)
    patience = sample_patience(risk_class, patience_params)
    group = sample_group()
    true_risk_score = RISK_SCORE.get(risk_class, 1.0)
    return Patient(
        id=patient_id,
        arrival_time=arrival_time,
        risk_class=risk_class,
        service_time=service_time,
        patience=patience,
        group=group,
        true_risk_score=true_risk_score,
    )


def get_expected_wait_for_info(
    info_policy: dict[str, Any],
    state: SystemState,
) -> float:
    """Return E[wait] shown to patient (minutes). none -> high constant, coarse_bins -> bin center, exact -> current mean wait."""
    info_mode = info_policy.get("info_mode", "none")

    if info_mode == "none":
        return 60.0  # conservative default when no info

    if info_mode == "coarse_bins":
        bins = info_policy.get("bins", [])
        if not bins:
            return 60.0

        n = len(state.queue)
        if n == 0:
            return 0.0

        # Simple: map queue length to approximate wait time
        avg_wait_approx = n * 15.0  # rough: 15 min per person ahead

        # Find the appropriate bin
        for upper_bound, label in bins:
            if avg_wait_approx <= upper_bound:
                # Return the midpoint of the bin
                if len(bins) == 1:
                    return upper_bound / 2.0
                # Find previous bound to calculate midpoint
                idx = bins.index((upper_bound, label))
                lower_bound = 0.0 if idx == 0 else bins[idx - 1][0]
                return (lower_bound + upper_bound) / 2.0

        # If beyond all bins, return last bin's upper bound
        return float(bins[-1][0])

    if info_mode == "exact":
        if not state.queue:
            return 0.0
        # Approximate: sum of remaining service times ahead / n_servers (rough)
        n_servers = len(state.servers)
        busy = sum(1 for s in state.servers if s.busy_until > state.current_time)
        if n_servers == 0:
            return 60.0
        # Very rough: mean wait = (queue_len * mean_service) / n_servers
        mean_svc = 20.0
        return (len(state.queue) * mean_svc) / max(1, n_servers - busy)

    return 60.0


def should_admit_patient(
    mechanism: dict,
    state: SystemState,
    patient: Patient,
    benefit_per_class: dict[str, float],
    c_wait: float,
) -> bool:
    """Decide if patient is admitted (True) or redirected (False). Uses info policy and redirect policy."""
    info_policy = mechanism.get("info_policy", {})
    redirect_policy = mechanism.get("redirect_exit_policy", {})

    # Check if patient should be redirected based on redirect policy
    if redirect_policy.get("redirect_low_risk", False):
        redirect_mode = redirect_policy.get("redirect_mode", "none")
        params = redirect_policy.get("params", {})

        if redirect_mode == "risk_cutoff":
            risk_threshold = params.get("risk_threshold", 0.5)
            if patient.true_risk_score < risk_threshold:
                return False  # redirected

        elif redirect_mode == "congestion_cutoff":
            congestion_threshold = params.get("congestion_threshold", 10)
            if len(state.queue) > congestion_threshold:
                return False  # redirected

        elif redirect_mode == "combined":
            risk_threshold = params.get("risk_threshold", 0.5)
            congestion_threshold = params.get("congestion_threshold", 10)
            if (
                patient.true_risk_score < risk_threshold
                and len(state.queue) > congestion_threshold
            ):
                return False  # redirected

    # Patient utility-based admission decision
    e_wait = get_expected_wait_for_info(info_policy, state)
    benefit = benefit_per_class.get(patient.risk_class, 20)

    if benefit - c_wait * e_wait <= 0:
        return False  # patient would not join

    return True


def select_next_patient(
    queue: list[Patient], mechanism: dict, current_time: float
) -> Patient | None:
    """Select next patient from queue by mechanism's service_rule: fifo, severity_priority, hybrid."""
    if not queue:
        return None

    service_policy = mechanism.get("service_policy", {})
    service_rule = service_policy.get("service_rule", "fifo")

    if service_rule == "fifo":
        return queue[0]

    if service_rule == "severity_priority":
        # Highest risk first; within same risk, FIFO
        return max(queue, key=lambda p: (p.true_risk_score, -p.arrival_time))

    if service_rule == "hybrid":
        params = service_policy.get("params", {})
        a = params.get("a", 1.0)
        b = params.get("b", 0.1)

        def score(p: Patient) -> tuple[float, float]:
            wait = current_time - p.arrival_time
            sc = a * p.true_risk_score + b * wait
            return (sc, -p.arrival_time)

        return max(queue, key=score)

    return queue[0]


def hazard_deterioration(
    wait_time: float, delay_min: float, base: float, per_min: float
) -> float:
    """Probability of adverse event in next small interval (rate). h(t) increases after delay."""
    if wait_time < delay_min:
        return base
    return base + per_min * (wait_time - delay_min)


def sample_reneging_time(patience: float) -> float:
    """Patient reneges at arrival_time + patience (deterministic in our simple model)."""
    return patience
