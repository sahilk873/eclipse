"""Episode runner: discrete-event loop with FIFO and severity priority."""

from __future__ import annotations

import heapq
from typing import Any

from sim.entities import (
    Patient,
    Server,
    SystemState,
    MetricsAccumulator,
    RISK_CRITICAL,
    STATUS_LEFT,
    STATUS_REDIRECTED,
)
from sim.events import Event, EventType
import numpy as np
from sim.processes import (
    set_seed,
    sample_interarrival,
    create_patient,
    should_admit_patient,
    select_next_patient,
    hazard_deterioration,
)


def run_episode(
    mechanism: dict,
    params: dict,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run one episode of length T minutes.
    Returns (metrics_dict, constraints_violated).
    """
    set_seed(seed)
    T = params.get("T", 480)
    rate_per_min = params.get("lambda", 4.0) / 60.0
    n_servers = params.get("n_servers", 3)
    Qmax = params.get("Qmax", 20)
    risk_mix = params.get("risk_mix", {"critical": 0.1, "urgent": 0.3, "low": 0.6})
    service_params = params.get("service", {})
    patience_params = params.get("patience", {})
    benefit_per_class = params.get(
        "benefit_per_class", {"critical": 100, "urgent": 50, "low": 20}
    )
    c_wait = params.get("c_wait", 0.5)
    deterioration_cfg = params.get("deterioration", {})
    deterioration_enabled = deterioration_cfg.get("enabled", False)
    delay_min = deterioration_cfg.get("critical_delay_min", 15)
    hazard_base = deterioration_cfg.get("hazard_base", 0.01)
    hazard_per_min = deterioration_cfg.get("hazard_per_min", 0.002)
    redirect_policy = mechanism.get("redirect_exit_policy", {})
    reneging_enabled = redirect_policy.get("reneging_enabled", True)
    last_deterioration_time = 0.0

    servers = [Server(id=i) for i in range(n_servers)]
    metrics = MetricsAccumulator()
    state = SystemState(servers=servers, current_time=0.0, metrics=metrics, Qmax=Qmax)

    # Event list: (time, rank, tiebreaker, event)
    event_id = [0]

    def push(ev: Event) -> None:
        event_id[0] += 1
        ev.tiebreaker = event_id[0]
        heapq.heappush(events, (ev.time, ev.rank, ev.tiebreaker, ev))

    events: list[tuple[float, int, int, Event]] = []
    # End episode
    push(Event.at(T, EventType.END_EPISODE))
    # First arrival
    t_next = sample_interarrival(rate_per_min)
    patient_id = [0]

    def schedule_next_arrival() -> None:
        patient_id[0] += 1
        pid = patient_id[0]
        arr_ev = Event.at(
            state.current_time + sample_interarrival(rate_per_min),
            EventType.ARRIVAL,
            {"patient_id": pid},
        )
        push(arr_ev)

    push(Event.at(t_next, EventType.ARRIVAL, {"patient_id": 1}))

    while events:
        _, _, _, ev = heapq.heappop(events)
        t = ev.time
        if t > T:
            continue
        state.current_time = t

        # Deterioration: for waiting critical patients, sample adverse event
        if deterioration_enabled and state.queue:
            dt = t - last_deterioration_time
            if dt > 0:
                for p in list(state.queue):
                    if p.risk_class == RISK_CRITICAL and not p.adverse_event:
                        wait_time = t - p.arrival_time
                        h = hazard_deterioration(
                            wait_time, delay_min, hazard_base, hazard_per_min
                        )
                        prob = min(1.0, h * dt)
                        if np.random.rand() < prob:
                            p.adverse_event = True
                            metrics.record_adverse_event()
                last_deterioration_time = t

        if ev.event_type == EventType.END_EPISODE:
            metrics.finalize_overload(t, Qmax)
            break

        if ev.event_type == EventType.ARRIVAL:
            schedule_next_arrival()
            pid = ev.payload["patient_id"]
            patient = create_patient(pid, t, risk_mix, service_params, patience_params)
            admit = should_admit_patient(
                mechanism, state, patient, benefit_per_class, c_wait
            )
            if not admit:
                patient.status = STATUS_REDIRECTED
                state.all_patients.append(patient)
                metrics.record_redirected(patient)
                continue
            state.add_to_queue(patient)
            if reneging_enabled and patient.patience < 1e6:
                renege_at = t + patient.patience
                if renege_at <= T:
                    push(
                        Event.at(
                            renege_at, EventType.RENEGING, {"patient_id": patient.id}
                        )
                    )
            # Start service on free servers if queue non-empty
            for server in state.servers:
                if server.busy_until <= state.current_time and state.queue:
                    next_p = select_next_patient(
                        state.queue, mechanism, state.current_time
                    )
                    if next_p is not None:
                        state.remove_from_queue(next_p)
                        next_p.wait_time = state.current_time - next_p.arrival_time
                        end_time = state.current_time + next_p.service_time
                        server.busy_until = end_time
                        server.current_patient_id = next_p.id
                        next_p.status = "served"
                        next_p.time_to_care = state.current_time - next_p.arrival_time
                        metrics.record_service_start(next_p, state.current_time)
                        metrics.update_overload(
                            state.current_time, len(state.queue), Qmax
                        )
                        push(
                            Event.at(
                                end_time,
                                EventType.END_SERVICE,
                                {"server_id": server.id},
                            )
                        )
            continue

        if ev.event_type == EventType.RENEGING:
            pid = ev.payload["patient_id"]
            for p in state.queue:
                if p.id == pid:
                    p.status = STATUS_LEFT
                    state.remove_from_queue(p)
                    metrics.record_left(p)
                    break
            continue

        if ev.event_type == EventType.END_SERVICE:
            server_id = ev.payload["server_id"]
            server = state.servers[server_id]
            state.free_server(server)
            # Select next from queue by mechanism
            next_p = select_next_patient(state.queue, mechanism, state.current_time)
            if next_p is not None:
                state.remove_from_queue(next_p)
                next_p.wait_time = state.current_time - next_p.arrival_time
                end_time = state.current_time + next_p.service_time
                server.busy_until = end_time
                server.current_patient_id = next_p.id
                next_p.status = "served"
                next_p.time_to_care = state.current_time - next_p.arrival_time
                metrics.record_service_start(next_p, state.current_time)
                metrics.update_overload(state.current_time, len(state.queue), Qmax)
                push(
                    Event.at(end_time, EventType.END_SERVICE, {"server_id": server_id})
                )
            continue

        if ev.event_type == EventType.START_SERVICE:
            server_id = ev.payload["server_id"]
            patient = ev.payload["patient"]
            server = state.servers[server_id]
            end_time = state.current_time + patient.service_time
            server.busy_until = end_time
            server.current_patient_id = patient.id
            patient.wait_time = state.current_time - patient.arrival_time
            patient.status = "served"
            patient.time_to_care = state.current_time - patient.arrival_time
            metrics.record_service_start(patient, state.current_time)
            state.remove_from_queue(patient)
            metrics.update_overload(state.current_time, len(state.queue), Qmax)
            push(Event.at(end_time, EventType.END_SERVICE, {"server_id": server_id}))
            continue

    # After loop: start any remaining queue with free servers (should have been handled by END_SERVICE)
    # We don't start new service in END_EPISODE; episode just ends. So we're done.

    # Build metrics dict
    n_critical_total = (
        metrics.critical_served + metrics.critical_redirected + metrics.critical_left
    )
    n_critical_missed = metrics.critical_redirected + metrics.critical_left
    missed_critical_rate = n_critical_missed / max(1, n_critical_total)

    critical_ttc_mean = (
        float(sum(metrics.critical_ttc_list) / len(metrics.critical_ttc_list))
        if metrics.critical_ttc_list
        else 0.0
    )
    critical_ttc_p95 = (
        float(
            sorted(metrics.critical_ttc_list)[
                int(0.95 * len(metrics.critical_ttc_list))
            ]
        )
        if len(metrics.critical_ttc_list) >= 20
        else critical_ttc_mean
    )
    if not metrics.critical_ttc_list:
        critical_ttc_p95 = 0.0

    noncritical_waits = metrics.noncritical_wait_times
    mean_wait = (
        float(sum(noncritical_waits) / len(noncritical_waits))
        if noncritical_waits
        else 0.0
    )
    p95_wait = (
        float(sorted(noncritical_waits)[int(0.95 * len(noncritical_waits))])
        if len(noncritical_waits) >= 20
        else mean_wait
    )

    throughput_per_hour = metrics.total_served / (T / 60.0) if T > 0 else 0.0
    utilization = 0.0  # could compute from server busy time
    adverse_rate = metrics.adverse_events_count / max(
        1, metrics.total_served + n_critical_missed
    )

    metrics_dict = {
        "critical_served": metrics.critical_served,
        "critical_redirected": metrics.critical_redirected,
        "critical_left": metrics.critical_left,
        "missed_critical_rate": missed_critical_rate,
        "critical_TTC_mean": critical_ttc_mean,
        "critical_TTC_p95": critical_ttc_p95,
        "adverse_events_count": metrics.adverse_events_count,
        "adverse_events_rate": adverse_rate,
        "throughput": throughput_per_hour,
        "total_served": metrics.total_served,
        "mean_wait": mean_wait,
        "p95_wait": p95_wait,
        "clinician_utilization": utilization,
        "overload_time": metrics.overload_time,
    }

    # Constraints (to be checked by eval)
    constraints = params.get("constraints", {})
    epsilon = constraints.get("missed_critical_epsilon", 0.02)
    T_crit = constraints.get("critical_TTC_minutes", 30)
    exceed_pct = constraints.get("critical_TTC_exceed_pct", 0.05)
    n_over_T_crit = sum(1 for x in metrics.critical_ttc_list if x > T_crit)
    pct_over_T_crit = n_over_T_crit / max(1, len(metrics.critical_ttc_list))

    constraints_violated = {
        "missed_critical_rate": missed_critical_rate > epsilon,
        "critical_TTC_exceeded": pct_over_T_crit > exceed_pct,
    }

    return metrics_dict, constraints_violated
