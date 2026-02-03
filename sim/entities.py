"""Simulator entities: Patient, Server, SystemState."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Risk classes
RISK_CRITICAL = "critical"
RISK_URGENT = "urgent"
RISK_LOW = "low"
RISK_CLASSES = (RISK_CRITICAL, RISK_URGENT, RISK_LOW)

# Patient status
STATUS_WAITING = "waiting"
STATUS_SERVED = "served"
STATUS_REDIRECTED = "redirected"
STATUS_LEFT = "left"


class RiskClass(str, Enum):
    CRITICAL = RISK_CRITICAL
    URGENT = RISK_URGENT
    LOW = RISK_LOW


class PatientStatus(str, Enum):
    WAITING = STATUS_WAITING
    SERVED = STATUS_SERVED
    REDIRECTED = STATUS_REDIRECTED
    LEFT = STATUS_LEFT


@dataclass
class Patient:
    """A patient in the system."""

    id: int
    arrival_time: float
    risk_class: str  # critical, urgent, low
    service_time: float
    patience: float  # max wait before reneging (minutes)
    group: str = "A"  # optional A/B for fairness
    true_risk_score: float = 0.0  # numeric for priority; set from risk_class
    status: str = STATUS_WAITING
    adverse_event: bool = False
    time_to_care: float | None = None  # set when service starts
    wait_time: float = 0.0  # time spent in queue
    reneging_time: float | None = None  # when they would renege (scheduled)

    def __hash__(self) -> int:
        return hash(self.id)

    def __lt__(self, other: Patient) -> bool:
        """For priority queue: lower id breaks ties."""
        return self.id < other.id


@dataclass
class Server:
    """A clinician/server."""

    id: int
    busy_until: float = 0.0  # time when current service ends
    current_patient_id: int | None = None

    @property
    def is_busy(self) -> bool:
        return self.busy_until > 0


@dataclass
class MetricsAccumulator:
    """Accumulates counts and lists for episode metrics."""

    critical_served: int = 0
    critical_redirected: int = 0
    critical_left: int = 0
    critical_ttc_list: list[float] = field(default_factory=list)
    adverse_events_count: int = 0
    total_served: int = 0
    noncritical_wait_times: list[float] = field(default_factory=list)
    overload_time: float = 0.0  # minutes with queue length > Qmax
    last_queue_len: int = 0
    last_overload_time: float = 0.0  # when overload started (for cumulative)

    def record_service_start(self, patient: Patient, current_time: float) -> None:
        if patient.risk_class == RISK_CRITICAL:
            self.critical_served += 1
            ttc = current_time - patient.arrival_time
            self.critical_ttc_list.append(ttc)
        self.total_served += 1
        if patient.wait_time > 0 and patient.risk_class != RISK_CRITICAL:
            self.noncritical_wait_times.append(patient.wait_time)

    def record_redirected(self, patient: Patient) -> None:
        if patient.risk_class == RISK_CRITICAL:
            self.critical_redirected += 1

    def record_left(self, patient: Patient) -> None:
        if patient.risk_class == RISK_CRITICAL:
            self.critical_left += 1

    def record_adverse_event(self) -> None:
        self.adverse_events_count += 1

    def update_overload(self, current_time: float, queue_len: int, Qmax: int) -> None:
        if queue_len > Qmax:
            if self.last_queue_len <= Qmax:
                self.last_overload_time = current_time
            self.last_queue_len = queue_len
        else:
            if self.last_queue_len > Qmax:
                self.overload_time += current_time - self.last_overload_time
            self.last_queue_len = queue_len

    def finalize_overload(self, current_time: float, Qmax: int) -> None:
        if self.last_queue_len > Qmax:
            self.overload_time += current_time - self.last_overload_time


@dataclass
class SystemState:
    """Current state of the system."""

    queue: list[Patient] = field(default_factory=list)  # ordered; next to serve per policy
    servers: list[Server] = field(default_factory=list)
    current_time: float = 0.0
    metrics: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    all_patients: list[Patient] = field(default_factory=list)  # for metrics on redirected/left
    Qmax: int = 20

    def add_to_queue(self, patient: Patient) -> None:
        patient.status = STATUS_WAITING
        self.queue.append(patient)
        self.all_patients.append(patient)
        self.metrics.update_overload(self.current_time, len(self.queue), self.Qmax)

    def remove_from_queue(self, patient: Patient) -> None:
        self.queue = [p for p in self.queue if p.id != patient.id]
        self.metrics.update_overload(self.current_time, len(self.queue), self.Qmax)

    def free_server(self, server: Server) -> None:
        server.busy_until = 0.0
        server.current_patient_id = None

    def assign_patient(self, server: Server, patient: Patient, end_time: float) -> None:
        server.busy_until = end_time
        server.current_patient_id = patient.id
        patient.status = STATUS_SERVED
        patient.time_to_care = self.current_time - patient.arrival_time
        self.metrics.record_service_start(patient, self.current_time)
        self.remove_from_queue(patient)
