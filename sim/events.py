"""Event types and priority-queue logic for discrete-event simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Use a total ordering: (time, event_type_rank, tiebreaker) so same-time events
# have deterministic order (e.g. EndService before Arrival before Reneging).
EVENT_RANK_END_EPISODE = 0
EVENT_RANK_END_SERVICE = 1
EVENT_RANK_START_SERVICE = 2
EVENT_RANK_ARRIVAL = 3
EVENT_RANK_DETERIORATION = 4
EVENT_RANK_RENEGING = 5


class EventType(str, Enum):
    ARRIVAL = "arrival"
    START_SERVICE = "start_service"
    END_SERVICE = "end_service"
    DETERIORATION_CHECK = "deterioration_check"
    RENEGING = "reneging"
    END_EPISODE = "end_episode"


def _event_rank(t: EventType) -> int:
    if t == EventType.END_EPISODE:
        return EVENT_RANK_END_EPISODE
    if t == EventType.END_SERVICE:
        return EVENT_RANK_END_SERVICE
    if t == EventType.START_SERVICE:
        return EVENT_RANK_START_SERVICE
    if t == EventType.ARRIVAL:
        return EVENT_RANK_ARRIVAL
    if t == EventType.DETERIORATION_CHECK:
        return EVENT_RANK_DETERIORATION
    if t == EventType.RENEGING:
        return EVENT_RANK_RENEGING
    return 99


@dataclass(order=True)
class Event:
    """Single event for the DES. Ordered by (time, rank, tiebreaker)."""

    time: float
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False, default=None)
    rank: int = field(compare=True, default=99)
    tiebreaker: int = field(compare=True, default=0)

    def __post_init__(self) -> None:
        if self.rank == 99:
            self.rank = _event_rank(self.event_type)

    @classmethod
    def at(cls, time: float, event_type: EventType, payload: Any = None, tiebreaker: int = 0) -> Event:
        return cls(
            time=time,
            event_type=event_type,
            payload=payload,
            rank=_event_rank(event_type),
            tiebreaker=tiebreaker,
        )
