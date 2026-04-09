"""Data structures for Stage 5: Simulation Engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    duration: float  # simulation end time in seconds
    seed: int = 42
    max_sync_wait: float = 86400.0  # deadlock timeout (1 day)
    batch_timeout: float = 7200.0  # max batch accumulation wait


@dataclass
class SimulatedEvent:
    """A single event produced by the simulation."""

    event_id: str
    activity: str
    timestamp: float  # seconds from simulation start
    objects: list[tuple[str, str]]  # [(object_id, object_type), ...]


@dataclass
class SimulatedObject:
    """An object created during the simulation."""

    object_id: str
    object_type: str


@dataclass
class SimulatedO2O:
    """An O2O relation created during the simulation."""

    source_id: str
    source_type: str
    target_id: str
    target_type: str
    qualifier: str


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""

    events: list[SimulatedEvent]
    objects: list[SimulatedObject]
    o2o_relations: list[SimulatedO2O]
    config: SimulationConfig
