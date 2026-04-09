"""Data structures for Stage 2: Object Interaction Graph and Type Classification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FittedDistribution:
    """Result of fitting a discrete distribution to cardinality samples."""

    name: str  # "poisson", "geometric", "nbinom", "empirical"
    params: dict[str, float]
    aic: float
    bic: float
    empirical_pmf: dict[int, float]
    n_samples: int
    _scipy_dist: object | None = field(default=None, repr=False)

    def rvs(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample from the fitted distribution."""
        if self._scipy_dist is not None:
            return self._scipy_dist.rvs(size=size, random_state=rng)

        # Empirical sampling
        values = np.array(list(self.empirical_pmf.keys()))
        probs = np.array(list(self.empirical_pmf.values()))
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(values, size=size, p=probs)


@dataclass
class BirthDeathTable:
    """Birth and death events for all objects (Task 2.1 output).

    DataFrame columns: object_id, object_type, birth_event_id, birth_activity,
    birth_timestamp, birth_co_objects, death_event_id, death_activity,
    death_timestamp.
    """

    df: pd.DataFrame


@dataclass
class ObjectInteractionGraph:
    """Type-level interaction graph (Task 2.2 output).

    DataFrame columns: type_1, type_2, e2o_cooccurrence, o2o_count,
    o2o_qualifier, t1_born_first_pct, t2_birth_has_t1_pct.
    """

    df: pd.DataFrame


@dataclass
class TypeClassification:
    """Root/derived classification and parent mapping (Task 2.3 output)."""

    classification: dict[str, str]  # {type_name -> "root" | "derived"}
    parent_map: dict[str, str]  # {derived_type -> parent_type}


@dataclass
class SpawningProfile:
    """Cardinality distribution for one (parent, child) spawning pair."""

    parent_type: str
    child_type: str
    raw_counts: list[int]
    fitted: FittedDistribution
    attribute_dependencies: dict[str, float] | None


@dataclass
class DiscoveryResult:
    """Aggregated output of Stage 2."""

    birth_death: BirthDeathTable
    oig: ObjectInteractionGraph
    type_classification: TypeClassification
    spawning_profiles: dict[tuple[str, str], SpawningProfile]


# ------------------------------------------------------------------
# Stage 3: Behavioral Discovery data structures
# ------------------------------------------------------------------


@dataclass
class ContinuousFittedDistribution:
    """Result of fitting a continuous distribution to duration/inter-arrival samples."""

    name: str  # "exponential", "gamma", "lognormal", "weibull", "normal", "empirical"
    params: dict[str, float]
    aic: float
    bic: float
    n_samples: int
    _scipy_dist: object | None = field(default=None, repr=False)
    _empirical_samples: np.ndarray | None = field(default=None, repr=False)

    def rvs(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample from the fitted distribution."""
        if self._scipy_dist is not None:
            return self._scipy_dist.rvs(size=size, random_state=rng)
        # Empirical fallback: resample with replacement
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(self._empirical_samples, size=size, replace=True)


@dataclass
class ArrivalModel:
    """Arrival rate model for a root object type."""

    object_type: str
    is_stationary: bool
    distribution: ContinuousFittedDistribution | None  # stationary case
    piecewise_rates: dict[str, float] | None  # non-stationary: {"HH:MM-HH:MM": rate}
    n_arrivals: int
    birth_timestamps: list[pd.Timestamp]


@dataclass
class DurationModel:
    """Activity duration model for a (type, activity) pair."""

    object_type: str
    activity: str
    distribution: ContinuousFittedDistribution
    n_observations: int
    raw_durations_seconds: list[float]


@dataclass
class ResourcePool:
    """Resource pool for an activity."""

    activity: str
    resources: list[str]


@dataclass
class BranchingModel:
    """Branching probabilities at choice points for one object type."""

    object_type: str
    transition_counts: dict[tuple[str, str], int]  # (from_act, to_act) -> count
    probabilities: dict[str, dict[str, float]]  # from_act -> {to_act -> prob}


@dataclass
class TypeDFG:
    """Per-type directly-follows graph with frequencies and probabilities."""

    object_type: str
    activities: set[str]
    edges: dict[tuple[str, str], int]  # (from_act, to_act) -> frequency
    probabilities: dict[str, dict[str, float]]  # from_act -> {to_act -> prob}


@dataclass
class BehavioralProfile:
    """Aggregated output of Stage 3."""

    arrival_models: dict[str, ArrivalModel]  # root_type -> ArrivalModel
    duration_models: dict[tuple[str, str], DurationModel]  # (type, activity) -> DurationModel
    resource_pools: dict[str, ResourcePool]  # activity -> ResourcePool
    branching_models: dict[str, BranchingModel]  # type -> BranchingModel
    type_dfgs: dict[str, TypeDFG]  # type -> TypeDFG


# ------------------------------------------------------------------
# Stage 4: Interaction Pattern Discovery data structures
# ------------------------------------------------------------------


@dataclass
class SynchronizationRule:
    """A discovered synchronization point where sibling objects converge."""

    activity: str  # e.g. "Pack Order"
    synced_type: str  # type being synced, e.g. "item"
    parent_type: str  # parent whose children sync, e.g. "order"
    condition: str  # "ALL" | "THRESHOLD(k)" | "TIME_BOUNDED"
    sync_delay_dist: ContinuousFittedDistribution
    wait_spread_dist: ContinuousFittedDistribution | None
    n_instances: int
    raw_sync_delays: list[float]  # seconds
    raw_wait_spreads: list[float]  # seconds


@dataclass
class BindingPolicy:
    """Policy governing how objects become bound mid-process."""

    source_type: str
    target_type: str
    binding_activity: str
    model: object | None  # sklearn LogisticRegression or None
    feature_names: list[str]
    capacity_dist: FittedDistribution | None
    hard_constraints: dict[str, str]
    n_instances: int
    f1_score: float | None


@dataclass
class BatchingRule:
    """A discovered batching point where unrelated objects are grouped."""

    activity: str  # e.g. "Ship"
    batched_type: str  # e.g. "order"
    trigger_type: str  # "schedule" | "threshold" | "hybrid" | "unknown"
    trigger_params: dict[str, float]
    batch_size_dist: FittedDistribution
    n_instances: int
    raw_batch_sizes: list[int]


@dataclass
class ReleaseRule:
    """A discovered point where two co-traveling types decouple."""

    type_1: str  # e.g. "order"
    type_2: str  # e.g. "item"
    release_activity: str  # e.g. "Pack Order"
    release_condition: str  # "deterministic" | "probabilistic(p=X)"
    probability: float
    n_instances: int


@dataclass
class InteractionPatterns:
    """Aggregated output of Stage 4."""

    synchronization_rules: dict[tuple[str, str], SynchronizationRule]
    binding_policies: dict[tuple[str, str], BindingPolicy]
    batching_rules: dict[tuple[str, str], BatchingRule]
    release_rules: dict[tuple[str, str], ReleaseRule]
