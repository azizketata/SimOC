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
