"""Task 7.2: Baseline simulation configurations for comparison."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

from simoc.ingestion.data_structures import OCELData
from simoc.discovery.data_structures import (
    ArrivalModel,
    BehavioralProfile,
    DiscoveryResult,
    InteractionPatterns,
    TypeClassification,
)
from simoc.discovery.behavioral import _fit_continuous
from simoc.simulation.runner import SimulationRunner

logger = logging.getLogger(__name__)


def build_flat_simod_runner(
    data: OCELData,
    discovery_result: DiscoveryResult,
    behavioral_profile: BehavioralProfile,
    case_type: str = "orders",
) -> SimulationRunner:
    """Baseline 1: Flat-log Simod — single case type, no interactions.

    Only simulates the case_type using its own DFG, durations, and
    arrival model. No spawning, no interaction patterns.
    """
    # Keep only case_type models
    bp = behavioral_profile
    flat_arrivals = {}
    if case_type in bp.arrival_models:
        flat_arrivals[case_type] = bp.arrival_models[case_type]
    else:
        # Fit arrival model from birth timestamps
        flat_arrivals[case_type] = _fit_arrival_from_births(
            discovery_result.birth_death, case_type
        )

    flat_durations = {
        k: v for k, v in bp.duration_models.items() if k[0] == case_type
    }
    flat_branching = {
        k: v for k, v in bp.branching_models.items() if k == case_type
    }
    flat_dfgs = {
        k: v for k, v in bp.type_dfgs.items() if k == case_type
    }

    flat_bp = BehavioralProfile(
        arrival_models=flat_arrivals,
        duration_models=flat_durations,
        resource_pools={},
        branching_models=flat_branching,
        type_dfgs=flat_dfgs,
    )

    flat_tc = TypeClassification(
        classification={case_type: "root"},
        parent_map={},
    )

    flat_dr = DiscoveryResult(
        birth_death=discovery_result.birth_death,
        oig=discovery_result.oig,
        type_classification=flat_tc,
        spawning_profiles={},
    )

    empty_ip = InteractionPatterns({}, {}, {}, {})

    return SimulationRunner(flat_dr, flat_bp, empty_ip)


def build_independent_runner(
    data: OCELData,
    discovery_result: DiscoveryResult,
    behavioral_profile: BehavioralProfile,
) -> SimulationRunner:
    """Baseline 2: Independent per-type simulation — no interactions.

    All types treated as root with their own arrival models.
    No spawning, no synchronization, no batching, no binding, no release.
    """
    bp = behavioral_profile

    # Build arrival models for ALL types (including derived)
    all_arrivals = dict(bp.arrival_models)
    for otype in data.object_types:
        if otype not in all_arrivals and otype in bp.type_dfgs:
            all_arrivals[otype] = _fit_arrival_from_births(
                discovery_result.birth_death, otype
            )

    indep_bp = BehavioralProfile(
        arrival_models=all_arrivals,
        duration_models=bp.duration_models,
        resource_pools=bp.resource_pools,
        branching_models=bp.branching_models,
        type_dfgs=bp.type_dfgs,
    )

    indep_tc = TypeClassification(
        classification={t: "root" for t in data.object_types},
        parent_map={},
    )

    indep_dr = DiscoveryResult(
        birth_death=discovery_result.birth_death,
        oig=discovery_result.oig,
        type_classification=indep_tc,
        spawning_profiles={},
    )

    empty_ip = InteractionPatterns({}, {}, {}, {})

    return SimulationRunner(indep_dr, indep_bp, empty_ip)


def build_random_binding_runner(
    discovery_result: DiscoveryResult,
    behavioral_profile: BehavioralProfile,
    interaction_patterns: InteractionPatterns,
) -> SimulationRunner:
    """Baseline 3: Full SimOC with randomized binding policy.

    Keeps spawning, sync, batching, and release but nullifies binding
    model and constraints so binding is effectively random.
    """
    modified_ip = deepcopy(interaction_patterns)
    for key in modified_ip.binding_policies:
        policy = modified_ip.binding_policies[key]
        policy.model = None
        policy.hard_constraints = {}
        policy.feature_names = []

    return SimulationRunner(discovery_result, behavioral_profile, modified_ip)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _fit_arrival_from_births(birth_death, object_type: str) -> ArrivalModel:
    """Fit an arrival model from birth timestamps in the birth_death table."""
    bd_rows = birth_death.df[birth_death.df["object_type"] == object_type]
    timestamps = sorted(bd_rows["birth_timestamp"].tolist())

    if len(timestamps) < 2:
        dist = _fit_continuous(np.array([3600.0]))  # fallback: 1 hour
        return ArrivalModel(
            object_type=object_type,
            is_stationary=True,
            distribution=dist,
            piecewise_rates=None,
            n_arrivals=len(timestamps),
            birth_timestamps=timestamps,
        )

    gaps = np.array([
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ])
    gaps = gaps[gaps > 0]  # remove zero gaps

    if len(gaps) == 0:
        dist = _fit_continuous(np.array([3600.0]))
    else:
        dist = _fit_continuous(gaps)

    return ArrivalModel(
        object_type=object_type,
        is_stationary=True,
        distribution=dist,
        piecewise_rates=None,
        n_arrivals=len(timestamps),
        birth_timestamps=timestamps,
    )
