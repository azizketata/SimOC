"""Shared fixtures for evaluation tests."""

from pathlib import Path

import pytest

from simoc.ingestion import OCELData, load_ocel
from simoc.discovery.interaction_graph import (
    compute_birth_death, build_oig, classify_types,
)
from simoc.discovery.cardinality import compute_spawning_profiles
from simoc.discovery.behavioral import discover_behavioral
from simoc.discovery.patterns import discover_patterns
from simoc.discovery.data_structures import (
    BehavioralProfile, DiscoveryResult, InteractionPatterns,
)
from simoc.simulation.runner import SimulationRunner
from simoc.simulation.data_structures import SimulationResult
from simoc.evaluation._helpers import sim_result_to_oceldata

FIXTURE_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="session")
def loaded_data() -> OCELData:
    return load_ocel(str(FIXTURE_DIR / "sample_order_process.json"))


@pytest.fixture(scope="session")
def discovery_result(loaded_data: OCELData) -> DiscoveryResult:
    bd = compute_birth_death(loaded_data)
    oig = build_oig(loaded_data, bd)
    tc = classify_types(oig, loaded_data.object_types)
    sp = compute_spawning_profiles(loaded_data, bd, tc)
    return DiscoveryResult(birth_death=bd, oig=oig,
                           type_classification=tc, spawning_profiles=sp)


@pytest.fixture(scope="session")
def behavioral_profile(loaded_data, discovery_result) -> BehavioralProfile:
    return discover_behavioral(
        loaded_data, discovery_result.birth_death,
        discovery_result.type_classification,
    )


@pytest.fixture(scope="session")
def interaction_patterns(loaded_data, discovery_result) -> InteractionPatterns:
    return discover_patterns(
        loaded_data, discovery_result.birth_death,
        discovery_result.type_classification,
    )


@pytest.fixture(scope="session")
def sim_runner(discovery_result, behavioral_profile, interaction_patterns):
    return SimulationRunner(discovery_result, behavioral_profile, interaction_patterns)


@pytest.fixture(scope="session")
def sim_result(sim_runner) -> SimulationResult:
    return sim_runner.run(duration=36000, seed=42)


@pytest.fixture(scope="session")
def synthetic_data(sim_result) -> OCELData:
    return sim_result_to_oceldata(sim_result)
