"""Shared fixtures for output tests."""

from pathlib import Path

import pytest

from simoc.ingestion import load_ocel, OCELData
from simoc.discovery.interaction_graph import compute_birth_death, build_oig, classify_types
from simoc.discovery.cardinality import compute_spawning_profiles
from simoc.discovery.behavioral import discover_behavioral
from simoc.discovery.patterns import discover_patterns
from simoc.discovery.data_structures import DiscoveryResult
from simoc.simulation.runner import SimulationRunner
from simoc.simulation.data_structures import SimulationResult
from simoc.output.ocel_export import to_ocel

FIXTURE_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="session")
def sim_result() -> SimulationResult:
    data = load_ocel(str(FIXTURE_DIR / "sample_order_process.json"))
    bd = compute_birth_death(data)
    oig = build_oig(data, bd)
    tc = classify_types(oig, data.object_types)
    sp = compute_spawning_profiles(data, bd, tc)
    bp = discover_behavioral(data, bd, tc)
    ip = discover_patterns(data, bd, tc)
    dr = DiscoveryResult(bd, oig, tc, sp)
    runner = SimulationRunner(dr, bp, ip)
    return runner.run(duration=36000, seed=42)


@pytest.fixture(scope="session")
def ocel_output(sim_result: SimulationResult):
    return to_ocel(sim_result)
