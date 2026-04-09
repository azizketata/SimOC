"""Shared fixtures for discovery tests."""

from pathlib import Path

import pytest

from simoc.ingestion import OCELData, load_ocel
from simoc.discovery.data_structures import (
    BehavioralProfile,
    BirthDeathTable,
    InteractionPatterns,
    ObjectInteractionGraph,
    TypeClassification,
)
from simoc.discovery.behavioral import discover_behavioral
from simoc.discovery.patterns import discover_patterns
from simoc.discovery.interaction_graph import (
    build_oig,
    classify_types,
    compute_birth_death,
)

FIXTURE_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="session")
def loaded_data() -> OCELData:
    return load_ocel(str(FIXTURE_DIR / "sample_order_process.json"))


@pytest.fixture(scope="session")
def birth_death(loaded_data: OCELData) -> BirthDeathTable:
    return compute_birth_death(loaded_data)


@pytest.fixture(scope="session")
def oig(loaded_data: OCELData, birth_death: BirthDeathTable) -> ObjectInteractionGraph:
    return build_oig(loaded_data, birth_death)


@pytest.fixture(scope="session")
def type_class(
    oig: ObjectInteractionGraph, loaded_data: OCELData
) -> TypeClassification:
    return classify_types(oig, loaded_data.object_types)


@pytest.fixture(scope="session")
def behavioral_profile(
    loaded_data: OCELData,
    birth_death: BirthDeathTable,
    type_class: TypeClassification,
) -> BehavioralProfile:
    return discover_behavioral(loaded_data, birth_death, type_class)


@pytest.fixture(scope="session")
def interaction_patterns(
    loaded_data: OCELData,
    birth_death: BirthDeathTable,
    type_class: TypeClassification,
) -> InteractionPatterns:
    return discover_patterns(loaded_data, birth_death, type_class)
