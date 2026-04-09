"""Object interaction discovery (Stages 2-4)."""

from simoc.discovery.cardinality import compute_spawning_profiles
from simoc.discovery.data_structures import (
    BirthDeathTable,
    DiscoveryResult,
    FittedDistribution,
    ObjectInteractionGraph,
    SpawningProfile,
    TypeClassification,
)
from simoc.discovery.interaction_graph import (
    build_oig,
    classify_types,
    compute_birth_death,
    discover_interaction_graph,
)

__all__ = [
    "BirthDeathTable",
    "DiscoveryResult",
    "FittedDistribution",
    "ObjectInteractionGraph",
    "SpawningProfile",
    "TypeClassification",
    "build_oig",
    "classify_types",
    "compute_birth_death",
    "compute_spawning_profiles",
    "discover_interaction_graph",
]
