"""Tasks 2.1–2.3: Birth/death computation, OIG construction, and type classification."""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict

import pandas as pd

from simoc.ingestion.data_structures import OCELData
from simoc.discovery.data_structures import (
    BirthDeathTable,
    ObjectInteractionGraph,
    TypeClassification,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def discover_interaction_graph(
    data: OCELData,
) -> tuple[BirthDeathTable, ObjectInteractionGraph, TypeClassification]:
    """Orchestrator: run Tasks 2.1–2.3 in sequence."""
    birth_death = compute_birth_death(data)
    oig = build_oig(data, birth_death)
    type_class = classify_types(oig, data.object_types)
    logger.info(
        "Type classification: %s",
        {t: type_class.classification[t] for t in sorted(type_class.classification)},
    )
    return birth_death, oig, type_class


def compute_birth_death(data: OCELData) -> BirthDeathTable:
    """Task 2.1: Compute birth and death events for every object."""
    # Build event_id -> related_objects lookup
    event_related: dict[str, list[tuple[str, str]]] = {}
    for _, row in data.events.iterrows():
        event_related[row["event_id"]] = row["related_objects"]

    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()

    rows = []
    for oid, lifecycle in data.lifecycles.items():
        birth_eid, birth_act, birth_ts = lifecycle[0]
        death_eid, death_act, death_ts = lifecycle[-1]

        # Co-objects at birth: all other objects in the birth event
        birth_co = [
            (co_oid, co_otype)
            for co_oid, co_otype in event_related.get(birth_eid, [])
            if co_oid != oid
        ]

        rows.append(
            {
                "object_id": oid,
                "object_type": oid_to_type.get(oid, ""),
                "birth_event_id": birth_eid,
                "birth_activity": birth_act,
                "birth_timestamp": birth_ts,
                "birth_co_objects": birth_co,
                "death_event_id": death_eid,
                "death_activity": death_act,
                "death_timestamp": death_ts,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Computed birth/death for %d objects.", len(df))
    return BirthDeathTable(df=df)


def build_oig(data: OCELData, birth_death: BirthDeathTable) -> ObjectInteractionGraph:
    """Task 2.2: Build the Object Interaction Graph."""
    types = data.object_types  # sorted unique types

    # --- Pre-compute helpers ---

    # E2O co-occurrence: count events with both T1 and T2 present
    cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in data.events.iterrows():
        types_in_event = {otype for _, otype in row["related_objects"]}
        for t1, t2 in itertools.product(types_in_event, types_in_event):
            cooccurrence[(t1, t2)] += 1

    # O2O counts and qualifiers
    o2o_count: dict[tuple[str, str], int] = defaultdict(int)
    o2o_quals: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, row in data.o2o.iterrows():
        key = (row["source_type"], row["target_type"])
        o2o_count[key] += 1
        qual = row.get("qualifier", "")
        if qual:
            o2o_quals[key].add(str(qual))

    # Birth timestamps per object
    birth_ts: dict[str, pd.Timestamp] = {}
    birth_co_by_oid: dict[str, list[tuple[str, str]]] = {}
    oid_to_type: dict[str, str] = {}
    for _, row in birth_death.df.iterrows():
        oid = row["object_id"]
        birth_ts[oid] = row["birth_timestamp"]
        birth_co_by_oid[oid] = row["birth_co_objects"]
        oid_to_type[oid] = row["object_type"]

    # Objects by type
    objects_by_type: dict[str, list[str]] = defaultdict(list)
    for oid, otype in oid_to_type.items():
        objects_by_type[otype].append(oid)

    # Co-occurring instance pairs: for each event, record (oid1, oid2) pairs
    # of different objects. Used for temporal ordering.
    cooccurring_pairs: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for _, row in data.events.iterrows():
        objs = row["related_objects"]  # list of (oid, otype)
        for (oid1, ot1), (oid2, ot2) in itertools.product(objs, objs):
            if oid1 != oid2:
                cooccurring_pairs[(ot1, ot2)].add((oid1, oid2))

    # --- Build OIG rows ---
    oig_rows = []
    for t1, t2 in itertools.product(types, types):
        # E2O co-occurrence
        e2o_cooc = cooccurrence.get((t1, t2), 0)

        # O2O
        o2o_cnt = o2o_count.get((t1, t2), 0)
        o2o_qual = ",".join(sorted(o2o_quals.get((t1, t2), set()))) or ""

        # Temporal ordering: % of co-occurring instance pairs where t1 born first
        pairs = cooccurring_pairs.get((t1, t2), set())
        if pairs and t1 != t2:
            t1_first = sum(
                1 for o1, o2 in pairs if birth_ts[o1] < birth_ts[o2]
            )
            t1_born_first_pct = t1_first / len(pairs)
        else:
            t1_born_first_pct = 0.0

        # Birth co-occurrence: % of T2 births that have a T1 co-object
        t2_objects = objects_by_type.get(t2, [])
        if t2_objects and t1 != t2:
            has_t1 = sum(
                1
                for oid in t2_objects
                if any(co_otype == t1 for _, co_otype in birth_co_by_oid.get(oid, []))
            )
            t2_birth_has_t1_pct = has_t1 / len(t2_objects)
        else:
            t2_birth_has_t1_pct = 0.0

        oig_rows.append(
            {
                "type_1": t1,
                "type_2": t2,
                "e2o_cooccurrence": e2o_cooc,
                "o2o_count": o2o_cnt,
                "o2o_qualifier": o2o_qual,
                "t1_born_first_pct": t1_born_first_pct,
                "t2_birth_has_t1_pct": t2_birth_has_t1_pct,
            }
        )

    df = pd.DataFrame(oig_rows)
    logger.info("Built OIG with %d type pairs.", len(df))
    return ObjectInteractionGraph(df=df)


def classify_types(
    oig: ObjectInteractionGraph, object_types: list[str]
) -> TypeClassification:
    """Task 2.3: Classify each object type as root or derived."""
    classification: dict[str, str] = {}
    parent_map: dict[str, str] = {}

    for t in object_types:
        # Find candidate parents: types T' where >80% of T's births have T' present
        candidates = oig.df[
            (oig.df["type_2"] == t)
            & (oig.df["type_1"] != t)
            & (oig.df["t2_birth_has_t1_pct"] > 0.8)
        ]

        if candidates.empty:
            classification[t] = "root"
        elif len(candidates) == 1:
            classification[t] = "derived"
            parent_map[t] = candidates.iloc[0]["type_1"]
        else:
            # Multiple candidates: pick highest percentage
            best = candidates.loc[candidates["t2_birth_has_t1_pct"].idxmax()]
            classification[t] = "derived"
            parent_map[t] = best["type_1"]
            logger.warning(
                "Type '%s' has multiple parent candidates: %s. Selected '%s'.",
                t,
                candidates["type_1"].tolist(),
                best["type_1"],
            )

    # Validate DAG: no cycles in parent chain
    _validate_dag(parent_map)

    return TypeClassification(classification=classification, parent_map=parent_map)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _validate_dag(parent_map: dict[str, str]) -> None:
    """Raise ValueError if the parent mapping contains a cycle."""
    for start in parent_map:
        visited: set[str] = set()
        current = start
        while current in parent_map:
            if current in visited:
                raise ValueError(
                    f"Cycle detected in parent mapping: "
                    f"{start} -> ... -> {current}"
                )
            visited.add(current)
            current = parent_map[current]
