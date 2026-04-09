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

    # Compute per-type counts and lifecycle lengths for master data detection
    object_counts = data.objects["object_type"].value_counts().to_dict()
    avg_lifecycle_lengths: dict[str, float] = {}
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    type_lc_lengths: dict[str, list[int]] = {t: [] for t in data.object_types}
    for oid, lc in data.lifecycles.items():
        otype = oid_to_type.get(oid)
        if otype:
            type_lc_lengths[otype].append(len(lc))
    for t, lengths in type_lc_lengths.items():
        avg_lifecycle_lengths[t] = sum(lengths) / len(lengths) if lengths else 0

    type_class = classify_types(
        oig, data.object_types, object_counts, avg_lifecycle_lengths
    )
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
    oig: ObjectInteractionGraph,
    object_types: list[str],
    object_counts: dict[str, int] | None = None,
    avg_lifecycle_lengths: dict[str, float] | None = None,
) -> TypeClassification:
    """Task 2.3: Classify each object type as root or derived.

    Parameters
    ----------
    object_counts : dict, optional
        {type -> instance count}. Used for master data detection.
    avg_lifecycle_lengths : dict, optional
        {type -> avg number of events per object}. Used for master data detection.
    """
    classification: dict[str, str] = {}
    parent_map: dict[str, str] = {}

    # Pre-classify master data types as root.
    # Master data = few instances with very long lifecycles (reference entities).
    master_data_types: set[str] = set()
    if object_counts and avg_lifecycle_lengths and len(object_types) > 3:
        median_count = sorted(object_counts.values())[len(object_counts) // 2]
        for t in object_types:
            count = object_counts.get(t, 0)
            avg_lc = avg_lifecycle_lengths.get(t, 0)
            # Heuristic: < 5% of median instance count AND lifecycle > 50 events
            if count < max(30, median_count * 0.05) and avg_lc > 50:
                master_data_types.add(t)
                classification[t] = "root"
                logger.info(
                    "Type '%s' detected as master data (count=%d, avg_lifecycle=%.0f). Forcing ROOT.",
                    t, count, avg_lc,
                )

    for t in object_types:
        if t in master_data_types:
            continue  # already classified as root
        # Find candidate parents: types T' where >80% of T's births have T' present
        # AND either T' is born before T OR T' has O2O relations to T
        # (O2O direction is strong structural evidence of parent-child)
        base = oig.df[
            (oig.df["type_2"] == t)
            & (oig.df["type_1"] != t)
            & (oig.df["t2_birth_has_t1_pct"] > 0.8)
        ]
        candidates = base[
            (base["t1_born_first_pct"] > 0.5) | (base["o2o_count"] > 0)
        ]

        if candidates.empty:
            classification[t] = "root"
        elif len(candidates) == 1:
            classification[t] = "derived"
            parent_map[t] = candidates.iloc[0]["type_1"]
        else:
            # Multiple candidates: prefer the one with O2O relations (strongest
            # structural signal), then fall back to highest birth co-occurrence.
            with_o2o = candidates[candidates["o2o_count"] > 0]
            if len(with_o2o) == 1:
                best = with_o2o.iloc[0]
            elif len(with_o2o) > 1:
                best = with_o2o.loc[with_o2o["o2o_count"].idxmax()]
            else:
                best = candidates.loc[candidates["t2_birth_has_t1_pct"].idxmax()]
            classification[t] = "derived"
            parent_map[t] = best["type_1"]
            logger.warning(
                "Type '%s' has multiple parent candidates: %s. Selected '%s'.",
                t,
                candidates["type_1"].tolist(),
                best["type_1"],
            )

    # Post-check: if a derived type's parent is master data, the parent is
    # a resource/reference, not a real spawning parent. Promote child to root.
    for t in list(parent_map.keys()):
        parent = parent_map[t]
        if parent in master_data_types:
            logger.info(
                "Promoting '%s' to root: parent '%s' is master data (resource).",
                t, parent,
            )
            del parent_map[t]
            classification[t] = "root"

    # Break any remaining cycles by demoting the type with fewer instances
    # (types with fewer instances are more likely to be master data / root)
    _break_cycles(parent_map, classification, object_types, oig)

    # Validate DAG: no cycles in parent chain
    _validate_dag(parent_map)

    return TypeClassification(classification=classification, parent_map=parent_map)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _break_cycles(
    parent_map: dict[str, str],
    classification: dict[str, str],
    object_types: list[str],
    oig,
) -> None:
    """Break cycles in parent mapping by promoting cycle members to root."""
    max_iterations = len(object_types)
    for _ in range(max_iterations):
        # Find a cycle
        cycle_found = False
        for start in list(parent_map.keys()):
            visited: set[str] = set()
            current = start
            while current in parent_map:
                if current in visited:
                    # Cycle detected — promote this type to root
                    logger.warning(
                        "Breaking cycle: promoting '%s' to root (was derived from '%s').",
                        current,
                        parent_map[current],
                    )
                    del parent_map[current]
                    classification[current] = "root"
                    cycle_found = True
                    break
                visited.add(current)
                current = parent_map[current]
            if cycle_found:
                break
        if not cycle_found:
            break


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
