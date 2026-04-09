"""Shared helpers for evaluation: format conversion and distribution extraction."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from simoc.ingestion.data_structures import OCELData
from simoc.ingestion.loader import ocel_to_oceldata
from simoc.output.ocel_export import to_ocel
from simoc.simulation.data_structures import SimulationResult


def sim_result_to_oceldata(
    result: SimulationResult,
    start_timestamp: str | pd.Timestamp | None = None,
) -> OCELData:
    """Convert SimulationResult to OCELData via PM4Py OCEL round-trip."""
    ocel = to_ocel(result, start_timestamp)
    return ocel_to_oceldata(ocel, source_file="<synthetic>")


def extract_activity_frequencies(data: OCELData) -> dict[str, int]:
    """Return {activity: count}."""
    return data.events["activity"].value_counts().to_dict()


def extract_cycle_times(data: OCELData) -> dict[str, list[float]]:
    """Return {object_type: [cycle_time_seconds, ...]}."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    result: dict[str, list[float]] = defaultdict(list)
    for oid, lc in data.lifecycles.items():
        if len(lc) < 2:
            continue
        otype = oid_to_type.get(oid, "")
        cycle = (lc[-1][2] - lc[0][2]).total_seconds()
        result[otype].append(cycle)
    return dict(result)


def extract_inter_arrival_times(data: OCELData, object_type: str) -> list[float]:
    """Return sorted inter-arrival gaps in seconds for the given type."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    births = []
    for oid, lc in data.lifecycles.items():
        if oid_to_type.get(oid) == object_type and lc:
            births.append(lc[0][2])
    births.sort()
    if len(births) < 2:
        return []
    return [(births[i + 1] - births[i]).total_seconds() for i in range(len(births) - 1)]


def extract_durations(data: OCELData) -> dict[tuple[str, str], list[float]]:
    """Return {(type, activity): [duration_seconds, ...]}."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    result: dict[tuple[str, str], list[float]] = defaultdict(list)
    for oid, lc in data.lifecycles.items():
        otype = oid_to_type.get(oid, "")
        for i in range(len(lc) - 1):
            _, act, ts = lc[i]
            _, _, ts_next = lc[i + 1]
            result[(otype, act)].append((ts_next - ts).total_seconds())
    return dict(result)


def extract_cardinality_counts(
    data: OCELData, parent_type: str, child_type: str
) -> list[int]:
    """Return children-per-parent counts from O2O relations."""
    counts: dict[str, int] = defaultdict(int)
    parent_oids = set(
        data.objects.loc[data.objects["object_type"] == parent_type, "object_id"]
    )
    for _, row in data.o2o.iterrows():
        if row["source_type"] == parent_type and row["target_type"] == child_type:
            counts[row["source_object_id"]] += 1
        elif row["source_type"] == child_type and row["target_type"] == parent_type:
            counts[row["target_object_id"]] += 1
    # Include parents with 0 children
    for pid in parent_oids:
        if pid not in counts:
            counts[pid] = 0
    return list(counts.values()) if counts else []


def extract_objects_per_event(data: OCELData) -> dict[str, list[int]]:
    """Return {activity: [n_objects_in_event, ...]}."""
    result: dict[str, list[int]] = defaultdict(list)
    for _, row in data.events.iterrows():
        result[row["activity"]].append(len(row["related_objects"]))
    return dict(result)


def discover_oc_dfg(data: OCELData) -> dict[tuple[str, str, str], int]:
    """Discover OC-DFG: {(source_act, target_act, type): frequency}."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    edges: dict[tuple[str, str, str], int] = defaultdict(int)
    for oid, lc in data.lifecycles.items():
        otype = oid_to_type.get(oid, "")
        for i in range(len(lc) - 1):
            _, act1, _ = lc[i]
            _, act2, _ = lc[i + 1]
            edges[(str(act1), str(act2), otype)] += 1
    return dict(edges)
