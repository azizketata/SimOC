"""Summary statistics for preprocessed OCEL 2.0 data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from simoc.ingestion.data_structures import OCELData


@dataclass
class OCELSummary:
    """Aggregate statistics for an OCEL 2.0 log."""

    num_events: int
    num_objects: int
    num_object_types: int
    num_o2o_relations: int
    num_distinct_qualifiers: int
    time_span: tuple[pd.Timestamp, pd.Timestamp]

    per_object_type: pd.DataFrame
    """Columns: object_type, count, avg_lifecycle_length,
    min_lifecycle_length, max_lifecycle_length."""

    per_activity: pd.DataFrame
    """Columns: activity, count, avg_objects_per_event."""

    def __str__(self) -> str:
        lines = [
            "=== OCEL 2.0 Summary ===",
            f"Events:          {self.num_events}",
            f"Objects:         {self.num_objects}",
            f"Object types:    {self.num_object_types}",
            f"O2O relations:   {self.num_o2o_relations}",
            f"O2O qualifiers:  {self.num_distinct_qualifiers}",
            f"Time span:       {self.time_span[0]} -> {self.time_span[1]}",
            "",
            "--- Per Object Type ---",
            self.per_object_type.to_string(index=False),
            "",
            "--- Per Activity ---",
            self.per_activity.to_string(index=False),
        ]
        return "\n".join(lines)


def compute_summary(data: OCELData) -> OCELSummary:
    """Compute summary statistics from preprocessed OCEL data."""
    # -- Global counts --
    num_events = len(data.events)
    num_objects = len(data.objects)
    num_object_types = len(data.object_types)
    num_o2o_relations = len(data.o2o)

    if num_o2o_relations > 0 and "qualifier" in data.o2o.columns:
        num_distinct_qualifiers = data.o2o["qualifier"].nunique()
    else:
        num_distinct_qualifiers = 0

    time_span = (
        data.events["timestamp"].min(),
        data.events["timestamp"].max(),
    )

    # -- Per object type --
    # Build lifecycle length per object, keyed by type
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
    type_lifecycle_lengths: dict[str, list[int]] = {t: [] for t in data.object_types}
    for oid, events in data.lifecycles.items():
        otype = oid_to_type.get(oid)
        if otype is not None:
            type_lifecycle_lengths[otype].append(len(events))

    rows_type = []
    for otype in data.object_types:
        lengths = type_lifecycle_lengths[otype]
        count = len(lengths)
        rows_type.append(
            {
                "object_type": otype,
                "count": count,
                "avg_lifecycle_length": sum(lengths) / count if count else 0,
                "min_lifecycle_length": min(lengths) if lengths else 0,
                "max_lifecycle_length": max(lengths) if lengths else 0,
            }
        )
    per_object_type = pd.DataFrame(rows_type)

    # -- Per activity --
    data.events["_num_objects"] = data.events["related_objects"].apply(len)
    per_activity = (
        data.events.groupby("activity")
        .agg(count=("activity", "size"), avg_objects_per_event=("_num_objects", "mean"))
        .reset_index()
    )
    data.events.drop(columns=["_num_objects"], inplace=True)

    return OCELSummary(
        num_events=num_events,
        num_objects=num_objects,
        num_object_types=num_object_types,
        num_o2o_relations=num_o2o_relations,
        num_distinct_qualifiers=num_distinct_qualifiers,
        time_span=time_span,
        per_object_type=per_object_type,
        per_activity=per_activity,
    )
