"""Core data structures for OCEL 2.0 ingestion."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class E2OIndex:
    """Bidirectional event-to-object index."""

    event_to_objects: dict[str, set[str]]  # event_id -> {object_ids}
    object_to_events: dict[str, set[str]]  # object_id -> {event_ids}


@dataclass
class OCELData:
    """Container for all preprocessed OCEL 2.0 data structures.

    This is the single output of Stage 1 and the primary input to all
    subsequent stages.
    """

    events: pd.DataFrame
    """Columns: event_id, activity, timestamp, related_objects.
    ``related_objects`` is a list column of (object_id, object_type) tuples.
    Sorted by timestamp."""

    objects: pd.DataFrame
    """Columns: object_id, object_type, plus any attribute columns.
    One row per object."""

    o2o: pd.DataFrame
    """Columns: source_object_id, source_type, target_object_id, target_type,
    qualifier.  One row per O2O relation."""

    lifecycles: dict[str, list[tuple[str, str, pd.Timestamp]]]
    """Mapping from object_id to its ordered list of
    (event_id, activity, timestamp) tuples, sorted by timestamp."""

    e2o_index: E2OIndex
    """Bidirectional event <-> object index."""

    source_file: str
    """Path to the original OCEL 2.0 file that was loaded."""

    object_types: list[str]
    """Sorted list of unique object type names in the log."""
