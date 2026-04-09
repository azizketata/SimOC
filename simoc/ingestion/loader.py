"""OCEL 2.0 loading, validation, and data structure construction."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pm4py

from simoc.ingestion.data_structures import E2OIndex, OCELData
from simoc.ingestion.validator import validate_ocel

logger = logging.getLogger(__name__)

# PM4Py default column names
_EID = "ocel:eid"
_OID = "ocel:oid"
_OID2 = "ocel:oid_2"
_OTYPE = "ocel:type"
_ACTIVITY = "ocel:activity"
_TIMESTAMP = "ocel:timestamp"
_QUALIFIER = "ocel:qualifier"


def load_ocel(file_path: str | Path) -> OCELData:
    """Load an OCEL 2.0 log file, validate it, and build core data structures.

    Parameters
    ----------
    file_path : str or Path
        Path to an OCEL 2.0 file (JSON-OCEL, SQLite, or XML).

    Returns
    -------
    OCELData
        Preprocessed data structures ready for downstream stages.

    Raises
    ------
    ValueError
        If the log fails structural validation.
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"OCEL file not found: {file_path}")

    logger.info("Loading OCEL 2.0 log from %s", file_path)
    ocel = pm4py.read_ocel2(str(file_path))

    # Validate structural invariants (raises on failure)
    validate_ocel(ocel)

    # Build core data structures
    data = ocel_to_oceldata(ocel, source_file=str(file_path))

    logger.info(
        "Loaded %d events, %d objects (%d types), %d O2O relations.",
        len(data.events),
        len(data.objects),
        len(data.object_types),
        len(data.o2o),
    )
    return data


def ocel_to_oceldata(ocel, source_file: str = "<synthetic>") -> OCELData:
    """Build OCELData from a PM4Py OCEL object (no file I/O, no validation).

    This is the core conversion logic used by both ``load_ocel`` (after
    file loading and validation) and the evaluation module (for converting
    simulation output back to OCELData format).
    """
    e2o_index = _build_e2o_index(ocel)
    objects_df = _build_objects_df(ocel)
    events_df = _build_events_df(ocel)
    o2o_df = _build_o2o_df(ocel, objects_df)
    lifecycles = _build_lifecycles(ocel, events_df, e2o_index)
    object_types = sorted(objects_df["object_type"].unique().tolist())

    return OCELData(
        events=events_df,
        objects=objects_df,
        o2o=o2o_df,
        lifecycles=lifecycles,
        e2o_index=e2o_index,
        source_file=source_file,
        object_types=object_types,
    )


# ------------------------------------------------------------------
# Private builders
# ------------------------------------------------------------------


def _build_e2o_index(ocel) -> E2OIndex:
    """Build bidirectional event <-> object index from relations table."""
    event_to_objects: dict[str, set[str]] = defaultdict(set)
    object_to_events: dict[str, set[str]] = defaultdict(set)

    for eid, oid in zip(ocel.relations[_EID], ocel.relations[_OID]):
        event_to_objects[eid].add(oid)
        object_to_events[oid].add(eid)

    return E2OIndex(
        event_to_objects=dict(event_to_objects),
        object_to_events=dict(object_to_events),
    )


def _build_objects_df(ocel) -> pd.DataFrame:
    """Rename PM4Py columns and return clean objects DataFrame."""
    df = ocel.objects.copy()
    rename = {_OID: "object_id", _OTYPE: "object_type"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df.reset_index(drop=True)


def _build_events_df(ocel) -> pd.DataFrame:
    """Build events DataFrame with a ``related_objects`` list column."""
    # Group relations by event to get list of (oid, otype) per event
    rels = ocel.relations[[_EID, _OID, _OTYPE]].copy()
    grouped = (
        rels.groupby(_EID)
        .apply(
            lambda g: list(zip(g[_OID], g[_OTYPE])),
            include_groups=False,
        )
        .rename("related_objects")
    )

    df = ocel.events.copy()
    rename = {_EID: "event_id", _ACTIVITY: "activity", _TIMESTAMP: "timestamp"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Merge related objects
    df = df.merge(grouped, left_on="event_id", right_index=True, how="left")

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _build_o2o_df(ocel, objects_df: pd.DataFrame) -> pd.DataFrame:
    """Build O2O relations DataFrame with resolved source/target types."""
    if ocel.o2o is None or ocel.o2o.empty:
        return pd.DataFrame(
            columns=[
                "source_object_id",
                "source_type",
                "target_object_id",
                "target_type",
                "qualifier",
            ]
        )

    df = ocel.o2o.copy()

    # Build oid->type lookup from clean objects DataFrame
    oid_to_type = objects_df.set_index("object_id")["object_type"].to_dict()

    df = df.rename(
        columns={
            _OID: "source_object_id",
            _OID2: "target_object_id",
            _QUALIFIER: "qualifier",
        }
    )

    df["source_type"] = df["source_object_id"].map(oid_to_type)
    df["target_type"] = df["target_object_id"].map(oid_to_type)

    # Ensure qualifier column exists
    if "qualifier" not in df.columns:
        df["qualifier"] = ""

    return df[
        ["source_object_id", "source_type", "target_object_id", "target_type", "qualifier"]
    ].reset_index(drop=True)


def _build_lifecycles(
    ocel, events_df: pd.DataFrame, e2o_index: E2OIndex
) -> dict[str, list[tuple[str, str, pd.Timestamp]]]:
    """Build per-object lifecycle: ordered list of (event_id, activity, timestamp)."""
    # Pre-build event lookup
    event_lookup: dict[str, tuple[str, pd.Timestamp]] = {}
    for _, row in events_df.iterrows():
        event_lookup[row["event_id"]] = (row["activity"], row["timestamp"])

    lifecycles: dict[str, list[tuple[str, str, pd.Timestamp]]] = {}
    for oid, event_ids in e2o_index.object_to_events.items():
        entries = []
        for eid in event_ids:
            if eid in event_lookup:
                activity, ts = event_lookup[eid]
                entries.append((eid, activity, ts))
        entries.sort(key=lambda x: x[2])
        lifecycles[oid] = entries

    return lifecycles
