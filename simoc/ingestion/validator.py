"""Validation of OCEL 2.0 structural invariants."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pm4py.objects.ocel.obj import OCEL

logger = logging.getLogger(__name__)

# PM4Py default column names
_EID = "ocel:eid"
_OID = "ocel:oid"
_OID2 = "ocel:oid_2"
_OTYPE = "ocel:type"
_TIMESTAMP = "ocel:timestamp"


def validate_ocel(ocel: OCEL) -> list[str]:
    """Check 5 structural invariants on a raw PM4Py OCEL object.

    Returns a list of violation messages.  An empty list means all checks
    passed.  Raises ``ValueError`` if any invariant is violated.
    """
    violations: list[str] = []
    violations.extend(_check_no_orphan_events(ocel))
    violations.extend(_check_object_types(ocel))
    violations.extend(_check_o2o_references(ocel))
    violations.extend(_check_timestamps(ocel))
    violations.extend(_check_no_duplicate_ids(ocel))

    if violations:
        msg = "OCEL validation failed:\n  " + "\n  ".join(violations)
        logger.error(msg)
        raise ValueError(msg)

    logger.info("OCEL validation passed — all 5 invariants satisfied.")
    return violations


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _check_no_orphan_events(ocel: OCEL) -> list[str]:
    """Every event must have at least one associated object."""
    violations: list[str] = []
    event_ids = set(ocel.events[_EID])
    related_event_ids = set(ocel.relations[_EID])
    orphans = event_ids - related_event_ids
    if orphans:
        sample = sorted(orphans)[:5]
        violations.append(
            f"Orphan events (no associated objects): {sample}"
            f"{' ...' if len(orphans) > 5 else ''} "
            f"({len(orphans)} total)"
        )
    return violations


def _check_object_types(ocel: OCEL) -> list[str]:
    """Every object must have a non-null type."""
    violations: list[str] = []
    null_mask = ocel.objects[_OTYPE].isna()
    if null_mask.any():
        bad_ids = ocel.objects.loc[null_mask, _OID].tolist()[:5]
        violations.append(
            f"Objects with null type: {bad_ids} ({null_mask.sum()} total)"
        )
    return violations


def _check_o2o_references(ocel: OCEL) -> list[str]:
    """Every O2O relation must reference existing objects."""
    violations: list[str] = []
    if ocel.o2o is None or ocel.o2o.empty:
        return violations

    valid_oids = set(ocel.objects[_OID])

    # Check source objects
    if _OID in ocel.o2o.columns:
        bad_sources = set(ocel.o2o[_OID]) - valid_oids
        if bad_sources:
            violations.append(
                f"O2O source object IDs not in objects table: "
                f"{sorted(bad_sources)[:5]}"
            )

    # Check target objects
    if _OID2 in ocel.o2o.columns:
        bad_targets = set(ocel.o2o[_OID2]) - valid_oids
        if bad_targets:
            violations.append(
                f"O2O target object IDs not in objects table: "
                f"{sorted(bad_targets)[:5]}"
            )

    return violations


def _check_timestamps(ocel: OCEL) -> list[str]:
    """Timestamps must be parseable (no NaT values)."""
    violations: list[str] = []
    ts = ocel.events[_TIMESTAMP]
    nat_count = ts.isna().sum()
    if nat_count:
        violations.append(f"Events with unparseable timestamps (NaT): {nat_count}")
    return violations


def _check_no_duplicate_ids(ocel: OCEL) -> list[str]:
    """Event IDs and object IDs must be unique."""
    violations: list[str] = []
    dup_events = ocel.events[_EID].duplicated().sum()
    if dup_events:
        violations.append(f"Duplicate event IDs: {dup_events}")

    dup_objects = ocel.objects[_OID].duplicated().sum()
    if dup_objects:
        violations.append(f"Duplicate object IDs: {dup_objects}")

    return violations
