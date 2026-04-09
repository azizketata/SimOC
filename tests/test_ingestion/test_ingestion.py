"""Acceptance tests for Stage 1: OCEL 2.0 Ingestion and Preprocessing.

Covers test criteria T1.1 through T1.5 from the development guide,
plus additional edge-case tests.
"""

from pathlib import Path

import pandas as pd
import pm4py
import pytest

from simoc.ingestion import OCELData, compute_summary, load_ocel
from simoc.ingestion.validator import validate_ocel


# ------------------------------------------------------------------
# T1.1 — Load benchmark log without errors
# ------------------------------------------------------------------

class TestT1_1_LoadWithoutErrors:
    def test_loads_successfully(self, loaded_data: OCELData):
        assert loaded_data is not None

    def test_event_count_matches(self, loaded_data: OCELData):
        # Fixture has 14 events
        assert len(loaded_data.events) == 14

    def test_object_count_matches(self, loaded_data: OCELData):
        # Fixture has 6 objects
        assert len(loaded_data.objects) == 6

    def test_object_types_discovered(self, loaded_data: OCELData):
        assert loaded_data.object_types == ["delivery", "item", "order"]


# ------------------------------------------------------------------
# T1.2 — Structural invariants
# ------------------------------------------------------------------

class TestT1_2_Validation:
    def test_valid_log_passes(self, sample_ocel_path: Path):
        """Valid log should raise no errors."""
        ocel = pm4py.read_ocel2(str(sample_ocel_path))
        violations = validate_ocel(ocel)
        assert violations == []

    def test_duplicate_event_ids_detected(self, sample_ocel_path: Path):
        """Injecting a duplicate event ID should trigger a violation."""
        ocel = pm4py.read_ocel2(str(sample_ocel_path))
        # Duplicate the first event row
        ocel.events = pd.concat([ocel.events, ocel.events.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate event IDs"):
            validate_ocel(ocel)

    def test_null_object_type_detected(self, sample_ocel_path: Path):
        """Null object type should trigger a violation."""
        ocel = pm4py.read_ocel2(str(sample_ocel_path))
        ocel.objects.loc[0, "ocel:type"] = None
        with pytest.raises(ValueError, match="null type"):
            validate_ocel(ocel)


# ------------------------------------------------------------------
# T1.3 — Lifecycle correctness
# ------------------------------------------------------------------

class TestT1_3_Lifecycles:
    def test_order_1_lifecycle(self, loaded_data: OCELData):
        """order_1 participates in: e1, e2, e3, e6, e11, e13, e14."""
        lc = loaded_data.lifecycles["order_1"]
        event_ids = [eid for eid, _, _ in lc]
        assert event_ids == ["e1", "e2", "e3", "e6", "e11", "e13", "e14"]

    def test_item_1_lifecycle(self, loaded_data: OCELData):
        """item_1 participates in: e2, e7, e11."""
        lc = loaded_data.lifecycles["item_1"]
        event_ids = [eid for eid, _, _ in lc]
        assert event_ids == ["e2", "e7", "e11"]

    def test_delivery_1_lifecycle(self, loaded_data: OCELData):
        """delivery_1 participates in: e13, e14."""
        lc = loaded_data.lifecycles["delivery_1"]
        event_ids = [eid for eid, _, _ in lc]
        assert event_ids == ["e13", "e14"]

    def test_lifecycles_sorted_by_timestamp(self, loaded_data: OCELData):
        """Every object's lifecycle must be sorted by timestamp."""
        for oid, lc in loaded_data.lifecycles.items():
            timestamps = [ts for _, _, ts in lc]
            assert timestamps == sorted(timestamps), f"{oid} lifecycle not sorted"


# ------------------------------------------------------------------
# T1.4 — E2O index bidirectional consistency
# ------------------------------------------------------------------

class TestT1_4_E2OIndex:
    def test_forward_implies_reverse(self, loaded_data: OCELData):
        """If oid in event_to_objects[eid], then eid in object_to_events[oid]."""
        idx = loaded_data.e2o_index
        for eid, oids in idx.event_to_objects.items():
            for oid in oids:
                assert eid in idx.object_to_events[oid], (
                    f"eid={eid}, oid={oid}: forward present but reverse missing"
                )

    def test_reverse_implies_forward(self, loaded_data: OCELData):
        """If eid in object_to_events[oid], then oid in event_to_objects[eid]."""
        idx = loaded_data.e2o_index
        for oid, eids in idx.object_to_events.items():
            for eid in eids:
                assert oid in idx.event_to_objects[eid], (
                    f"oid={oid}, eid={eid}: reverse present but forward missing"
                )


# ------------------------------------------------------------------
# T1.5 — Summary statistics sanity
# ------------------------------------------------------------------

class TestT1_5_SummaryStats:
    def test_totals_match(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        assert summary.num_events == len(loaded_data.events)
        assert summary.num_objects == len(loaded_data.objects)
        assert summary.num_object_types == len(loaded_data.object_types)

    def test_per_type_sums(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        assert summary.per_object_type["count"].sum() == summary.num_objects

    def test_no_nan_values(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        assert not summary.per_object_type.isna().any().any()
        assert not summary.per_activity.isna().any().any()

    def test_no_negative_lifecycle_lengths(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        assert (summary.per_object_type["min_lifecycle_length"] >= 0).all()

    def test_time_span_ordered(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        assert summary.time_span[0] <= summary.time_span[1]

    def test_str_output(self, loaded_data: OCELData):
        summary = compute_summary(loaded_data)
        text = str(summary)
        assert "OCEL 2.0 Summary" in text


# ------------------------------------------------------------------
# Additional edge-case tests
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_events_sorted_by_timestamp(self, loaded_data: OCELData):
        assert loaded_data.events["timestamp"].is_monotonic_increasing

    def test_related_objects_not_empty(self, loaded_data: OCELData):
        """Every event must have at least one related object."""
        for _, row in loaded_data.events.iterrows():
            assert len(row["related_objects"]) >= 1, (
                f"Event {row['event_id']} has no related objects"
            )

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ocel("nonexistent_file.json")

    def test_o2o_relations_have_types(self, loaded_data: OCELData):
        """Every O2O row should have resolved source and target types."""
        if len(loaded_data.o2o) > 0:
            assert loaded_data.o2o["source_type"].notna().all()
            assert loaded_data.o2o["target_type"].notna().all()
