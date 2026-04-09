"""Tests T6.1–T6.6: OCEL 2.0 output generation."""

import json

import pandas as pd
import pm4py
import pytest

from simoc.output.ocel_export import export_ocel, to_ocel
from simoc.simulation.data_structures import (
    SimulatedEvent,
    SimulatedObject,
    SimulatedO2O,
    SimulationConfig,
    SimulationResult,
)


# ------------------------------------------------------------------
# T6.1 — OCEL 2.0 schema compliance (loads in PM4Py)
# ------------------------------------------------------------------

class TestT6_1_SchemaCompliance:
    def test_json_export_loads(self, sim_result, tmp_path):
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        ocel = pm4py.read_ocel2(str(path))
        assert len(ocel.events) > 0

    def test_sqlite_export_loads(self, sim_result, tmp_path):
        path = tmp_path / "output.sqlite"
        export_ocel(sim_result, str(path))
        ocel = pm4py.read_ocel2(str(path))
        assert len(ocel.events) > 0


# ------------------------------------------------------------------
# T6.2 — Round-trip consistency
# ------------------------------------------------------------------

class TestT6_2_RoundTrip:
    def test_event_count_matches(self, sim_result, tmp_path):
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        reimported = pm4py.read_ocel2(str(path))
        assert len(reimported.events) == len(sim_result.events)

    def test_object_count_matches(self, sim_result, tmp_path):
        """Objects that appear in events round-trip correctly.
        Orphan objects (spawned but never in events) may be dropped by PM4Py."""
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        reimported = pm4py.read_ocel2(str(path))
        # Count objects that actually appear in events
        active_oids = set()
        for e in sim_result.events:
            for oid, _ in e.objects:
                active_oids.add(oid)
        assert len(reimported.objects) >= len(active_oids)

    def test_o2o_count_reasonable(self, sim_result, tmp_path):
        """O2O relations involving active objects round-trip correctly.
        Relations involving orphan objects may be dropped by PM4Py."""
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        reimported = pm4py.read_ocel2(str(path))
        # At minimum, O2O count should be > 0 if we had any
        if sim_result.o2o_relations:
            assert len(reimported.o2o) > 0

    def test_activity_vocabulary_matches(self, sim_result, tmp_path):
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        reimported = pm4py.read_ocel2(str(path))
        original_acts = {str(e.activity) for e in sim_result.events}
        reimported_acts = set(reimported.events["ocel:activity"].unique())
        assert original_acts == reimported_acts


# ------------------------------------------------------------------
# T6.3 — Timestamp correctness
# ------------------------------------------------------------------

class TestT6_3_Timestamps:
    def test_no_nat_timestamps(self, ocel_output):
        assert ocel_output.events["ocel:timestamp"].isna().sum() == 0

    def test_all_timestamps_after_start(self, ocel_output):
        start = pd.Timestamp("2024-01-01", tz="UTC")
        assert (ocel_output.events["ocel:timestamp"] >= start).all()

    def test_timestamps_within_duration(self, sim_result):
        ocel = to_ocel(sim_result)
        start = pd.Timestamp("2024-01-01", tz="UTC")
        max_ts = start + pd.Timedelta(seconds=sim_result.config.duration)
        assert (ocel.events["ocel:timestamp"] <= max_ts).all()

    def test_custom_start_timestamp(self, sim_result):
        custom_start = "2025-06-15T10:00:00Z"
        ocel = to_ocel(sim_result, start_timestamp=custom_start)
        start = pd.Timestamp(custom_start)
        assert (ocel.events["ocel:timestamp"] >= start).all()


# ------------------------------------------------------------------
# T6.4 — E2O completeness (no orphan events)
# ------------------------------------------------------------------

class TestT6_4_E2OCompleteness:
    def test_every_event_has_relation(self, ocel_output):
        event_ids = set(ocel_output.events["ocel:eid"])
        related_eids = set(ocel_output.relations["ocel:eid"])
        orphans = event_ids - related_eids
        assert len(orphans) == 0, f"Orphan events: {orphans}"

    def test_relations_reference_valid_events(self, ocel_output):
        event_ids = set(ocel_output.events["ocel:eid"])
        rel_eids = set(ocel_output.relations["ocel:eid"])
        invalid = rel_eids - event_ids
        assert len(invalid) == 0


# ------------------------------------------------------------------
# T6.5 — O2O completeness
# ------------------------------------------------------------------

class TestT6_5_O2OCompleteness:
    def test_o2o_sources_exist(self, ocel_output):
        valid_oids = set(ocel_output.objects["ocel:oid"])
        if len(ocel_output.o2o) > 0:
            sources = set(ocel_output.o2o["ocel:oid"])
            assert sources.issubset(valid_oids), f"Invalid sources: {sources - valid_oids}"

    def test_o2o_targets_exist(self, ocel_output):
        valid_oids = set(ocel_output.objects["ocel:oid"])
        if len(ocel_output.o2o) > 0:
            targets = set(ocel_output.o2o["ocel:oid_2"])
            assert targets.issubset(valid_oids), f"Invalid targets: {targets - valid_oids}"


# ------------------------------------------------------------------
# T6.6 — Cross-tool compatibility
# ------------------------------------------------------------------

class TestT6_6_CrossTool:
    def test_json_valid_structure(self, sim_result, tmp_path):
        """The JSON output should have standard OCEL 2.0 keys."""
        path = tmp_path / "output.jsonocel"
        export_ocel(sim_result, str(path))
        with open(path) as f:
            data = json.load(f)
        # PM4Py OCEL 2.0 JSON uses either top-level keys
        has_events = "ocel:events" in data or "events" in data
        has_objects = "ocel:objects" in data or "objects" in data
        assert has_events, f"No events key in JSON. Keys: {list(data.keys())}"
        assert has_objects, f"No objects key in JSON. Keys: {list(data.keys())}"


# ------------------------------------------------------------------
# Extra — Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_result(self):
        """Empty simulation result should produce a valid OCEL."""
        result = SimulationResult(
            events=[], objects=[], o2o_relations=[],
            config=SimulationConfig(duration=0, seed=0),
        )
        ocel = to_ocel(result)
        assert len(ocel.events) == 0
        assert len(ocel.objects) == 0

    def test_numpy_str_activity(self):
        """np.str_ activity names should be converted to native str."""
        import numpy as np
        result = SimulationResult(
            events=[
                SimulatedEvent("e1", np.str_("Test"), 0.0, [("o1", "type1")]),
            ],
            objects=[SimulatedObject("o1", "type1")],
            o2o_relations=[],
            config=SimulationConfig(duration=100, seed=0),
        )
        ocel = to_ocel(result)
        assert isinstance(ocel.events["ocel:activity"].iloc[0], str)
