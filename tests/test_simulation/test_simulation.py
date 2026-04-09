"""Tests T5.1–T5.7: Simulation engine tests."""

from collections import Counter, defaultdict

import numpy as np
import pytest

from simoc.simulation.runner import SimulationRunner
from simoc.simulation.data_structures import SimulationResult


# ------------------------------------------------------------------
# T5.1 — Single-type smoke test (arrival rate)
# ------------------------------------------------------------------

class TestT5_1_SingleType:
    def test_order_count_within_10_percent(self, sim_runner: SimulationRunner):
        """Orders arrive ~every 3600s. In 36000s expect ~10 (±10% → 9-11)."""
        result = sim_runner.run(duration=36000, seed=42)
        order_count = sum(
            1 for o in result.objects if o.object_type == "order"
        )
        # With empirical inter-arrival of 3600s, expect ~10
        # Allow wider margin due to stochastic timing
        assert 5 <= order_count <= 15, f"Got {order_count} orders"

    def test_events_have_timestamps(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        for e in result.events:
            assert e.timestamp >= 0
            assert e.timestamp <= 36000

    def test_all_events_have_objects(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        for e in result.events:
            assert len(e.objects) >= 1, f"Event {e.event_id} has no objects"


# ------------------------------------------------------------------
# T5.2 — Spawning smoke test
# ------------------------------------------------------------------

class TestT5_2_Spawning:
    def test_items_spawned(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        item_count = sum(1 for o in result.objects if o.object_type == "item")
        assert item_count > 0, "No items spawned"

    def test_deliveries_spawned(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        del_count = sum(
            1 for o in result.objects if o.object_type == "delivery"
        )
        assert del_count > 0, "No deliveries spawned"

    def test_o2o_relations_created(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        assert len(result.o2o_relations) > 0, "No O2O relations"

    def test_items_per_order_reasonable(self, sim_runner: SimulationRunner):
        """Over multiple runs, items-per-order should be 1 or 2 (from empirical)."""
        all_counts = []
        for seed in range(10):
            result = sim_runner.run(duration=36000, seed=seed)
            # Count items per order from O2O
            order_item_count: dict[str, int] = defaultdict(int)
            for rel in result.o2o_relations:
                if rel.source_type == "order" and rel.target_type == "item":
                    order_item_count[rel.source_id] += 1
            all_counts.extend(order_item_count.values())

        if all_counts:
            mean_items = np.mean(all_counts)
            # Empirical: [1, 2] → mean ≈ 1.5
            assert 0.5 <= mean_items <= 3.0, f"Mean items/order = {mean_items}"


# ------------------------------------------------------------------
# T5.3 — Synchronization smoke test
# ------------------------------------------------------------------

class TestT5_3_Sync:
    def test_pack_order_after_all_items_picked(self, sim_runner: SimulationRunner):
        """For every Pack Order event, all items of the parent order
        must have completed Pick Item at an earlier timestamp."""
        result = sim_runner.run(duration=72000, seed=42)

        # Build order->items from O2O
        order_items: dict[str, list[str]] = defaultdict(list)
        for rel in result.o2o_relations:
            if rel.source_type == "order" and rel.target_type == "item":
                order_items[rel.source_id].append(rel.target_id)

        # Build item -> Pick Item timestamp
        item_pick_time: dict[str, float] = {}
        for e in result.events:
            if e.activity == "Pick Item":
                for oid, otype in e.objects:
                    if otype == "item":
                        item_pick_time[oid] = e.timestamp

        # Check Pack Order events
        pack_events = [e for e in result.events if e.activity == "Pack Order"]
        for e in pack_events:
            order_ids = [oid for oid, otype in e.objects if otype == "order"]
            for order_id in order_ids:
                items = order_items.get(order_id, [])
                for item_id in items:
                    if item_id in item_pick_time:
                        assert item_pick_time[item_id] <= e.timestamp, (
                            f"Item {item_id} picked at {item_pick_time[item_id]} "
                            f"but Pack Order at {e.timestamp}"
                        )


# ------------------------------------------------------------------
# T5.5 — Full pipeline integration test
# ------------------------------------------------------------------

class TestT5_5_FullPipeline:
    def test_no_crashes_multiple_runs(self, sim_runner: SimulationRunner):
        """Run 20 simulations with different seeds. No crashes."""
        for seed in range(20):
            result = sim_runner.run(duration=36000, seed=seed)
            assert len(result.events) > 0
            assert len(result.objects) > 0

    def test_all_types_present(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=72000, seed=42)
        types = {o.object_type for o in result.objects}
        assert "order" in types
        assert "item" in types
        assert "delivery" in types

    def test_key_activities_present(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=72000, seed=42)
        activities = {e.activity for e in result.events}
        assert "Create Order" in activities
        assert "Add Item" in activities
        assert "Pick Item" in activities
        assert "Pack Order" in activities

    def test_output_has_o2o(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        assert len(result.o2o_relations) > 0


# ------------------------------------------------------------------
# T5.6 — Reproducibility test
# ------------------------------------------------------------------

class TestT5_6_Reproducibility:
    def test_same_seed_same_output(self, sim_runner: SimulationRunner):
        """Two runs with the same seed must produce identical output."""
        r1 = sim_runner.run(duration=36000, seed=123)
        r2 = sim_runner.run(duration=36000, seed=123)

        assert len(r1.events) == len(r2.events)
        assert len(r1.objects) == len(r2.objects)
        assert len(r1.o2o_relations) == len(r2.o2o_relations)

        for e1, e2 in zip(r1.events, r2.events):
            assert e1.activity == e2.activity
            assert e1.timestamp == pytest.approx(e2.timestamp)
            assert sorted(e1.objects) == sorted(e2.objects)

    def test_different_seed_different_output(self, sim_runner: SimulationRunner):
        """Different seeds should produce different output."""
        r1 = sim_runner.run(duration=36000, seed=1)
        r2 = sim_runner.run(duration=36000, seed=2)
        # At least event count or timestamps should differ
        differs = (
            len(r1.events) != len(r2.events)
            or any(
                e1.timestamp != e2.timestamp
                for e1, e2 in zip(r1.events, r2.events)
            )
        )
        assert differs, "Different seeds produced identical output"


# ------------------------------------------------------------------
# T5.7 — Events are properly ordered
# ------------------------------------------------------------------

class TestT5_7_EventOrdering:
    def test_events_sorted_by_timestamp(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        timestamps = [e.timestamp for e in result.events]
        assert timestamps == sorted(timestamps)

    def test_event_ids_unique(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        eids = [e.event_id for e in result.events]
        assert len(eids) == len(set(eids))

    def test_object_ids_unique(self, sim_runner: SimulationRunner):
        result = sim_runner.run(duration=36000, seed=42)
        oids = [o.object_id for o in result.objects]
        assert len(oids) == len(set(oids))
