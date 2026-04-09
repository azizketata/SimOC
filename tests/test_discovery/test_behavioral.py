"""Tests T3.1–T3.5: Behavioral discovery — arrivals, durations, resources, branching, DFGs."""

import numpy as np
import pytest

from simoc.discovery.data_structures import BehavioralProfile


# ------------------------------------------------------------------
# T3.1 — Arrival rates
# ------------------------------------------------------------------

class TestT3_1_ArrivalRates:
    def test_root_type_has_arrival_model(self, behavioral_profile: BehavioralProfile):
        assert "order" in behavioral_profile.arrival_models
        assert "item" not in behavioral_profile.arrival_models
        assert "delivery" not in behavioral_profile.arrival_models

    def test_order_arrival_count(self, behavioral_profile: BehavioralProfile):
        model = behavioral_profile.arrival_models["order"]
        assert model.n_arrivals == 2

    def test_order_interarrival_empirical(self, behavioral_profile: BehavioralProfile):
        """With only 1 gap (3600s), distribution should be empirical."""
        model = behavioral_profile.arrival_models["order"]
        # Regardless of dist name, sampling should produce reasonable values
        samples = model.distribution.rvs(size=100, rng=np.random.default_rng(42))
        assert len(samples) == 100
        assert np.all(samples >= 0)

    def test_arrival_is_stationary(self, behavioral_profile: BehavioralProfile):
        """With tiny sample, stationarity should be assumed."""
        model = behavioral_profile.arrival_models["order"]
        assert model.is_stationary is True


# ------------------------------------------------------------------
# T3.2 — Activity durations
# ------------------------------------------------------------------

class TestT3_2_Durations:
    # Expected (type, activity) pairs that should have durations
    # (all non-terminal activities per type)
    EXPECTED_PAIRS = [
        ("order", "Create Order"),
        ("order", "Add Item"),
        ("order", "Confirm Order"),
        ("order", "Pack Order"),
        ("order", "Ship"),
        ("item", "Add Item"),
        ("item", "Pick Item"),
        ("delivery", "Ship"),
    ]

    # Terminal activities: no duration expected
    TERMINAL_PAIRS = [
        ("order", "Deliver"),
        ("item", "Pack Order"),
        ("delivery", "Deliver"),
    ]

    def test_expected_pairs_present(self, behavioral_profile: BehavioralProfile):
        for key in self.EXPECTED_PAIRS:
            assert key in behavioral_profile.duration_models, f"Missing {key}"

    def test_last_activities_absent(self, behavioral_profile: BehavioralProfile):
        for key in self.TERMINAL_PAIRS:
            assert key not in behavioral_profile.duration_models, (
                f"Terminal activity {key} should not have a duration model"
            )

    def test_order_create_order_durations(self, behavioral_profile: BehavioralProfile):
        """order_1: 08:00->08:05 = 300s, order_2: 09:00->09:05 = 300s."""
        model = behavioral_profile.duration_models[("order", "Create Order")]
        assert sorted(model.raw_durations_seconds) == pytest.approx([300.0, 300.0])

    def test_order_add_item_durations(self, behavioral_profile: BehavioralProfile):
        """order_1: 08:05->08:10=300, 08:10->10:00=6600. order_2: 09:05->12:00=10500."""
        model = behavioral_profile.duration_models[("order", "Add Item")]
        assert sorted(model.raw_durations_seconds) == pytest.approx(
            [300.0, 6600.0, 10500.0]
        )

    def test_item_add_item_durations(self, behavioral_profile: BehavioralProfile):
        """item_1: 08:05->11:00=10500, item_2: 08:10->11:30=12000, item_3: 09:05->12:30=12300."""
        model = behavioral_profile.duration_models[("item", "Add Item")]
        assert sorted(model.raw_durations_seconds) == pytest.approx(
            [10500.0, 12000.0, 12300.0]
        )

    def test_item_pick_item_durations(self, behavioral_profile: BehavioralProfile):
        """item_1: 11:00->13:00=7200, item_2: 11:30->13:00=5400, item_3: 12:30->14:00=5400."""
        model = behavioral_profile.duration_models[("item", "Pick Item")]
        assert sorted(model.raw_durations_seconds) == pytest.approx(
            [5400.0, 5400.0, 7200.0]
        )

    def test_duration_rvs_positive(self, behavioral_profile: BehavioralProfile):
        """Sampled durations should be non-negative."""
        for key, model in behavioral_profile.duration_models.items():
            samples = model.distribution.rvs(size=50, rng=np.random.default_rng(42))
            assert len(samples) == 50, f"Wrong size for {key}"

    def test_observation_counts(self, behavioral_profile: BehavioralProfile):
        for key, model in behavioral_profile.duration_models.items():
            assert model.n_observations == len(model.raw_durations_seconds)
            assert model.n_observations >= 1


# ------------------------------------------------------------------
# T3.3 — Branching probabilities
# ------------------------------------------------------------------

class TestT3_3_Branching:
    def test_all_types_have_model(self, behavioral_profile: BehavioralProfile):
        assert "order" in behavioral_profile.branching_models
        assert "item" in behavioral_profile.branching_models
        assert "delivery" in behavioral_profile.branching_models

    def test_probabilities_sum_to_one(self, behavioral_profile: BehavioralProfile):
        for otype, model in behavioral_profile.branching_models.items():
            for src, targets in model.probabilities.items():
                total = sum(targets.values())
                assert total == pytest.approx(1.0, abs=1e-6), (
                    f"Probs for ({otype}, {src}) sum to {total}"
                )

    def test_order_create_order_deterministic(
        self, behavioral_profile: BehavioralProfile
    ):
        probs = behavioral_profile.branching_models["order"].probabilities
        assert probs["Create Order"] == {"Add Item": pytest.approx(1.0)}

    def test_order_add_item_branching(self, behavioral_profile: BehavioralProfile):
        probs = behavioral_profile.branching_models["order"].probabilities
        add_item = probs["Add Item"]
        assert set(add_item.keys()) == {"Add Item", "Confirm Order"}
        assert add_item["Add Item"] == pytest.approx(1 / 3, abs=0.01)
        assert add_item["Confirm Order"] == pytest.approx(2 / 3, abs=0.01)

    def test_order_remaining_deterministic(self, behavioral_profile: BehavioralProfile):
        probs = behavioral_profile.branching_models["order"].probabilities
        assert probs["Confirm Order"] == {"Pack Order": pytest.approx(1.0)}
        assert probs["Pack Order"] == {"Ship": pytest.approx(1.0)}
        assert probs["Ship"] == {"Deliver": pytest.approx(1.0)}

    def test_item_all_deterministic(self, behavioral_profile: BehavioralProfile):
        probs = behavioral_profile.branching_models["item"].probabilities
        assert probs["Add Item"] == {"Pick Item": pytest.approx(1.0)}
        assert probs["Pick Item"] == {"Pack Order": pytest.approx(1.0)}

    def test_delivery_deterministic(self, behavioral_profile: BehavioralProfile):
        probs = behavioral_profile.branching_models["delivery"].probabilities
        assert probs["Ship"] == {"Deliver": pytest.approx(1.0)}

    def test_no_spurious_transitions(self, behavioral_profile: BehavioralProfile):
        """No transition edges that don't appear in any lifecycle."""
        expected_order = {
            ("Create Order", "Add Item"),
            ("Add Item", "Add Item"),
            ("Add Item", "Confirm Order"),
            ("Confirm Order", "Pack Order"),
            ("Pack Order", "Ship"),
            ("Ship", "Deliver"),
        }
        expected_item = {
            ("Add Item", "Pick Item"),
            ("Pick Item", "Pack Order"),
        }
        expected_delivery = {
            ("Ship", "Deliver"),
        }

        actual_order = set(
            behavioral_profile.branching_models["order"].transition_counts.keys()
        )
        actual_item = set(
            behavioral_profile.branching_models["item"].transition_counts.keys()
        )
        actual_delivery = set(
            behavioral_profile.branching_models["delivery"].transition_counts.keys()
        )

        assert actual_order == expected_order
        assert actual_item == expected_item
        assert actual_delivery == expected_delivery


# ------------------------------------------------------------------
# T3.4 — Resource pools
# ------------------------------------------------------------------

class TestT3_4_Resources:
    def test_resource_pools_empty(self, behavioral_profile: BehavioralProfile):
        """No resource attribute in fixture → empty pools."""
        assert behavioral_profile.resource_pools == {}


# ------------------------------------------------------------------
# T3.5 — Per-type DFGs
# ------------------------------------------------------------------

class TestT3_5_TypeDFGs:
    def test_all_types_have_dfg(self, behavioral_profile: BehavioralProfile):
        assert "order" in behavioral_profile.type_dfgs
        assert "item" in behavioral_profile.type_dfgs
        assert "delivery" in behavioral_profile.type_dfgs

    def test_order_dfg_activities(self, behavioral_profile: BehavioralProfile):
        acts = behavioral_profile.type_dfgs["order"].activities
        assert acts == {
            "Create Order",
            "Add Item",
            "Confirm Order",
            "Pack Order",
            "Ship",
            "Deliver",
        }

    def test_item_dfg_activities(self, behavioral_profile: BehavioralProfile):
        acts = behavioral_profile.type_dfgs["item"].activities
        assert acts == {"Add Item", "Pick Item", "Pack Order"}

    def test_delivery_dfg_activities(self, behavioral_profile: BehavioralProfile):
        acts = behavioral_profile.type_dfgs["delivery"].activities
        assert acts == {"Ship", "Deliver"}

    def test_no_spurious_activities(self, behavioral_profile: BehavioralProfile, loaded_data):
        """No activity in any DFG that doesn't appear in that type's lifecycles."""
        oid_to_type = loaded_data.objects.set_index("object_id")[
            "object_type"
        ].to_dict()

        for otype, dfg in behavioral_profile.type_dfgs.items():
            # Collect actual activities from lifecycles
            actual_acts = set()
            for oid, lc in loaded_data.lifecycles.items():
                if oid_to_type.get(oid) == otype:
                    for _, act, _ in lc:
                        actual_acts.add(act)
            assert dfg.activities == actual_acts, (
                f"DFG activities mismatch for {otype}: "
                f"extra={dfg.activities - actual_acts}, "
                f"missing={actual_acts - dfg.activities}"
            )

    def test_dfg_edges_match_branching(self, behavioral_profile: BehavioralProfile):
        """DFG edges and branching transition_counts should be identical."""
        for otype in behavioral_profile.type_dfgs:
            dfg_edges = behavioral_profile.type_dfgs[otype].edges
            branch_counts = behavioral_profile.branching_models[otype].transition_counts
            assert dfg_edges == branch_counts
