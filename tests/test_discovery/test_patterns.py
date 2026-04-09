"""Tests T4.1–T4.6: Interaction pattern discovery.

Covers synchronization, binding, batching, and release pattern detection
against the sample order process fixture.
"""

import numpy as np
import pytest

from simoc.discovery.data_structures import InteractionPatterns


# ------------------------------------------------------------------
# T4.1 — Synchronization point identification
# ------------------------------------------------------------------

class TestT4_1_SyncDetection:
    def test_pack_order_item_sync_detected(self, interaction_patterns: InteractionPatterns):
        assert ("Pack Order", "item") in interaction_patterns.synchronization_rules

    def test_sync_rule_count(self, interaction_patterns: InteractionPatterns):
        """Pack Order + item is the primary sync. Deliver + delivery is a trivial
        sync (single child converges back with parent after Ship)."""
        assert len(interaction_patterns.synchronization_rules) == 2
        assert ("Pack Order", "item") in interaction_patterns.synchronization_rules
        assert ("Deliver", "delivery") in interaction_patterns.synchronization_rules

    def test_sync_parent_is_order(self, interaction_patterns: InteractionPatterns):
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert rule.parent_type == "order"

    def test_sync_synced_type(self, interaction_patterns: InteractionPatterns):
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert rule.synced_type == "item"

    def test_sync_instance_count(self, interaction_patterns: InteractionPatterns):
        """Two Pack Order events (e11, e12) are sync points."""
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert rule.n_instances == 2

    def test_sync_delays(self, interaction_patterns: InteractionPatterns):
        """Both sync delays should be 5400 seconds (1.5 hours)."""
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert sorted(rule.raw_sync_delays) == pytest.approx([5400.0, 5400.0])

    def test_wait_spreads(self, interaction_patterns: InteractionPatterns):
        """e11: spread=1800s (30 min), e12: spread=0s (1 child)."""
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert sorted(rule.raw_wait_spreads) == pytest.approx([0.0, 1800.0])


# ------------------------------------------------------------------
# T4.2 — Synchronization condition classification
# ------------------------------------------------------------------

class TestT4_2_SyncCondition:
    def test_condition_all(self, interaction_patterns: InteractionPatterns):
        """Both instances have all children present → condition = ALL."""
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert rule.condition == "ALL"

    def test_sync_delay_dist_samplable(self, interaction_patterns: InteractionPatterns):
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        samples = rule.sync_delay_dist.rvs(size=50, rng=np.random.default_rng(42))
        assert len(samples) == 50

    def test_wait_spread_dist_exists(self, interaction_patterns: InteractionPatterns):
        """At least one non-zero spread exists (e11: 1800s), so dist should exist."""
        rule = interaction_patterns.synchronization_rules[("Pack Order", "item")]
        assert rule.wait_spread_dist is not None


# ------------------------------------------------------------------
# T4.3 — Binding relation identification
# ------------------------------------------------------------------

class TestT4_3_Binding:
    def test_no_binding_detected(self, interaction_patterns: InteractionPatterns):
        """All O2O relations in fixture are spawning. No binding expected."""
        assert len(interaction_patterns.binding_policies) == 0


# ------------------------------------------------------------------
# T4.5 — Batching trigger classification
# ------------------------------------------------------------------

class TestT4_5_Batching:
    def test_ship_order_batching_detected(self, interaction_patterns: InteractionPatterns):
        """Ship batches orders from different parents (root type)."""
        assert ("Ship", "order") in interaction_patterns.batching_rules

    def test_ship_trigger_unknown(self, interaction_patterns: InteractionPatterns):
        """Only 1 Ship event — trigger type cannot be determined."""
        rule = interaction_patterns.batching_rules[("Ship", "order")]
        assert rule.trigger_type == "unknown"

    def test_ship_batch_size(self, interaction_patterns: InteractionPatterns):
        """2 orders in the single Ship event."""
        rule = interaction_patterns.batching_rules[("Ship", "order")]
        assert rule.raw_batch_sizes == [2]

    def test_ship_instance_count(self, interaction_patterns: InteractionPatterns):
        rule = interaction_patterns.batching_rules[("Ship", "order")]
        assert rule.n_instances == 1

    def test_deliver_not_new_batching(self, interaction_patterns: InteractionPatterns):
        """Deliver has same objects as Ship from the same preceding event.
        Not new batching — just continuation."""
        assert ("Deliver", "order") not in interaction_patterns.batching_rules


# ------------------------------------------------------------------
# T4.6 — Release point correctness
# ------------------------------------------------------------------

class TestT4_6_Release:
    def test_order_item_release_detected(self, interaction_patterns: InteractionPatterns):
        """Orders and items release (decouple)."""
        # Check both orderings since types are sorted
        has_rule = (
            ("item", "order") in interaction_patterns.release_rules
            or ("order", "item") in interaction_patterns.release_rules
        )
        assert has_rule

    def test_release_at_pack_order(self, interaction_patterns: InteractionPatterns):
        key = (
            ("item", "order")
            if ("item", "order") in interaction_patterns.release_rules
            else ("order", "item")
        )
        rule = interaction_patterns.release_rules[key]
        assert rule.release_activity == "Pack Order"

    def test_release_deterministic(self, interaction_patterns: InteractionPatterns):
        key = (
            ("item", "order")
            if ("item", "order") in interaction_patterns.release_rules
            else ("order", "item")
        )
        rule = interaction_patterns.release_rules[key]
        assert rule.release_condition == "deterministic"
        assert rule.probability == pytest.approx(1.0)

    def test_release_instance_count(self, interaction_patterns: InteractionPatterns):
        """3 (order, item) pairs all release at Pack Order."""
        key = (
            ("item", "order")
            if ("item", "order") in interaction_patterns.release_rules
            else ("order", "item")
        )
        rule = interaction_patterns.release_rules[key]
        assert rule.n_instances == 3

    def test_order_delivery_no_release(self, interaction_patterns: InteractionPatterns):
        """Orders and deliveries end together (joint termination) — no release."""
        assert ("delivery", "order") not in interaction_patterns.release_rules
        assert ("order", "delivery") not in interaction_patterns.release_rules

    def test_item_delivery_no_release(self, interaction_patterns: InteractionPatterns):
        """Items and deliveries never co-occur — no release possible."""
        assert ("delivery", "item") not in interaction_patterns.release_rules
        assert ("item", "delivery") not in interaction_patterns.release_rules
