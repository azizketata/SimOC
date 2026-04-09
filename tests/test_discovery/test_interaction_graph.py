"""Tests T2.1–T2.3, T2.5: Birth/death, OIG, type classification, DAG."""

import pytest

from simoc.ingestion import OCELData
from simoc.discovery.data_structures import (
    BirthDeathTable,
    ObjectInteractionGraph,
    TypeClassification,
)


# ------------------------------------------------------------------
# T2.1 — Birth and death events
# ------------------------------------------------------------------

class TestT2_1_BirthDeath:
    def test_all_objects_present(self, birth_death: BirthDeathTable):
        assert len(birth_death.df) == 6

    def test_birth_is_first_event(
        self, birth_death: BirthDeathTable, loaded_data: OCELData
    ):
        for _, row in birth_death.df.iterrows():
            oid = row["object_id"]
            expected_eid = loaded_data.lifecycles[oid][0][0]
            assert row["birth_event_id"] == expected_eid, f"{oid} birth mismatch"

    def test_death_is_last_event(
        self, birth_death: BirthDeathTable, loaded_data: OCELData
    ):
        for _, row in birth_death.df.iterrows():
            oid = row["object_id"]
            expected_eid = loaded_data.lifecycles[oid][-1][0]
            assert row["death_event_id"] == expected_eid, f"{oid} death mismatch"

    def test_birth_before_death(self, birth_death: BirthDeathTable):
        for _, row in birth_death.df.iterrows():
            assert row["birth_timestamp"] <= row["death_timestamp"]

    def test_birth_co_objects_order1(self, birth_death: BirthDeathTable):
        """order_1 is born alone in e1 — no co-objects."""
        row = birth_death.df[birth_death.df["object_id"] == "order_1"].iloc[0]
        assert row["birth_co_objects"] == []

    def test_birth_co_objects_item1(self, birth_death: BirthDeathTable):
        """item_1 is born in e2 with order_1."""
        row = birth_death.df[birth_death.df["object_id"] == "item_1"].iloc[0]
        co_oids = [oid for oid, _ in row["birth_co_objects"]]
        assert "order_1" in co_oids

    def test_birth_co_objects_delivery1(self, birth_death: BirthDeathTable):
        """delivery_1 is born in e13 with order_1 and order_2."""
        row = birth_death.df[birth_death.df["object_id"] == "delivery_1"].iloc[0]
        co_oids = sorted([oid for oid, _ in row["birth_co_objects"]])
        assert co_oids == ["order_1", "order_2"]


# ------------------------------------------------------------------
# T2.2 — Object Interaction Graph
# ------------------------------------------------------------------

class TestT2_2_OIG:
    def test_all_ordered_pairs_present(self, oig: ObjectInteractionGraph):
        """3 types → 9 ordered pairs (including self-pairs)."""
        assert len(oig.df) == 9

    def test_order_item_cooccurrence(self, oig: ObjectInteractionGraph):
        """Events e2,e3,e5,e11,e12 involve both order and item → 5."""
        row = oig.df[
            (oig.df["type_1"] == "order") & (oig.df["type_2"] == "item")
        ].iloc[0]
        assert row["e2o_cooccurrence"] == 5

    def test_cooccurrence_symmetric(self, oig: ObjectInteractionGraph):
        """Co-occurrence should be the same for (T1,T2) and (T2,T1)."""
        for _, row in oig.df.iterrows():
            reverse = oig.df[
                (oig.df["type_1"] == row["type_2"])
                & (oig.df["type_2"] == row["type_1"])
            ]
            if not reverse.empty:
                assert row["e2o_cooccurrence"] == reverse.iloc[0]["e2o_cooccurrence"]

    def test_o2o_count_order_item(self, oig: ObjectInteractionGraph):
        """O2O: order->item with qualifier 'contains', count=2 (order_1 has 2 rels)."""
        row = oig.df[
            (oig.df["type_1"] == "order") & (oig.df["type_2"] == "item")
        ].iloc[0]
        assert row["o2o_count"] > 0

    def test_item_birth_has_order(self, oig: ObjectInteractionGraph):
        """100% of item births have an order co-object."""
        row = oig.df[
            (oig.df["type_1"] == "order") & (oig.df["type_2"] == "item")
        ].iloc[0]
        assert row["t2_birth_has_t1_pct"] == pytest.approx(1.0)

    def test_delivery_birth_has_order(self, oig: ObjectInteractionGraph):
        """100% of delivery births have an order co-object."""
        row = oig.df[
            (oig.df["type_1"] == "order") & (oig.df["type_2"] == "delivery")
        ].iloc[0]
        assert row["t2_birth_has_t1_pct"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# T2.3 — Type classification
# ------------------------------------------------------------------

class TestT2_3_Classification:
    def test_order_is_root(self, type_class: TypeClassification):
        assert type_class.classification["order"] == "root"

    def test_item_is_derived(self, type_class: TypeClassification):
        assert type_class.classification["item"] == "derived"

    def test_delivery_is_derived(self, type_class: TypeClassification):
        assert type_class.classification["delivery"] == "derived"

    def test_item_parent_is_order(self, type_class: TypeClassification):
        assert type_class.parent_map["item"] == "order"

    def test_delivery_parent_is_order(self, type_class: TypeClassification):
        assert type_class.parent_map["delivery"] == "order"

    def test_root_not_in_parent_map(self, type_class: TypeClassification):
        assert "order" not in type_class.parent_map


# ------------------------------------------------------------------
# T2.5 — DAG property
# ------------------------------------------------------------------

class TestT2_5_DAG:
    def test_no_self_parent(self, type_class: TypeClassification):
        for child, parent in type_class.parent_map.items():
            assert child != parent

    def test_parent_chain_acyclic(self, type_class: TypeClassification):
        """Follow parent chains — no type should be visited twice."""
        for start in type_class.parent_map:
            visited: set[str] = set()
            current = start
            while current in type_class.parent_map:
                assert current not in visited, f"Cycle at {current}"
                visited.add(current)
                current = type_class.parent_map[current]
