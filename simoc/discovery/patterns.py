"""Tasks 4.1–4.4: Interaction pattern discovery.

Discovers synchronization, binding, batching, and release patterns
from OCEL 2.0 event logs. This is the core novel contribution of SimOC.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from simoc.ingestion.data_structures import OCELData
from simoc.discovery.data_structures import (
    BatchingRule,
    BindingPolicy,
    BirthDeathTable,
    ContinuousFittedDistribution,
    InteractionPatterns,
    ReleaseRule,
    SynchronizationRule,
    TypeClassification,
)
from simoc.discovery.behavioral import _fit_continuous
from simoc.discovery.cardinality import _fit_cardinality, _group_children_by_parent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def discover_patterns(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> InteractionPatterns:
    """Orchestrator: discover all interaction patterns (Tasks 4.1–4.4)."""
    sync_rules = discover_synchronization(data, birth_death, type_classification)
    binding_policies = discover_binding(data, birth_death)
    batching_rules = discover_batching(data, birth_death, type_classification)
    release_rules = discover_release(data)

    logger.info(
        "Pattern discovery complete: %d sync, %d binding, %d batching, %d release.",
        len(sync_rules),
        len(binding_policies),
        len(batching_rules),
        len(release_rules),
    )
    return InteractionPatterns(
        synchronization_rules=sync_rules,
        binding_policies=binding_policies,
        batching_rules=batching_rules,
        release_rules=release_rules,
    )


# ------------------------------------------------------------------
# Task 4.1: Synchronization Discovery
# ------------------------------------------------------------------


def discover_synchronization(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> dict[tuple[str, str], SynchronizationRule]:
    """Discover synchronization points where sibling objects converge."""
    rules: dict[tuple[str, str], SynchronizationRule] = {}

    # For each derived type, find events where children of the same parent converge
    for child_type, parent_type in type_classification.parent_map.items():
        parent_to_children = _group_children_by_parent(
            data, birth_death, parent_type, child_type
        )

        # Reverse map: child_oid -> parent_oid
        child_to_parent: dict[str, str] = {}
        for pid, children in parent_to_children.items():
            for cid in children:
                child_to_parent[cid] = pid

        # Scan events for sync points
        # Key: (activity, child_type) -> list of sync instances
        sync_instances: dict[str, list[dict]] = defaultdict(list)

        for _, event_row in data.events.iterrows():
            eid = event_row["event_id"]
            activity = event_row["activity"]
            ts = event_row["timestamp"]
            related = event_row["related_objects"]

            # Find children of each parent in this event
            # Only count children that had prior events (not being born here)
            parent_children_in_event: dict[str, list[str]] = defaultdict(list)
            for oid, otype in related:
                if otype == child_type and oid in child_to_parent:
                    # Skip if this is the child's birth event (spawning, not sync)
                    child_lc = data.lifecycles.get(oid, [])
                    if child_lc and child_lc[0][0] == eid:
                        continue  # born at this event
                    pid = child_to_parent[oid]
                    parent_children_in_event[pid].append(oid)

            # Check if parent is also in the event (for sync, parent waits)
            parent_oids_in_event = {oid for oid, otype in related if otype == parent_type}

            for pid, children_present in parent_children_in_event.items():
                total_children = len(parent_to_children.get(pid, []))
                if total_children == 0:
                    continue

                # Sync requires: parent present AND at least 1 child
                # (single-child parents count as trivial sync)
                if pid not in parent_oids_in_event:
                    continue

                # Compute ready times for each child
                ready_times = []
                for cid in children_present:
                    rt = _compute_ready_time(data.lifecycles[cid], eid)
                    ready_times.append(rt)

                if not ready_times:
                    continue

                earliest_ready = min(ready_times)
                latest_ready = max(ready_times)
                sync_delay = (ts - latest_ready).total_seconds()
                wait_spread = (latest_ready - earliest_ready).total_seconds()

                sync_instances[activity].append(
                    {
                        "event_id": eid,
                        "parent_oid": pid,
                        "children_present": len(children_present),
                        "total_children": total_children,
                        "sync_delay": sync_delay,
                        "wait_spread": wait_spread,
                    }
                )

        # Build rules from collected instances
        for activity, instances in sync_instances.items():
            if not instances:
                continue

            delays = [inst["sync_delay"] for inst in instances]
            spreads = [inst["wait_spread"] for inst in instances]
            children_present = [inst["children_present"] for inst in instances]
            total_children = [inst["total_children"] for inst in instances]

            condition = _classify_sync_condition(children_present, total_children)

            delay_dist = _fit_continuous(np.array(delays))
            nonzero_spreads = [s for s in spreads if s > 0]
            spread_dist = (
                _fit_continuous(np.array(nonzero_spreads))
                if nonzero_spreads
                else None
            )

            key = (activity, child_type)
            rules[key] = SynchronizationRule(
                activity=activity,
                synced_type=child_type,
                parent_type=parent_type,
                condition=condition,
                sync_delay_dist=delay_dist,
                wait_spread_dist=spread_dist,
                n_instances=len(instances),
                raw_sync_delays=delays,
                raw_wait_spreads=spreads,
            )
            logger.info(
                "Sync rule (%s, %s): condition=%s, n=%d, mean_delay=%.0fs",
                activity,
                child_type,
                condition,
                len(instances),
                np.mean(delays),
            )

    return rules


# ------------------------------------------------------------------
# Task 4.2: Binding Discovery
# ------------------------------------------------------------------


def discover_binding(
    data: OCELData,
    birth_death: BirthDeathTable,
) -> dict[tuple[str, str], BindingPolicy]:
    """Discover binding relations created mid-process (not at birth)."""
    if data.o2o.empty:
        logger.info("No O2O relations found. No binding to discover.")
        return {}

    # Build birth co-object and birth timestamp lookups
    birth_co_lookup: dict[str, set[str]] = {}
    birth_ts_lookup: dict[str, pd.Timestamp] = {}
    for _, row in birth_death.df.iterrows():
        oid = row["object_id"]
        birth_co_lookup[oid] = {co_oid for co_oid, _ in row["birth_co_objects"]}
        birth_ts_lookup[oid] = row["birth_timestamp"]

    # Check each O2O relation: spawning or binding?
    # Spawning: both objects born at the same time (or target born at source's birth)
    # Binding: one object PRE-EXISTED before the other was born, then they
    #          co-occur at the newer object's birth event. The pre-existing
    #          object was BOUND into the newer one, not spawned by it.
    binding_relations: list[dict] = []
    for _, row in data.o2o.iterrows():
        src = row["source_object_id"]
        tgt = row["target_object_id"]

        src_at_tgt_birth = src in birth_co_lookup.get(tgt, set())
        tgt_at_src_birth = tgt in birth_co_lookup.get(src, set())

        src_ts = birth_ts_lookup.get(src)
        tgt_ts = birth_ts_lookup.get(tgt)

        is_binding = False

        if src_ts is not None and tgt_ts is not None:
            if tgt_ts < src_ts and tgt_at_src_birth:
                # Target pre-existed, then appeared at SOURCE's birth.
                # The SOURCE is the new object absorbing old targets → BINDING.
                # Example: package (src, new) born with pre-existing items (tgt, old)
                is_binding = True
            elif src_ts < tgt_ts and src_at_tgt_birth:
                # Source pre-existed, target is born at this event.
                # The TARGET is the new object created by the source → SPAWNING.
                # Example: order (src, old) creates item (tgt, new)
                is_binding = False
            elif not src_at_tgt_birth and not tgt_at_src_birth:
                # Neither present at the other's birth → BINDING (created later)
                is_binding = True
            # else: born at the same event and co-present → SPAWNING
        else:
            if not src_at_tgt_birth and not tgt_at_src_birth:
                is_binding = True

        if is_binding:
            binding_relations.append(
                {
                    "source": src,
                    "target": tgt,
                    "source_type": row["source_type"],
                    "target_type": row["target_type"],
                    "qualifier": row.get("qualifier", ""),
                }
            )

    if not binding_relations:
        logger.info("All O2O relations are spawning. No binding detected.")
        return {}

    # Group binding relations by type pair
    # Full binding ML pipeline would go here for real logs
    logger.info(
        "Found %d binding relations. ML pipeline not yet exercised "
        "(requires richer dataset).",
        len(binding_relations),
    )

    policies: dict[tuple[str, str], BindingPolicy] = {}
    type_pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rel in binding_relations:
        key = (rel["source_type"], rel["target_type"])
        type_pairs[key].append(rel)

    for (src_type, tgt_type), rels in type_pairs.items():
        # Find the binding activity: first co-occurrence after birth
        binding_activity = _find_binding_activity(data, rels)

        policies[(src_type, tgt_type)] = BindingPolicy(
            source_type=src_type,
            target_type=tgt_type,
            binding_activity=binding_activity,
            model=None,
            feature_names=[],
            capacity_dist=None,
            hard_constraints={},
            n_instances=len(rels),
            f1_score=None,
        )

    return policies


# ------------------------------------------------------------------
# Task 4.3: Batching Discovery
# ------------------------------------------------------------------


def discover_batching(
    data: OCELData,
    birth_death: BirthDeathTable,
    type_classification: TypeClassification,
) -> dict[tuple[str, str], BatchingRule]:
    """Discover batching points where unrelated objects are grouped."""
    # Build parent lookup for derived types
    child_to_parent: dict[str, str] = {}
    for child_type, parent_type in type_classification.parent_map.items():
        parent_to_children = _group_children_by_parent(
            data, birth_death, parent_type, child_type
        )
        for pid, children in parent_to_children.items():
            for cid in children:
                child_to_parent[cid] = pid

    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()

    # Scan events for batching candidates
    batch_events: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for _, event_row in data.events.iterrows():
        eid = event_row["event_id"]
        activity = event_row["activity"]
        ts = event_row["timestamp"]
        related = event_row["related_objects"]

        # Group objects by type
        objs_by_type: dict[str, list[str]] = defaultdict(list)
        for oid, otype in related:
            objs_by_type[otype].append(oid)

        for otype, oids in objs_by_type.items():
            if len(oids) < 2:
                continue

            # Check 1: Do they share a common parent? If yes, this is sync.
            if _objects_share_parent(oids, child_to_parent):
                continue

            # Check 2: Do they have different ready times?
            # If all came from the same preceding event, this is continuation.
            preceding_events = set()
            for oid in oids:
                prev_eid = _get_preceding_event_id(data.lifecycles.get(oid, []), eid)
                if prev_eid is not None:
                    preceding_events.add(prev_eid)

            if len(preceding_events) <= 1 and preceding_events:
                # All objects came from the same event — continuation, not batching
                continue

            batch_events[(activity, otype)].append(
                {
                    "event_id": eid,
                    "timestamp": ts,
                    "batch_size": len(oids),
                    "object_ids": oids,
                }
            )

    # Build batching rules
    rules: dict[tuple[str, str], BatchingRule] = {}
    for (activity, otype), events in batch_events.items():
        batch_sizes = [e["batch_size"] for e in events]
        timestamps = [e["timestamp"] for e in events]

        trigger_type, trigger_params = _classify_trigger_type(timestamps, batch_sizes)

        size_dist = _fit_cardinality(batch_sizes)

        key = (activity, otype)
        rules[key] = BatchingRule(
            activity=activity,
            batched_type=otype,
            trigger_type=trigger_type,
            trigger_params=trigger_params,
            batch_size_dist=size_dist,
            n_instances=len(events),
            raw_batch_sizes=batch_sizes,
        )
        logger.info(
            "Batching rule (%s, %s): trigger=%s, n=%d, mean_size=%.1f",
            activity,
            otype,
            trigger_type,
            len(events),
            np.mean(batch_sizes),
        )

    return rules


# ------------------------------------------------------------------
# Task 4.4: Release Discovery
# ------------------------------------------------------------------


def discover_release(data: OCELData) -> dict[tuple[str, str], ReleaseRule]:
    """Discover where co-traveling object types decouple."""
    oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()

    # Build event timestamp and activity lookup
    event_ts: dict[str, pd.Timestamp] = {}
    event_act: dict[str, str] = {}
    for _, row in data.events.iterrows():
        event_ts[row["event_id"]] = row["timestamp"]
        event_act[row["event_id"]] = row["activity"]

    # Build per-object: last event timestamp and last event id
    obj_last_ts: dict[str, pd.Timestamp] = {}
    for oid, lc in data.lifecycles.items():
        if lc:
            obj_last_ts[oid] = lc[-1][2]

    # For efficiency, work at the TYPE-PAIR level with sampling.
    # For each event, group objects by type. For each cross-type pair
    # of types in the event, track the last co-occurrence per instance pair.
    # Use a compact representation: for each (oid1, oid2), only keep the
    # last shared event (overwrite as we scan chronologically).
    unique_types = sorted(set(oid_to_type.values()))
    type_pairs_to_check = [
        (t1, t2) for i, t1 in enumerate(unique_types)
        for t2 in unique_types[i + 1:]
    ]

    # For each type pair, sample instance pairs to check (cap at 200 for perf)
    release_candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for t1, t2 in type_pairs_to_check:
        # Sample instances for efficiency (cap each type at 100)
        t1_all = [oid for oid, t in oid_to_type.items() if t == t1]
        t2_all = [oid for oid, t in oid_to_type.items() if t == t2]

        import random as _random
        _rng = _random.Random(42)
        t1_sample = set(t1_all if len(t1_all) <= 100 else _rng.sample(t1_all, 100))
        t2_sample = set(t2_all if len(t2_all) <= 100 else _rng.sample(t2_all, 100))

        # Find last co-occurrence per sampled pair
        pair_last: dict[tuple[str, str], str] = {}
        for eid, oids in data.e2o_index.event_to_objects.items():
            oids_t1 = [o for o in oids if o in t1_sample]
            oids_t2 = [o for o in oids if o in t2_sample]
            if not oids_t1 or not oids_t2:
                continue
            ts = event_ts[eid]
            for o1 in oids_t1:
                for o2 in oids_t2:
                    key = (o1, o2)
                    if key not in pair_last or ts > event_ts[pair_last[key]]:
                        pair_last[key] = eid

        if not pair_last:
            continue

        for (o1, o2), last_eid in pair_last.items():
            last_ts_val = event_ts[last_eid]
            last_act = event_act[last_eid]

            o1_continues = obj_last_ts.get(o1, last_ts_val) > last_ts_val
            o2_continues = obj_last_ts.get(o2, last_ts_val) > last_ts_val

            if o1_continues or o2_continues:
                release_candidates[(t1, t2)].append(
                    {
                        "oid1": o1,
                        "oid2": o2,
                        "release_eid": last_eid,
                        "release_activity": last_act,
                    }
                )

    # Aggregate release rules per type pair
    rules: dict[tuple[str, str], ReleaseRule] = {}

    for type_key, candidates in release_candidates.items():
        if not candidates:
            continue

        # Count releases per activity
        activity_counts = Counter(c["release_activity"] for c in candidates)
        dominant_activity, dominant_count = activity_counts.most_common(1)[0]
        total = len(candidates)
        probability = dominant_count / total

        condition = (
            "deterministic"
            if probability >= 0.9
            else f"probabilistic(p={probability:.2f})"
        )

        rules[type_key] = ReleaseRule(
            type_1=type_key[0],
            type_2=type_key[1],
            release_activity=dominant_activity,
            release_condition=condition,
            probability=probability,
            n_instances=total,
        )
        logger.info(
            "Release rule (%s, %s): activity=%s, condition=%s, n=%d",
            type_key[0],
            type_key[1],
            dominant_activity,
            condition,
            total,
        )

    return rules


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _compute_ready_time(
    lifecycle: list[tuple[str, str, pd.Timestamp]], sync_eid: str
) -> pd.Timestamp:
    """Get timestamp of event immediately before sync_eid in lifecycle."""
    for i, (eid, _, ts) in enumerate(lifecycle):
        if eid == sync_eid:
            if i > 0:
                return lifecycle[i - 1][2]
            return ts  # born at sync event
    raise ValueError(f"Event {sync_eid} not found in lifecycle")


def _classify_sync_condition(
    children_present: list[int], total_children: list[int]
) -> str:
    """Classify sync condition as ALL, THRESHOLD, etc."""
    if not children_present:
        return "ALL"

    all_present = sum(
        1 for cp, tc in zip(children_present, total_children) if cp >= tc
    )
    fraction = all_present / len(children_present)

    if fraction >= 0.9:
        return "ALL"

    median_fraction = np.median(
        [cp / tc for cp, tc in zip(children_present, total_children) if tc > 0]
    )
    return f"THRESHOLD({median_fraction:.2f})"


def _objects_share_parent(
    oids: list[str], child_to_parent: dict[str, str]
) -> bool:
    """Check if all given oids share a common parent."""
    parents = set()
    for oid in oids:
        parent = child_to_parent.get(oid)
        if parent is None:
            return False  # root type, no parent → can't share
        parents.add(parent)

    return len(parents) == 1


def _get_preceding_event_id(
    lifecycle: list[tuple[str, str, pd.Timestamp]], target_eid: str
) -> str | None:
    """Get the event_id immediately before target_eid in lifecycle."""
    for i, (eid, _, _) in enumerate(lifecycle):
        if eid == target_eid:
            return lifecycle[i - 1][0] if i > 0 else None
    return None


def _classify_trigger_type(
    timestamps: list[pd.Timestamp], batch_sizes: list[int]
) -> tuple[str, dict[str, float]]:
    """Classify batching trigger type."""
    if len(timestamps) < 3:
        return "unknown", {}

    # Test schedule: low CV of inter-batch times
    if len(timestamps) >= 2:
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean_gap = np.mean(gaps)
        if mean_gap > 0:
            cv_gaps = np.std(gaps) / mean_gap
            if cv_gaps < 0.3:
                return "schedule", {"interval_seconds": mean_gap}

    # Test threshold: low CV of batch sizes
    mean_size = np.mean(batch_sizes)
    if mean_size > 0:
        cv_sizes = np.std(batch_sizes) / mean_size
        if cv_sizes < 0.3:
            return "threshold", {"threshold": float(mean_size)}

    return "hybrid", {}


def _find_binding_activity(data: OCELData, relations: list[dict]) -> str:
    """Find the activity where binding relations are created."""
    # Pre-build event lookups for O(1) access
    event_ts: dict[str, pd.Timestamp] = {}
    event_act: dict[str, str] = {}
    for _, row in data.events.iterrows():
        event_ts[row["event_id"]] = row["timestamp"]
        event_act[row["event_id"]] = row["activity"]

    activities: list[str] = []
    # Sample up to 500 relations for performance
    sample = relations if len(relations) <= 500 else relations[:500]
    for rel in sample:
        src, tgt = rel["source"], rel["target"]
        src_events = data.e2o_index.object_to_events.get(src, set())
        tgt_events = data.e2o_index.object_to_events.get(tgt, set())
        shared = src_events & tgt_events
        if shared:
            first_eid = min(shared, key=lambda e: event_ts.get(e, pd.Timestamp.max))
            if first_eid in event_act:
                activities.append(event_act[first_eid])

    if activities:
        return Counter(activities).most_common(1)[0][0]
    return ""
