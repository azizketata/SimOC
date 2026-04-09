"""Interaction Mediator: central coordinator for cross-type interaction patterns."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Generator

import numpy as np
import simpy

from simoc.discovery.data_structures import (
    BehavioralProfile,
    BirthDeathTable,
    BranchingModel,
    InteractionPatterns,
    SpawningProfile,
    TypeClassification,
)
from simoc.simulation.data_structures import (
    SimulatedEvent,
    SimulatedO2O,
    SimulatedObject,
    SimulationConfig,
)

if TYPE_CHECKING:
    from simoc.simulation.agent import Agent, DerivedAgent

logger = logging.getLogger(__name__)


class InteractionMediator:
    """Central coordinator for spawning, sync, batching, binding, and release."""

    def __init__(
        self,
        env: simpy.Environment,
        type_classification: TypeClassification,
        spawning_profiles: dict[tuple[str, str], SpawningProfile],
        behavioral_profile: BehavioralProfile,
        interaction_patterns: InteractionPatterns,
        birth_death: BirthDeathTable,
        rng: np.random.Generator,
        config: SimulationConfig,
    ):
        self.env = env
        self.type_classification = type_classification
        self.spawning_profiles = spawning_profiles
        self.behavioral = behavioral_profile
        self.patterns = interaction_patterns
        self.birth_death = birth_death
        self.rng = rng
        self.config = config

        # Output collectors
        self._events: list[SimulatedEvent] = []
        self._objects: list[SimulatedObject] = []
        self._o2o_relations: list[SimulatedO2O] = []
        self._event_counter: int = 0

        # Agent registry
        self._agents: dict[str, Agent] = {}
        self._parent_children: dict[str, list[str]] = defaultdict(list)
        self._child_parent: dict[str, str] = {}

        # Pending children: parent_oid -> {birth_activity -> [child agents]}
        self._pending_children: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Sync state
        self._sync_events: dict[tuple[str, str], simpy.Event] = {}
        self._sync_ready: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._sync_target_count: dict[tuple[str, str], int] = {}

        # Batch state
        self._batch_queues: dict[tuple[str, str], list[tuple]] = defaultdict(list)

        # Binding state: agents ready to be bound into a new object
        # Key: (source_type, target_type) from binding policy
        # Value: list of target agents ready for binding
        self._binding_queues: dict[tuple[str, str], list] = defaultdict(list)

        # Precompute binding thresholds from discovered cardinality
        self._binding_thresholds: dict[tuple[str, str], int] = {}
        self._precompute_binding_thresholds()

        # ID generation
        self._id_counters: dict[str, int] = defaultdict(int)

        # Precompute start activities per type
        self._start_activities: dict[str, str] = {}
        self._compute_start_activities()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _compute_start_activities(self) -> None:
        """Determine the first activity for each object type from the DFG."""
        for otype, dfg in self.behavioral.type_dfgs.items():
            targets = {tgt for (_, tgt) in dfg.edges.keys()}
            sources = {src for (src, _) in dfg.edges.keys()}
            starts = sources - targets
            if starts:
                self._start_activities[otype] = sorted(starts)[0]
            else:
                # Cycle or single activity — use birth activity
                type_rows = self.birth_death.df[
                    self.birth_death.df["object_type"] == otype
                ]
                if not type_rows.empty:
                    self._start_activities[otype] = (
                        type_rows["birth_activity"].mode().iloc[0]
                    )

    def _precompute_binding_thresholds(self) -> None:
        """Compute how many target objects trigger a binding event."""
        for (src_type, tgt_type), policy in self.patterns.binding_policies.items():
            if policy.capacity_dist is not None:
                threshold = max(1, int(policy.capacity_dist.rvs(size=1, rng=self.rng)[0]))
            else:
                # Estimate mean group size: total relations / estimated sources
                # Use sqrt as a heuristic when we don't have exact source count
                threshold = max(2, round(policy.n_instances ** 0.5))
                # Cap at reasonable size
                threshold = min(threshold, 20)
            self._binding_thresholds[(src_type, tgt_type)] = threshold

    # ------------------------------------------------------------------
    # ID generation and registration
    # ------------------------------------------------------------------

    def generate_id(self, object_type: str) -> str:
        self._id_counters[object_type] += 1
        return f"sim_{object_type}_{self._id_counters[object_type]}"

    def next_event_id(self) -> str:
        self._event_counter += 1
        return f"sim_e_{self._event_counter}"

    def register(self, agent: Agent) -> None:
        self._agents[agent.object_id] = agent
        self._objects.append(
            SimulatedObject(object_id=agent.object_id, object_type=agent.object_type)
        )

    def record_event(self, event: SimulatedEvent) -> None:
        self._events.append(event)

    # ------------------------------------------------------------------
    # Getters for discovery results
    # ------------------------------------------------------------------

    def get_start_activity(self, object_type: str) -> str:
        return self._start_activities[object_type]

    def get_branching(self, object_type: str) -> BranchingModel | None:
        return self.behavioral.branching_models.get(object_type)

    def sample_duration(self, object_type: str, activity: str) -> float:
        model = self.behavioral.duration_models.get((object_type, activity))
        if model is None:
            return 0.0
        return float(model.distribution.rvs(size=1, rng=self.rng)[0])

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def handle_spawning(self, parent: Agent) -> None:
        """Create all derived children for a new parent. Children are created
        but not started — they wait in _pending_children until the parent
        reaches their birth activity."""
        from simoc.simulation.agent import DerivedAgent

        parent_type = parent.object_type

        for (pt, ct), profile in self.spawning_profiles.items():
            if pt != parent_type:
                continue

            count = int(profile.fitted.rvs(size=1, rng=self.rng)[0])
            count = max(0, count)

            birth_act = self._start_activities.get(ct, "")

            for _ in range(count):
                child_id = self.generate_id(ct)
                child = DerivedAgent(
                    object_id=child_id,
                    object_type=ct,
                    env=self.env,
                    mediator=self,
                    rng=self.rng,
                    parent=parent,
                )
                self.register(child)
                self._parent_children[parent.object_id].append(child_id)
                self._child_parent[child_id] = parent.object_id

                # Record O2O relation
                self._o2o_relations.append(
                    SimulatedO2O(
                        source_id=parent.object_id,
                        source_type=parent_type,
                        target_id=child_id,
                        target_type=ct,
                        qualifier="spawns",
                    )
                )

                # Queue child for start when parent reaches birth activity
                self._pending_children[parent.object_id][birth_act].append(child)

            logger.debug(
                "Parent %s spawned %d children of type %s",
                parent.object_id,
                count,
                ct,
            )

    def handle_activity_spawning(
        self, parent: Agent, activity: str
    ) -> list[tuple[str, str]]:
        """Start one pending child at this activity. Returns co_objects."""
        pending = self._pending_children.get(parent.object_id, {}).get(activity, [])
        if not pending:
            return []

        # Start one child
        child = pending.pop(0)
        self.env.process(child.lifecycle())
        return [(child.object_id, child.object_type)]

    def override_branching(self, agent: Agent, current_activity: str) -> str | None:
        """Force self-loop if there are still pending children at this activity."""
        pending = self._pending_children.get(agent.object_id, {}).get(
            current_activity, []
        )
        if pending:
            return current_activity  # force self-loop
        return None

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def is_sync_point(self, agent: Agent, activity: str) -> bool:
        """Is this activity a sync point where this agent (as parent) must wait?"""
        for (act, synced_type), rule in self.patterns.synchronization_rules.items():
            if act == activity and rule.parent_type == agent.object_type:
                # Check if this agent actually has children of the synced type
                children = self._parent_children.get(agent.object_id, [])
                has_synced = any(
                    self._agents[cid].object_type == synced_type
                    for cid in children
                    if cid in self._agents
                )
                if has_synced:
                    return True
        return False

    def is_child_sync_point(self, agent: Agent, activity: str) -> bool:
        """Is this activity a sync point where this agent is the synced child?"""
        key = (activity, agent.object_type)
        return key in self.patterns.synchronization_rules

    def wait_for_sync(
        self, agent: Agent, activity: str
    ) -> Generator[simpy.Event, None, list[tuple[str, str]]]:
        """Parent yields here until all children of synced_type are ready."""
        for (act, synced_type), rule in self.patterns.synchronization_rules.items():
            if act != activity or rule.parent_type != agent.object_type:
                continue

            children = self._parent_children.get(agent.object_id, [])
            synced_children = [
                cid
                for cid in children
                if cid in self._agents
                and self._agents[cid].object_type == synced_type
            ]

            if not synced_children:
                return []

            gate_key = (agent.object_id, activity)
            self._sync_events[gate_key] = self.env.event()
            self._sync_target_count[gate_key] = len(synced_children)

            # Check if children already reported ready
            already_ready = self._sync_ready.get(gate_key, set())
            if self._check_sync_condition(rule.condition, len(already_ready), len(synced_children)):
                self._sync_events[gate_key].succeed()

            # Wait with deadlock timeout
            yield self._sync_events[gate_key] | self.env.timeout(
                self.config.max_sync_wait
            )

            sync_succeeded = self._sync_events[gate_key].triggered

            if not sync_succeeded:
                logger.debug(
                    "Sync timeout for %s at %s after %.0fs",
                    agent.object_id,
                    activity,
                    self.config.max_sync_wait,
                )

            # Add sync delay only if sync actually succeeded
            if sync_succeeded:
                delay = float(rule.sync_delay_dist.rvs(size=1, rng=self.rng)[0])
                if delay > 0:
                    yield self.env.timeout(delay)

            # Return only children that actually reported ready
            ready = self._sync_ready.get(gate_key, set())
            return [(cid, synced_type) for cid in synced_children if cid in ready]

        return []

    def signal_child_ready(self, child: Agent, sync_activity: str) -> None:
        """Called when a child reaches the sync activity (signals readiness)."""
        parent_id = self._child_parent.get(child.object_id)
        if parent_id is None:
            return

        gate_key = (parent_id, sync_activity)
        self._sync_ready[gate_key].add(child.object_id)

        # If gate already created, check if condition now met
        if gate_key in self._sync_events and not self._sync_events[gate_key].triggered:
            target = self._sync_target_count.get(gate_key, 0)
            ready_count = len(self._sync_ready[gate_key])

            rule_key = (sync_activity, child.object_type)
            rule = self.patterns.synchronization_rules.get(rule_key)
            if rule and self._check_sync_condition(rule.condition, ready_count, target):
                self._sync_events[gate_key].succeed()

    @staticmethod
    def _check_sync_condition(condition: str, ready: int, total: int) -> bool:
        if total == 0:
            return True
        if condition == "ALL":
            return ready >= total
        if condition.startswith("THRESHOLD"):
            k = float(condition.split("(")[1].rstrip(")"))
            return ready / total >= k
        return ready >= total

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def is_batch_point(self, agent: Agent, activity: str) -> bool:
        return (activity, agent.object_type) in self.patterns.batching_rules

    def wait_for_batch(
        self, agent: Agent, activity: str
    ) -> Generator[simpy.Event, None, list[tuple[str, str]]]:
        """Queue agent for batching. Yields until batch fires."""
        key = (activity, agent.object_type)
        rule = self.patterns.batching_rules[key]

        # Each agent gets its own wake-up event
        my_event = self.env.event()
        my_event.co_objects = []  # type: ignore[attr-defined]
        self._batch_queues[key].append((agent, my_event))

        # Check trigger
        self._check_batch_trigger(key, rule)

        # Wait for batch or timeout
        yield my_event | self.env.timeout(self.config.batch_timeout)

        return getattr(my_event, "co_objects", [])

    def _check_batch_trigger(self, key: tuple, rule) -> None:
        queue = self._batch_queues[key]
        if rule.trigger_type == "threshold":
            threshold = int(rule.trigger_params.get("threshold", 2))
        elif rule.trigger_type == "unknown":
            # Use batch size distribution as ad-hoc threshold
            threshold = max(1, int(rule.batch_size_dist.rvs(size=1, rng=self.rng)[0]))
        else:
            threshold = max(1, int(rule.batch_size_dist.rvs(size=1, rng=self.rng)[0]))

        if len(queue) >= threshold:
            self._fire_batch(key)

    def _fire_batch(self, key: tuple) -> None:
        queue = self._batch_queues[key]
        all_agents = [a for a, _ in queue]
        for agent, event in queue:
            if not event.triggered:
                co = [
                    (a.object_id, a.object_type)
                    for a in all_agents
                    if a is not agent
                ]
                event.co_objects = co  # type: ignore[attr-defined]
                event.succeed()
        self._batch_queues[key] = []

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def notify_state_change(self, agent: Agent, completed_activity: str) -> None:
        """Called after an agent completes an activity."""
        # Check if this completion makes a child ready for sync
        branching = self.get_branching(agent.object_type)
        if branching and completed_activity in branching.probabilities:
            for next_act in branching.probabilities[completed_activity]:
                sync_key = (next_act, agent.object_type)
                if sync_key in self.patterns.synchronization_rules:
                    self.signal_child_ready(agent, next_act)

        # Check if this completion makes the agent eligible for binding
        self._check_binding_readiness(agent, completed_activity)

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def _check_binding_readiness(self, agent: Agent, completed_activity: str) -> None:
        """Check if an agent is ready to be bound into a new object."""
        for (src_type, tgt_type), policy in self.patterns.binding_policies.items():
            if agent.object_type != tgt_type:
                continue

            # The binding activity is where the new object is created.
            # The target agent is ready when it reaches the activity
            # immediately BEFORE the binding activity in its DFG.
            binding_act = policy.binding_activity
            dfg = self.behavioral.type_dfgs.get(tgt_type)
            if dfg is None:
                continue

            # Check if the completed activity leads to the binding activity
            branching = self.get_branching(tgt_type)
            if branching and completed_activity in branching.probabilities:
                if binding_act in branching.probabilities[completed_activity]:
                    # This agent just completed the pre-binding activity
                    key = (src_type, tgt_type)
                    self._binding_queues[key].append(agent)

                    # Check if enough targets have accumulated
                    threshold = self._binding_thresholds.get(key, 2)
                    if len(self._binding_queues[key]) >= threshold:
                        self._execute_binding(key, policy)

    def _execute_binding(self, key: tuple[str, str], policy) -> None:
        """Create a new source-type object and bind target objects to it."""
        from simoc.simulation.agent import RootAgent

        src_type, tgt_type = key
        queue = self._binding_queues[key]
        threshold = self._binding_thresholds.get(key, 2)

        # Select targets: affinity-based (SimOC) or FIFO (random baseline)
        n_bind = min(len(queue), threshold)

        if policy.hard_constraints.get("mode") == "random":
            # Random baseline: shuffle queue, then take first N (no affinity)
            shuffled = list(queue)
            self.rng.shuffle(shuffled)
            bound_targets = shuffled[:n_bind]
            self._binding_queues[key] = shuffled[n_bind:]
        else:
            # SimOC: parent-affinity greedy selection
            # Pick seed, then prefer items sharing a parent with the group
            seed = queue.pop(0)
            selected = [seed]
            remaining = list(queue)
            while len(selected) < n_bind and remaining:
                parents_in_group = {
                    self._child_parent.get(s.object_id) for s in selected
                } - {None}
                # Find first candidate sharing a parent; fallback to first
                best_idx = 0
                if parents_in_group:
                    for i, c in enumerate(remaining):
                        if self._child_parent.get(c.object_id) in parents_in_group:
                            best_idx = i
                            break
                selected.append(remaining.pop(best_idx))
            bound_targets = selected
            self._binding_queues[key] = remaining

        # Create the new source object (e.g., package)
        new_id = self.generate_id(src_type)
        new_agent = RootAgent(
            object_id=new_id,
            object_type=src_type,
            env=self.env,
            mediator=self,
            rng=self.rng,
        )
        self.register(new_agent)

        # Register O2O relations (source -> each target)
        for target in bound_targets:
            self._o2o_relations.append(
                SimulatedO2O(
                    source_id=new_id,
                    source_type=src_type,
                    target_id=target.object_id,
                    target_type=tgt_type,
                    qualifier="contains",
                )
            )

        # Log the binding event (e.g., "create package")
        co_objects = [(t.object_id, t.object_type) for t in bound_targets]
        event = SimulatedEvent(
            event_id=self.next_event_id(),
            activity=policy.binding_activity,
            timestamp=self.env.now,
            objects=[(new_id, src_type)] + co_objects,
        )
        self.record_event(event)

        # Start the new object's lifecycle (if it has a DFG)
        if src_type in self.behavioral.type_dfgs:
            self.env.process(new_agent.lifecycle())

        logger.debug(
            "Binding: created %s with %d %s targets at t=%.0f",
            new_id, n_bind, tgt_type, self.env.now,
        )

    # ------------------------------------------------------------------
    # Output collection
    # ------------------------------------------------------------------

    def get_events(self) -> list[SimulatedEvent]:
        return list(self._events)

    def get_objects(self) -> list[SimulatedObject]:
        return list(self._objects)

    def get_o2o(self) -> list[SimulatedO2O]:
        return list(self._o2o_relations)
