"""Simulation runner: top-level orchestrator for the simulation engine."""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np
import simpy

from simoc.discovery.data_structures import (
    BehavioralProfile,
    DiscoveryResult,
    InteractionPatterns,
)
from simoc.simulation.agent import RootAgent
from simoc.simulation.data_structures import SimulationConfig, SimulationResult
from simoc.simulation.mediator import InteractionMediator

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Run agent-based simulations from discovered parameters."""

    def __init__(
        self,
        discovery_result: DiscoveryResult,
        behavioral_profile: BehavioralProfile,
        interaction_patterns: InteractionPatterns,
        real_data: "OCELData | None" = None,
    ):
        self.discovery = discovery_result
        self.behavioral = behavioral_profile
        self.patterns = interaction_patterns

        # Learn per-activity type composition from real log
        self._activity_type_composition: dict[str, dict[str, float]] = {}
        self._master_data_counts: dict[str, int] = {}
        if real_data is not None:
            self._learn_event_composition(real_data)

    def run(
        self,
        duration: float,
        seed: int = 42,
        max_sync_wait: float | None = None,
        batch_timeout: float = 7200.0,
    ) -> SimulationResult:
        """Run a simulation for the given duration.

        Parameters
        ----------
        duration : float
            Simulation end time in seconds.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        SimulationResult
            All simulated events, objects, and O2O relations.
        """
        # Default sync timeout: 10% of simulation duration (generous but not infinite)
        if max_sync_wait is None:
            max_sync_wait = max(3600.0, duration * 0.1)

        config = SimulationConfig(
            duration=duration,
            seed=seed,
            max_sync_wait=max_sync_wait,
            batch_timeout=batch_timeout,
        )
        rng = np.random.default_rng(seed)
        env = simpy.Environment()

        mediator = InteractionMediator(
            env=env,
            type_classification=self.discovery.type_classification,
            spawning_profiles=self.discovery.spawning_profiles,
            behavioral_profile=self.behavioral,
            interaction_patterns=self.patterns,
            birth_death=self.discovery.birth_death,
            rng=rng,
            config=config,
            activity_type_composition=self._activity_type_composition,
        )

        # Create master data pools (employees, products, customers — long-lived entities)
        for md_type, md_count in self._master_data_counts.items():
            for _ in range(md_count):
                md_id = mediator.generate_id(md_type)
                from simoc.simulation.agent import Agent
                md_agent = Agent(md_id, md_type, env, mediator, rng)
                md_agent.state = "ACTIVE"
                mediator.register(md_agent)
                mediator.add_to_master_data_pool(md_type, md_id)

        # Start root type generators (excluding master data — they don't have lifecycles)
        for root_type in self._get_root_types():
            if root_type in self._master_data_counts:
                continue  # master data already created above
            env.process(self._root_generator(root_type, env, mediator, rng, config))

        # Run simulation
        env.run(until=duration)

        # Collect results
        events = sorted(mediator.get_events(), key=lambda e: e.timestamp)

        # Summary diagnostics
        from collections import Counter
        type_counts = Counter(o.object_type for o in mediator.get_objects())
        act_counts = Counter(str(e.activity) for e in events)
        logger.info(
            "Simulation complete: %d events, %d objects, %d O2O. "
            "Types: %s. Top activities: %s",
            len(events),
            len(mediator.get_objects()),
            len(mediator.get_o2o()),
            dict(type_counts),
            dict(act_counts.most_common(5)),
        )

        return SimulationResult(
            events=events,
            objects=mediator.get_objects(),
            o2o_relations=mediator.get_o2o(),
            config=config,
        )

    def _get_root_types(self) -> list[str]:
        # Root types that have arrival models AND are not genuinely created by binding.
        # A type is "binding-created" only if its start activity IS the binding activity
        # (e.g., packages start at "create package" = binding activity → excluded).
        # Master data types like customers may appear as binding sources but are NOT
        # created by binding (e.g., customers start at "place order" ≠ "confirm order").
        binding_born_types: set[str] = set()
        for (src_type, _), policy in self.patterns.binding_policies.items():
            start_act = self._start_activity_for_type(src_type)
            if start_act and start_act == policy.binding_activity:
                binding_born_types.add(src_type)

        return [
            t
            for t, role in self.discovery.type_classification.classification.items()
            if role == "root"
            and t in self.behavioral.arrival_models
            and t not in binding_born_types
        ]

    def _learn_event_composition(self, data) -> None:
        """Learn which types participate in each activity (and at what count)."""
        from collections import defaultdict, Counter

        act_type_counts: dict[str, list[Counter]] = defaultdict(list)
        for _, row in data.events.iterrows():
            act = str(row["activity"])
            type_counter = Counter(ot for _, ot in row["related_objects"])
            act_type_counts[act].append(type_counter)

        for act, counters in act_type_counts.items():
            all_types = set()
            for c in counters:
                all_types.update(c.keys())
            self._activity_type_composition[act] = {
                t: np.mean([c.get(t, 0) for c in counters])
                for t in all_types
            }

        # Count master data instances (types with few instances, long lifecycles)
        type_counts = data.objects["object_type"].value_counts().to_dict()
        oid_to_type = data.objects.set_index("object_id")["object_type"].to_dict()
        for otype, count in type_counts.items():
            avg_lc = np.mean([
                len(data.lifecycles.get(oid, []))
                for oid in data.objects.loc[
                    data.objects["object_type"] == otype, "object_id"
                ]
            ])
            # Master data: few instances, long lifecycles
            if count <= 50 and avg_lc > 20:
                self._master_data_counts[otype] = count

    def _start_activity_for_type(self, otype: str) -> str | None:
        """Get the start activity for a type from its DFG."""
        dfg = self.behavioral.type_dfgs.get(otype)
        if dfg is None:
            return None
        targets = {tgt for (_, tgt) in dfg.edges.keys()}
        sources = {src for (src, _) in dfg.edges.keys()}
        starts = sources - targets
        return sorted(starts)[0] if starts else None

    def _root_generator(
        self,
        root_type: str,
        env: simpy.Environment,
        mediator: InteractionMediator,
        rng: np.random.Generator,
        config: SimulationConfig,
    ) -> Generator:
        """Generate root type agents according to the arrival model."""
        arrival_model = self.behavioral.arrival_models.get(root_type)
        if arrival_model is None:
            logger.warning("No arrival model for root type '%s'. Skipping.", root_type)
            return

        while True:
            # Sample inter-arrival gap
            gap = float(arrival_model.distribution.rvs(size=1, rng=rng)[0])
            gap = max(0.0, gap)
            yield env.timeout(gap)

            if env.now >= config.duration:
                break

            # Create new root agent
            agent = RootAgent(
                object_id=mediator.generate_id(root_type),
                object_type=root_type,
                env=env,
                mediator=mediator,
                rng=rng,
            )
            mediator.register(agent)

            # Spawn children (creates but doesn't start them)
            mediator.handle_spawning(agent)

            # Start agent lifecycle
            env.process(agent.lifecycle())
