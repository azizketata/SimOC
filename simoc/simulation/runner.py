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
    ):
        self.discovery = discovery_result
        self.behavioral = behavioral_profile
        self.patterns = interaction_patterns

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
        )

        # Start root type generators
        for root_type in self._get_root_types():
            env.process(self._root_generator(root_type, env, mediator, rng, config))

        # Run simulation
        env.run(until=duration)

        # Collect results
        events = sorted(mediator.get_events(), key=lambda e: e.timestamp)

        logger.info(
            "Simulation complete: %d events, %d objects, %d O2O relations.",
            len(events),
            len(mediator.get_objects()),
            len(mediator.get_o2o()),
        )

        return SimulationResult(
            events=events,
            objects=mediator.get_objects(),
            o2o_relations=mediator.get_o2o(),
            config=config,
        )

    def _get_root_types(self) -> list[str]:
        return [
            t
            for t, role in self.discovery.type_classification.classification.items()
            if role == "root"
        ]

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
