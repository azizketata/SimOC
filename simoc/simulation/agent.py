"""Agent base class and type-specific agent implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generator

import numpy as np
import simpy

from simoc.simulation.data_structures import SimulatedEvent

if TYPE_CHECKING:
    from simoc.simulation.mediator import InteractionMediator

logger = logging.getLogger(__name__)


class Agent:
    """Base class for all simulation agents."""

    def __init__(
        self,
        object_id: str,
        object_type: str,
        env: simpy.Environment,
        mediator: InteractionMediator,
        rng: np.random.Generator,
    ):
        self.object_id = object_id
        self.object_type = object_type
        self.env = env
        self.mediator = mediator
        self.rng = rng
        self.state: str = "CREATED"
        self.completed_activities: list[str] = []

    def log_event(self, activity: str, co_objects: list[tuple[str, str]]) -> None:
        """Record an event at the current simulation time.

        Automatically includes co-objects based on the real log's
        per-activity type composition template.
        """
        template_co = self.mediator.get_event_co_objects(self, activity)
        combined = [(self.object_id, self.object_type)] + co_objects + template_co

        # Deduplicate while preserving order
        seen: set[tuple[str, str]] = set()
        unique = []
        for pair in combined:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)

        event = SimulatedEvent(
            event_id=self.mediator.next_event_id(),
            activity=activity,
            timestamp=self.env.now,
            objects=unique,
        )
        self.mediator.record_event(event)

    def _choose_next(self, current: str) -> str | None:
        """Choose the next activity using branching probabilities."""
        # Check if mediator overrides branching (for spawning loops)
        override = self.mediator.override_branching(self, current)
        if override is not None:
            return override

        branching = self.mediator.get_branching(self.object_type)
        if branching is None or current not in branching.probabilities:
            return None  # terminal activity

        targets = branching.probabilities[current]
        activities = list(targets.keys())
        probs = [targets[a] for a in activities]
        return self.rng.choice(activities, p=probs)

    def lifecycle(self) -> Generator:
        """Abstract lifecycle process. Must be overridden."""
        raise NotImplementedError


class RootAgent(Agent):
    """Agent for root object types (independent arrivals)."""

    def lifecycle(self) -> Generator:
        current = self.mediator.get_start_activity(self.object_type)

        while current is not None:
            co_objects: list[tuple[str, str]] = []

            # 1. Check sync (parent waits for children)
            if self.mediator.is_sync_point(self, current):
                sync_co = yield from self.mediator.wait_for_sync(self, current)
                co_objects.extend(sync_co)

            # 2. Check batch
            if self.mediator.is_batch_point(self, current):
                batch_co = yield from self.mediator.wait_for_batch(self, current)
                co_objects.extend(batch_co)

            # 3. Handle spawning at this activity (start pending children)
            spawn_co = self.mediator.handle_activity_spawning(self, current)
            co_objects.extend(spawn_co)

            # 4. Log event
            self.log_event(current, co_objects)
            self.state = current
            self.completed_activities.append(current)

            # 5. Notify mediator of state change
            self.mediator.notify_state_change(self, current)

            # 6. Choose next activity
            next_act = self._choose_next(current)

            # 7. Wait duration (if not terminal)
            if next_act is not None:
                dur = max(0.0, self.mediator.sample_duration(self.object_type, current))
                yield self.env.timeout(dur)

            current = next_act

        self.state = "COMPLETED"


class DerivedAgent(Agent):
    """Agent for derived object types (spawned by a parent)."""

    def __init__(
        self,
        object_id: str,
        object_type: str,
        env: simpy.Environment,
        mediator: InteractionMediator,
        rng: np.random.Generator,
        parent: Agent,
    ):
        super().__init__(object_id, object_type, env, mediator, rng)
        self.parent = parent

    def lifecycle(self) -> Generator:
        start_act = self.mediator.get_start_activity(self.object_type)

        # Birth activity was already logged by parent → skip logging
        # but record state
        self.state = start_act
        self.completed_activities.append(start_act)

        # Notify mediator (for sync readiness tracking)
        self.mediator.notify_state_change(self, start_act)

        # Determine next activity after birth
        current = self._choose_next(start_act)

        # Wait for birth activity duration before moving on
        if current is not None:
            dur = max(0.0, self.mediator.sample_duration(self.object_type, start_act))
            yield self.env.timeout(dur)

        while current is not None:
            # Check if this is a sync point where I am the synced type
            # → signal readiness and terminate (parent will log the event)
            if self.mediator.is_child_sync_point(self, current):
                self.mediator.signal_child_ready(self, current)
                self.state = "COMPLETED"
                return

            co_objects: list[tuple[str, str]] = []

            # Handle spawning at this activity
            spawn_co = self.mediator.handle_activity_spawning(self, current)
            co_objects.extend(spawn_co)

            # Log event
            self.log_event(current, co_objects)
            self.state = current
            self.completed_activities.append(current)

            # Notify mediator
            self.mediator.notify_state_change(self, current)

            # Choose next
            next_act = self._choose_next(current)

            # Wait duration
            if next_act is not None:
                dur = max(0.0, self.mediator.sample_duration(self.object_type, current))
                yield self.env.timeout(dur)

            current = next_act

        self.state = "COMPLETED"
