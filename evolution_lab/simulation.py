"""Simulation loop for EvolutionLab."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from .agent import Agent
    from .config import SimulationConfig
    from .cultural_memory import CulturalMemoryBank
    from .evolution import EvolutionEngine
    from .metrics import GenerationMetrics, MetricsTracker
    from .world import World
except ImportError:
    from agent import Agent
    from config import SimulationConfig
    from cultural_memory import CulturalMemoryBank
    from evolution import EvolutionEngine
    from metrics import GenerationMetrics, MetricsTracker
    from world import World


GenerationCallback = Callable[[int, GenerationMetrics, list[Agent], World], None]


class Simulation:
    """Coordinates world updates, agent evaluation, and evolution."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.world = World(config, self.rng)
        self.population = [Agent.random(config, self.rng) for _ in range(config.population_size)]
        self.evolution = EvolutionEngine(config, self.rng)
        self.metrics = MetricsTracker()
        self.cultural_memory = (
            CulturalMemoryBank(config.cultural_memory_size, config.cultural_memory_radius)
            if config.cultural_memory_enabled
            else None
        )
        self.generation = 0
        self.step_in_generation = 0

    def step(self) -> None:
        """Advance the current generation by one simulation step."""

        self.update_local_signals()
        eaters: list[Agent] = []
        for agent in self.population:
            if not agent.alive:
                continue
            if self.cultural_memory is not None:
                (
                    agent.cultural_dx,
                    agent.cultural_dy,
                    agent.cultural_confidence,
                ) = self.cultural_memory.read(agent.x, agent.y, self.config.width, self.config.height)
            nearest_food = self.world.nearest_food(agent.x, agent.y)
            action = agent.choose_action(nearest_food, self.config, self.rng)
            agent.apply_action(action, self.config)
            if agent.alive and self.world.consume_at(agent.x, agent.y):
                agent.consume_food(self.config)
                if self.cultural_memory is not None and agent.food_eaten >= self.config.cultural_write_threshold:
                    self.cultural_memory.write_food(
                        agent.x,
                        agent.y,
                        agent.memory_dx * self.config.width,
                        agent.memory_dy * self.config.height,
                        agent.signal,
                        self.config.food_fitness,
                        self.generation,
                    )
                eaters.append(agent)
            else:
                agent.decay_lifetime_learning(self.config)

        if eaters:
            self.apply_cooperation_credit(eaters)

        self.step_in_generation += 1

    def update_local_signals(self) -> None:
        """Let agents hear the average broadcast signal in their neighborhood."""

        radius = self.config.communication_radius
        buckets: dict[tuple[int, int], list[Agent]] = {}
        for agent in self.population:
            if agent.alive:
                buckets.setdefault((agent.x, agent.y), []).append(agent)

        for agent in self.population:
            if not agent.alive:
                agent.local_signal = 0.0
                continue
            total_signal = 0.0
            count = 0
            for x in range(agent.x - radius, agent.x + radius + 1):
                for y in range(agent.y - radius, agent.y + radius + 1):
                    for neighbor in buckets.get((x, y), []):
                        if neighbor is agent:
                            continue
                        if (neighbor.x - agent.x) ** 2 + (neighbor.y - agent.y) ** 2 <= radius * radius:
                            total_signal += neighbor.signal
                            count += 1
            agent.local_signal = total_signal / count if count and self.config.communication_enabled else 0.0

    def apply_cooperation_credit(self, eaters: list[Agent]) -> None:
        """Reward nearby living agents when group members find food."""

        if not self.config.cooperation_enabled:
            return
        radius2 = self.config.cooperation_radius * self.config.cooperation_radius
        for eater in eaters:
            for agent in self.population:
                if agent is eater or not agent.alive:
                    continue
                if (agent.x - eater.x) ** 2 + (agent.y - eater.y) ** 2 <= radius2:
                    agent.cooperative_events += 1

    def complete_generation(self) -> GenerationMetrics:
        """Finalize fitness, record metrics, and produce the next generation."""

        for agent in self.population:
            agent.finalize_fitness(self.config)

        extras = self.cultural_memory.snapshot_stats() if self.cultural_memory is not None else None
        fitnesses = np.array([agent.fitness for agent in self.population], dtype=float)
        food_consumed = int(sum(agent.food_eaten for agent in self.population))
        diversity = self.metrics.population_diversity(self.population)
        metrics = self.metrics.record_values(self.generation, fitnesses, food_consumed, diversity, extras)
        self.population = self.evolution.next_generation(self.population)
        if self.cultural_memory is not None:
            self.cultural_memory.mark_generation_transfer(len(self.population))
        self.world.reset()
        self.generation += 1
        self.step_in_generation = 0
        return metrics

    def run_generation(self) -> GenerationMetrics:
        """Run exactly one full generation."""

        while self.step_in_generation < self.config.generation_steps:
            self.step()
        return self.complete_generation()

    def run(
        self,
        generations: int,
        callback: GenerationCallback | None = None,
        autosave: bool = True,
    ) -> MetricsTracker:
        """Run several generations and optionally save metrics and plots."""

        for _ in range(generations):
            metrics = self.run_generation()
            if callback is not None:
                callback(self.generation - 1, metrics, self.population, self.world)

        if autosave:
            self.metrics.save_csv(self.config.metrics_path)
            self.metrics.plot(self.config.plots_dir)
        return self.metrics
