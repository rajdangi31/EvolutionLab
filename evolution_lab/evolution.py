"""Evolutionary selection and mutation."""

from __future__ import annotations

import numpy as np

try:
    from .agent import Agent
    from .config import SimulationConfig
except ImportError:
    from agent import Agent
    from config import SimulationConfig


class EvolutionEngine:
    """Rank agents, preserve elites, and create mutated offspring."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng

    def next_generation(self, population: list[Agent]) -> list[Agent]:
        """Return a new population using selection and mutation only."""

        ranked = sorted(population, key=lambda agent: agent.fitness, reverse=True)
        survivor_count = max(1, int(len(ranked) * self.config.survivor_fraction))
        survivors = ranked[:survivor_count]

        next_population: list[Agent] = []

        # Elites keep their exact brains so successful behaviors are not lost.
        for survivor in survivors:
            brain = survivor.brain.clone()
            next_population.append(Agent(0, 0, self.config.initial_energy, brain))

        # The rest of the generation is filled with mutated clones of elites.
        while len(next_population) < self.config.population_size:
            parent = self.rng.choice(survivors)
            brain = parent.brain.clone()
            brain.mutate(self.rng, self.config.mutation_rate, self.config.mutation_strength)
            next_population.append(Agent(0, 0, self.config.initial_energy, brain))

        for agent in next_population:
            agent.reset_lifetime(self.config, self.rng)

        return next_population

