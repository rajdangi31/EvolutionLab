"""Grid world containing food."""

from __future__ import annotations

import numpy as np

try:
    from .config import SimulationConfig
except ImportError:
    from config import SimulationConfig


class World:
    """A bounded 2D world with respawning food."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.food: set[tuple[int, int]] = set()
        self.reset()

    def reset(self) -> None:
        """Randomly repopulate the world with food."""

        self.food.clear()
        max_cells = self.config.width * self.config.height
        target_count = min(self.config.food_count, max_cells)
        while len(self.food) < target_count:
            self.food.add(self.random_empty_position())

    def random_empty_position(self) -> tuple[int, int]:
        """Return a random cell not already occupied by food when possible."""

        if len(self.food) >= self.config.width * self.config.height:
            return (
                int(self.rng.integers(0, self.config.width)),
                int(self.rng.integers(0, self.config.height)),
            )
        while True:
            position = (
                int(self.rng.integers(0, self.config.width)),
                int(self.rng.integers(0, self.config.height)),
            )
            if position not in self.food:
                return position

    def nearest_food(self, x: int, y: int) -> tuple[int, int]:
        """Return the nearest food position by Euclidean distance."""

        if not self.food:
            return x, y
        food_positions = np.asarray(tuple(self.food), dtype=float)
        deltas = food_positions - np.array([x, y], dtype=float)
        index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        nearest = food_positions[index]
        return int(nearest[0]), int(nearest[1])

    def consume_at(self, x: int, y: int) -> bool:
        """Consume and respawn food at a position if food exists there."""

        position = (x, y)
        if position not in self.food:
            return False
        self.food.remove(position)
        self.food.add(self.random_empty_position())
        return True

