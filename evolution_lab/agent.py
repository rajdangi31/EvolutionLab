"""Agent definitions and behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import SimulationConfig
    from .neural_network import NeuralNetwork
except ImportError:  # Allows running modules from inside evolution_lab/.
    from config import SimulationConfig
    from neural_network import NeuralNetwork


ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_COUNT = 5
BRAIN_INPUT_COUNT = 13
BRAIN_OUTPUT_COUNT = 6

ACTION_DELTAS: dict[int, tuple[int, int]] = {
    ACTION_STAY: (0, 0),
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}


@dataclass(slots=True)
class Agent:
    """An evolving food-seeking agent."""

    x: int
    y: int
    energy: float
    brain: NeuralNetwork
    age: int = 0
    fitness: float = 0.0
    food_eaten: int = 0
    cooperative_events: int = 0
    previous_action: int = ACTION_STAY
    memory_dx: float = 0.0
    memory_dy: float = 0.0
    memory_strength: float = 0.0
    signal: float = 0.0
    local_signal: float = 0.0
    cultural_dx: float = 0.0
    cultural_dy: float = 0.0
    cultural_confidence: float = 0.0
    action_bias: np.ndarray | None = None
    alive: bool = True

    @classmethod
    def random(cls, config: SimulationConfig, rng: np.random.Generator) -> "Agent":
        """Create an agent at a random position with a random brain."""

        brain = NeuralNetwork.random(BRAIN_INPUT_COUNT, config.hidden_size, BRAIN_OUTPUT_COUNT, rng)
        return cls(
            x=int(rng.integers(0, config.width)),
            y=int(rng.integers(0, config.height)),
            energy=config.initial_energy,
            brain=brain,
        )

    def reset_lifetime(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        """Place the agent into a new generation without changing its brain."""

        self.x = int(rng.integers(0, config.width))
        self.y = int(rng.integers(0, config.height))
        self.energy = config.initial_energy
        self.age = 0
        self.fitness = 0.0
        self.food_eaten = 0
        self.cooperative_events = 0
        self.previous_action = ACTION_STAY
        self.memory_dx = 0.0
        self.memory_dy = 0.0
        self.memory_strength = 0.0
        self.signal = 0.0
        self.local_signal = 0.0
        self.cultural_dx = 0.0
        self.cultural_dy = 0.0
        self.cultural_confidence = 0.0
        self.action_bias = np.zeros(ACTION_COUNT, dtype=float)
        self.alive = True

    def observe(self, nearest_food: tuple[int, int], config: SimulationConfig) -> np.ndarray:
        """Build the normalized observation vector used by the neural policy."""

        food_x, food_y = nearest_food
        dx = food_x - self.x
        dy = food_y - self.y
        diagonal = float(np.hypot(config.width, config.height))
        distance = float(np.hypot(dx, dy))
        sees_food = distance <= config.sensory_range
        if sees_food:
            observed_dx = dx
            observed_dy = dy
            if config.internal_memory_enabled:
                self.memory_dx = dx / max(1, config.width)
                self.memory_dy = dy / max(1, config.height)
                self.memory_strength = 1.0
        else:
            observed_dx = self.memory_dx * config.width if config.internal_memory_enabled else 0.0
            observed_dy = self.memory_dy * config.height if config.internal_memory_enabled else 0.0
            distance = diagonal
            if config.internal_memory_enabled:
                self.memory_strength *= config.memory_decay
            else:
                self.memory_strength = 0.0
                self.memory_dx = 0.0
                self.memory_dy = 0.0
        previous_action_encoded = (self.previous_action / (ACTION_COUNT - 1)) * 2.0 - 1.0
        return np.array(
            [
                observed_dx / max(1, config.width),
                observed_dy / max(1, config.height),
                distance / max(1.0, diagonal),
                self.energy / max(1.0, config.max_energy),
                previous_action_encoded,
                1.0 if sees_food else -1.0,
                self.memory_dx,
                self.memory_dy,
                self.memory_strength,
                self.local_signal if config.communication_enabled else 0.0,
                self.cultural_dx if config.cultural_memory_enabled else 0.0,
                self.cultural_dy if config.cultural_memory_enabled else 0.0,
                self.cultural_confidence if config.cultural_memory_enabled else 0.0,
            ],
            dtype=float,
        )

    def choose_action(
        self,
        nearest_food: tuple[int, int],
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> int:
        """Sample the next action and communication signal from the brain."""

        action_bias = self.action_bias if config.lifetime_learning_enabled else None
        action, signal = self.brain.policy(
            self.observe(nearest_food, config),
            ACTION_COUNT,
            rng,
            action_bias=action_bias,
        )
        self.signal = signal if config.communication_enabled else 0.0
        return action

    def apply_action(self, action: int, config: SimulationConfig) -> None:
        """Move the agent and pay the action energy cost."""

        if not self.alive:
            return

        dx, dy = ACTION_DELTAS[action]
        self.x = min(config.width - 1, max(0, self.x + dx))
        self.y = min(config.height - 1, max(0, self.y + dy))
        self.energy -= config.stay_energy_cost if action == ACTION_STAY else config.move_energy_cost
        if config.communication_enabled:
            self.energy -= abs(self.signal) * config.communication_cost
        self.age += 1
        self.previous_action = action
        if self.energy <= 0.0:
            self.energy = 0.0
            self.alive = False

    def consume_food(self, config: SimulationConfig) -> None:
        """Reward the agent for eating one food item."""

        self.food_eaten += 1
        self.energy = min(config.max_energy, self.energy + config.food_energy)
        if config.lifetime_learning_enabled and self.action_bias is not None:
            self.action_bias *= config.action_bias_decay
            self.action_bias[self.previous_action] += config.lifetime_learning_rate

    def decay_lifetime_learning(self, config: SimulationConfig) -> None:
        """Fade simple action preference memory without gradients."""

        if config.lifetime_learning_enabled and self.action_bias is not None:
            self.action_bias *= config.action_bias_decay

    def finalize_fitness(self, config: SimulationConfig) -> None:
        """Compute the generation fitness from food, survival, and energy."""

        self.fitness = (
            self.food_eaten * config.food_fitness
            + self.cooperative_events * config.cooperation_fitness
            + self.age * config.survival_fitness
            + self.energy * config.remaining_energy_fitness
        )
