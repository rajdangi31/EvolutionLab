"""Single-agent tabular Q-learning baseline for comparison with evolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .agent import ACTION_COUNT, ACTION_DELTAS, ACTION_STAY
    from .config import QLearningConfig, SimulationConfig
    from .world import World
except ImportError:
    from agent import ACTION_COUNT, ACTION_DELTAS, ACTION_STAY
    from config import QLearningConfig, SimulationConfig
    from world import World


State = tuple[int, int, int, int, int]


@dataclass(slots=True)
class QLearningEpisodeMetrics:
    """Metrics for one Q-learning training episode."""

    episode: int
    fitness: float
    rolling_average_fitness: float
    food_consumed: int
    survival_steps: int
    epsilon: float
    q_states: int


class QLearningTrainer:
    """Train one agent with tabular Q-learning in the same food world."""

    def __init__(self, sim_config: SimulationConfig, rl_config: QLearningConfig) -> None:
        self.sim_config = sim_config
        self.rl_config = rl_config
        self.rng = np.random.default_rng(sim_config.seed)
        self.world = World(sim_config, self.rng)
        self.q_table: dict[State, np.ndarray] = {}
        self.history: list[QLearningEpisodeMetrics] = []

    def train(self) -> list[QLearningEpisodeMetrics]:
        """Train for the configured number of episodes."""

        for episode in range(self.rl_config.episodes):
            epsilon = self.epsilon_for_episode(episode)
            fitness, food_consumed, survival_steps = self.run_episode(episode, epsilon)
            recent = [row.fitness for row in self.history[-49:]] + [fitness]
            metrics = QLearningEpisodeMetrics(
                episode=episode,
                fitness=fitness,
                rolling_average_fitness=float(np.mean(recent)),
                food_consumed=food_consumed,
                survival_steps=survival_steps,
                epsilon=epsilon,
                q_states=len(self.q_table),
            )
            self.history.append(metrics)
            if episode == 0 or (episode + 1) % 10 == 0:
                print(
                    f"episode={episode + 1}/{self.rl_config.episodes} "
                    f"fitness={fitness:.2f} "
                    f"rolling_avg={metrics.rolling_average_fitness:.2f} "
                    f"food={food_consumed} "
                    f"epsilon={epsilon:.3f} "
                    f"q_states={len(self.q_table)}"
                )

        self.save_csv(self.rl_config.metrics_path)
        self.plot(self.rl_config.plots_dir)
        return self.history

    def run_episode(self, episode: int, epsilon: float) -> tuple[float, int, int]:
        """Run one training episode and update the Q-table."""

        config = self.sim_config
        self.world.reset()
        x = int(self.rng.integers(0, config.width))
        y = int(self.rng.integers(0, config.height))
        energy = config.initial_energy
        previous_action = ACTION_STAY
        food_consumed = 0
        survival_steps = 0
        state = self.state_for(x, y, energy, previous_action)

        for _ in range(config.generation_steps):
            action = self.choose_action(state, epsilon)
            dx, dy = ACTION_DELTAS[action]
            x = min(config.width - 1, max(0, x + dx))
            y = min(config.height - 1, max(0, y + dy))

            cost = config.stay_energy_cost if action == ACTION_STAY else config.move_energy_cost
            energy -= cost
            survival_steps += 1
            reward = config.survival_fitness - cost

            if self.world.consume_at(x, y):
                food_consumed += 1
                energy = min(config.max_energy, energy + config.food_energy)
                reward += config.food_fitness

            done = energy <= 0.0
            if done:
                energy = 0.0
                reward -= config.food_fitness * 0.25

            next_state = self.state_for(x, y, energy, action)
            self.update_q_value(state, action, reward, next_state, done)
            state = next_state
            previous_action = action

            if done:
                break

        fitness = (
            food_consumed * config.food_fitness
            + survival_steps * config.survival_fitness
            + energy * config.remaining_energy_fitness
        )
        return fitness, food_consumed, survival_steps

    def state_for(self, x: int, y: int, energy: float, previous_action: int) -> State:
        """Discretize the environment observation into a Q-table key."""

        nearest_x, nearest_y = self.world.nearest_food(x, y)
        dx = self.bucket_delta(nearest_x - x)
        dy = self.bucket_delta(nearest_y - y)
        distance = float(np.hypot(nearest_x - x, nearest_y - y))
        distance_bin = int(np.digitize(distance, self.rl_config.distance_bins))
        normalized_energy = energy / max(1.0, self.sim_config.max_energy)
        energy_bin = int(np.digitize(normalized_energy, self.rl_config.energy_bins))
        return dx, dy, distance_bin, energy_bin, previous_action

    def bucket_delta(self, delta: int) -> int:
        """Keep nearby direction exact and compress far-away food offsets."""

        clip = self.rl_config.delta_clip
        return int(min(clip + 1, max(-clip - 1, delta)))

    def choose_action(self, state: State, epsilon: float) -> int:
        """Use epsilon-greedy exploration."""

        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, ACTION_COUNT))
        return int(np.argmax(self.q_values(state)))

    def update_q_value(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """Apply the standard Q-learning Bellman update."""

        q_values = self.q_values(state)
        bootstrap = 0.0 if done else float(np.max(self.q_values(next_state)))
        target = reward + self.rl_config.discount_factor * bootstrap
        q_values[action] += self.rl_config.learning_rate * (target - q_values[action])

    def q_values(self, state: State) -> np.ndarray:
        """Return Q-values for a state, creating a row when first visited."""

        if state not in self.q_table:
            self.q_table[state] = np.zeros(ACTION_COUNT, dtype=float)
        return self.q_table[state]

    def epsilon_for_episode(self, episode: int) -> float:
        """Linearly decay exploration over the first part of training."""

        decay_episodes = max(1, int(self.rl_config.episodes * self.rl_config.epsilon_decay_fraction))
        progress = min(1.0, episode / decay_episodes)
        return self.rl_config.initial_epsilon + progress * (
            self.rl_config.final_epsilon - self.rl_config.initial_epsilon
        )

    def save_csv(self, path: Path) -> None:
        """Save Q-learning metrics to CSV."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "episode",
                    "fitness",
                    "rolling_average_fitness",
                    "food_consumed",
                    "survival_steps",
                    "epsilon",
                    "q_states",
                ]
            )
            for row in self.history:
                writer.writerow(
                    [
                        row.episode,
                        row.fitness,
                        row.rolling_average_fitness,
                        row.food_consumed,
                        row.survival_steps,
                        row.epsilon,
                        row.q_states,
                    ]
                )

    def plot(self, output_dir: Path) -> None:
        """Write Q-learning training plots."""

        if not self.history:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        episodes = [row.episode for row in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, [row.fitness for row in self.history], alpha=0.35, label="Episode fitness")
        plt.plot(episodes, [row.rolling_average_fitness for row in self.history], label="50-episode rolling average")
        plt.xlabel("Episode")
        plt.ylabel("Fitness")
        plt.title("Single-agent Q-learning fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "q_learning_fitness.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, [row.food_consumed for row in self.history], label="Food consumed")
        plt.xlabel("Episode")
        plt.ylabel("Food consumed")
        plt.title("Single-agent Q-learning food collection")
        plt.tight_layout()
        plt.savefig(output_dir / "q_learning_food.png", dpi=150)
        plt.close()
