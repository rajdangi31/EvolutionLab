"""Batched headless simulation with optional CUDA acceleration through CuPy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("CUPY_CACHE_DIR", str(Path(".cupy-cache")))

try:
    from .config import SimulationConfig
    from .cultural_memory import CulturalMemoryBank
    from .metrics import MetricsTracker
except ImportError:
    from config import SimulationConfig
    from cultural_memory import CulturalMemoryBank
    from metrics import MetricsTracker


@dataclass(slots=True)
class BackendInfo:
    """Selected array backend details."""

    name: str
    device: str


class BatchedSimulation:
    """Vectorized simulation for fast headless experiments.

    This path represents the whole population as arrays instead of Python agent
    objects. When `use_gpu` is true, those arrays live on the CUDA device
    through CuPy. This is where an NVIDIA GPU can actually help.
    """

    input_size = 13
    action_count = 5
    output_size = 6

    def __init__(self, config: SimulationConfig, use_gpu: bool) -> None:
        self.config = config
        self.xp, self.backend = self._select_backend(use_gpu)
        self.rng = self._create_rng(config.seed)
        self.metrics = MetricsTracker()
        self.generation = 0

        population = config.population_size
        hidden = config.hidden_size

        self.x = self.rng.integers(0, config.width, size=population)
        self.y = self.rng.integers(0, config.height, size=population)
        self.energy = self.xp.full(population, config.initial_energy, dtype=self.xp.float32)
        self.age = self.xp.zeros(population, dtype=self.xp.int32)
        self.food_eaten = self.xp.zeros(population, dtype=self.xp.int32)
        self.cooperative_events = self.xp.zeros(population, dtype=self.xp.int32)
        self.previous_action = self.xp.zeros(population, dtype=self.xp.int32)
        self.memory_dx = self.xp.zeros(population, dtype=self.xp.float32)
        self.memory_dy = self.xp.zeros(population, dtype=self.xp.float32)
        self.memory_strength = self.xp.zeros(population, dtype=self.xp.float32)
        self.signal = self.xp.zeros(population, dtype=self.xp.float32)
        self.local_signal = self.xp.zeros(population, dtype=self.xp.float32)
        self.cultural_dx = self.xp.zeros(population, dtype=self.xp.float32)
        self.cultural_dy = self.xp.zeros(population, dtype=self.xp.float32)
        self.cultural_confidence = self.xp.zeros(population, dtype=self.xp.float32)
        self.action_bias = self.xp.zeros((population, self.action_count), dtype=self.xp.float32)
        self.alive = self.xp.ones(population, dtype=bool)
        self.cultural_memory = (
            CulturalMemoryBank(config.cultural_memory_size, config.cultural_memory_radius)
            if config.cultural_memory_enabled
            else None
        )

        self.w1 = self.normal(
            0.0,
            np.sqrt(1.0 / self.input_size),
            size=(population, self.input_size, hidden),
        ).astype(self.xp.float32)
        self.b1 = self.xp.zeros((population, hidden), dtype=self.xp.float32)
        self.w2 = self.normal(
            0.0,
            np.sqrt(1.0 / hidden),
            size=(population, hidden, self.output_size),
        ).astype(self.xp.float32)
        self.b2 = self.xp.zeros((population, self.output_size), dtype=self.xp.float32)

        self.food_x = self.xp.empty(config.food_count, dtype=self.xp.int32)
        self.food_y = self.xp.empty(config.food_count, dtype=self.xp.int32)
        self.reset_food()

    @staticmethod
    def _select_backend(use_gpu: bool):
        if not use_gpu:
            return np, BackendInfo("numpy", "CPU")

        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "GPU mode requires CuPy. Install the CUDA 12 build with: "
                ".venv/bin/python -m pip install cupy-cuda12x"
            ) from exc

        try:
            device_id = cp.cuda.Device().id
            properties = cp.cuda.runtime.getDeviceProperties(device_id)
            device_name = properties["name"].decode("utf-8")
            cp.cuda.Device(device_id).use()
        except Exception as exc:  # pragma: no cover - depends on local CUDA driver.
            raise RuntimeError(
                "CuPy is installed, but CUDA is not usable from this environment. "
                "Check that the NVIDIA driver is installed and that `nvidia-smi` works."
            ) from exc

        return cp, BackendInfo("cupy", f"CUDA:{device_id} {device_name}")

    def _create_rng(self, seed: int | None):
        if self.backend.name == "cupy":
            return self.xp.random.default_rng(seed)
        return np.random.default_rng(seed)

    def reset_food(self) -> None:
        """Place food randomly. Exact uniqueness is less important in batched mode."""

        self.food_x = self.rng.integers(0, self.config.width, size=self.config.food_count).astype(self.xp.int32)
        self.food_y = self.rng.integers(0, self.config.height, size=self.config.food_count).astype(self.xp.int32)

    def reset_lifetimes(self) -> None:
        """Start a new generation with the current evolved brains."""

        population = self.config.population_size
        self.x = self.rng.integers(0, self.config.width, size=population).astype(self.xp.int32)
        self.y = self.rng.integers(0, self.config.height, size=population).astype(self.xp.int32)
        self.energy.fill(self.config.initial_energy)
        self.age.fill(0)
        self.food_eaten.fill(0)
        self.cooperative_events.fill(0)
        self.previous_action.fill(0)
        self.memory_dx.fill(0.0)
        self.memory_dy.fill(0.0)
        self.memory_strength.fill(0.0)
        self.signal.fill(0.0)
        self.local_signal.fill(0.0)
        self.cultural_dx.fill(0.0)
        self.cultural_dy.fill(0.0)
        self.cultural_confidence.fill(0.0)
        self.action_bias.fill(0.0)
        self.alive.fill(True)
        self.reset_food()

    def step(self) -> None:
        """Advance all agents by one vectorized step."""

        xp = self.xp
        config = self.config
        population = config.population_size

        self.update_local_signals()
        self.update_cultural_inputs()

        dx_food = self.food_x[None, :] - self.x[:, None]
        dy_food = self.food_y[None, :] - self.y[:, None]
        dist2 = dx_food * dx_food + dy_food * dy_food
        nearest = xp.argmin(dist2, axis=1)
        nearest_dx = self.food_x[nearest] - self.x
        nearest_dy = self.food_y[nearest] - self.y
        nearest_distance = xp.sqrt(nearest_dx * nearest_dx + nearest_dy * nearest_dy)
        sees_food = nearest_distance <= config.sensory_range
        diagonal = float(np.hypot(config.width, config.height))

        if config.internal_memory_enabled:
            self.memory_dx = xp.where(sees_food, nearest_dx / max(1, config.width), self.memory_dx * config.memory_decay)
            self.memory_dy = xp.where(sees_food, nearest_dy / max(1, config.height), self.memory_dy * config.memory_decay)
            self.memory_strength = xp.where(sees_food, 1.0, self.memory_strength * config.memory_decay)
            observed_dx = xp.where(sees_food, nearest_dx / max(1, config.width), self.memory_dx)
            observed_dy = xp.where(sees_food, nearest_dy / max(1, config.height), self.memory_dy)
        else:
            self.memory_dx.fill(0.0)
            self.memory_dy.fill(0.0)
            self.memory_strength.fill(0.0)
            observed_dx = xp.where(sees_food, nearest_dx / max(1, config.width), 0.0)
            observed_dy = xp.where(sees_food, nearest_dy / max(1, config.height), 0.0)
        observed_distance = xp.where(sees_food, nearest_distance / max(1.0, diagonal), 1.0)

        obs = xp.empty((population, self.input_size), dtype=xp.float32)
        obs[:, 0] = observed_dx
        obs[:, 1] = observed_dy
        obs[:, 2] = observed_distance
        obs[:, 3] = self.energy / max(1.0, config.max_energy)
        obs[:, 4] = (self.previous_action / (self.action_count - 1)) * 2.0 - 1.0
        obs[:, 5] = xp.where(sees_food, 1.0, -1.0)
        obs[:, 6] = self.memory_dx
        obs[:, 7] = self.memory_dy
        obs[:, 8] = self.memory_strength
        obs[:, 9] = self.local_signal if config.communication_enabled else 0.0
        obs[:, 10] = self.cultural_dx if config.cultural_memory_enabled else 0.0
        obs[:, 11] = self.cultural_dy if config.cultural_memory_enabled else 0.0
        obs[:, 12] = self.cultural_confidence if config.cultural_memory_enabled else 0.0

        hidden = xp.tanh(xp.einsum("ni,nih->nh", obs, self.w1) + self.b1)
        logits = xp.einsum("nh,nho->no", hidden, self.w2) + self.b2
        action_logits = logits[:, : self.action_count]
        if config.lifetime_learning_enabled:
            action_logits = action_logits + self.action_bias
        self.signal = xp.where(
            self.alive & config.communication_enabled,
            xp.tanh(logits[:, self.action_count]),
            0.0,
        )
        action_logits = action_logits - xp.max(action_logits, axis=1, keepdims=True)
        probabilities = xp.exp(action_logits)
        probabilities = probabilities / xp.sum(probabilities, axis=1, keepdims=True)
        cumulative = xp.cumsum(probabilities, axis=1)
        draws = self.rng.random(population)[:, None]
        actions = xp.argmax(draws <= cumulative, axis=1).astype(xp.int32)
        actions = xp.where(self.alive, actions, 0)

        move_dx = xp.zeros(population, dtype=xp.int32)
        move_dy = xp.zeros(population, dtype=xp.int32)
        move_dy = xp.where(actions == 1, -1, move_dy)
        move_dy = xp.where(actions == 2, 1, move_dy)
        move_dx = xp.where(actions == 3, -1, move_dx)
        move_dx = xp.where(actions == 4, 1, move_dx)

        self.x = xp.clip(self.x + move_dx, 0, config.width - 1)
        self.y = xp.clip(self.y + move_dy, 0, config.height - 1)
        action_cost = xp.where(actions == 0, config.stay_energy_cost, config.move_energy_cost)
        if config.communication_enabled:
            action_cost = action_cost + xp.abs(self.signal) * config.communication_cost
        self.energy = xp.where(self.alive, self.energy - action_cost, self.energy)
        self.age = xp.where(self.alive, self.age + 1, self.age)
        self.previous_action = actions
        self.alive = self.alive & (self.energy > 0.0)
        self.energy = xp.maximum(self.energy, 0.0)

        self.consume_food()
        if config.lifetime_learning_enabled:
            self.action_bias *= config.action_bias_decay

    def update_local_signals(self) -> None:
        """Average nearby broadcast signals for every agent."""

        if not self.config.communication_enabled:
            self.local_signal.fill(0.0)
            return
        xp = self.xp
        config = self.config
        dx_agents = self.x[None, :] - self.x[:, None]
        dy_agents = self.y[None, :] - self.y[:, None]
        dist2 = dx_agents * dx_agents + dy_agents * dy_agents
        neighbor_mask = (
            (dist2 <= config.communication_radius * config.communication_radius)
            & self.alive[:, None]
            & self.alive[None, :]
        )
        neighbor_mask = neighbor_mask & ~xp.eye(config.population_size, dtype=bool)
        counts = xp.sum(neighbor_mask, axis=1)
        totals = xp.sum(neighbor_mask * self.signal[None, :], axis=1)
        self.local_signal = xp.where(counts > 0, totals / xp.maximum(counts, 1), 0.0)

    def update_cultural_inputs(self) -> None:
        """Read shared cultural knowledge into vectorized input arrays."""

        if self.cultural_memory is None:
            self.cultural_dx.fill(0.0)
            self.cultural_dy.fill(0.0)
            self.cultural_confidence.fill(0.0)
            return
        x_values = self.to_numpy(self.x)
        y_values = self.to_numpy(self.y)
        dx_values, dy_values, confidence_values = self.cultural_memory.read_many(
            x_values,
            y_values,
            self.config.width,
            self.config.height,
        )
        self.cultural_dx = self.xp.asarray(dx_values, dtype=self.xp.float32)
        self.cultural_dy = self.xp.asarray(dy_values, dtype=self.xp.float32)
        self.cultural_confidence = self.xp.asarray(confidence_values, dtype=self.xp.float32)

    def consume_food(self) -> None:
        """Reward agents on food cells and respawn consumed food entries."""

        xp = self.xp
        config = self.config
        agent_cells = self.y * config.width + self.x
        food_cells = self.food_y * config.width + self.food_x
        on_food = xp.any(agent_cells[:, None] == food_cells[None, :], axis=1) & self.alive

        self.food_eaten = xp.where(on_food, self.food_eaten + 1, self.food_eaten)
        self.energy = xp.where(on_food, xp.minimum(config.max_energy, self.energy + config.food_energy), self.energy)
        self.apply_cooperation_credit(on_food)
        self.update_lifetime_learning(on_food)
        self.write_cultural_events(on_food)

        if int(self.to_numpy(xp.sum(on_food))) == 0:
            return

        eaten_food = xp.any(food_cells[:, None] == agent_cells[on_food][None, :], axis=1)
        eaten_count = int(self.to_numpy(xp.sum(eaten_food)))
        if eaten_count:
            self.food_x = xp.where(
                eaten_food,
                self.rng.integers(0, config.width, size=config.food_count).astype(xp.int32),
                self.food_x,
            )
            self.food_y = xp.where(
                eaten_food,
                self.rng.integers(0, config.height, size=config.food_count).astype(xp.int32),
                self.food_y,
            )

    def apply_cooperation_credit(self, on_food) -> None:
        """Credit nearby agents when another agent consumes food."""

        if not self.config.cooperation_enabled:
            return
        xp = self.xp
        config = self.config
        eater_count = int(self.to_numpy(xp.sum(on_food)))
        if eater_count == 0:
            return
        eater_x = self.x[on_food]
        eater_y = self.y[on_food]
        dx = self.x[:, None] - eater_x[None, :]
        dy = self.y[:, None] - eater_y[None, :]
        near_eater = (dx * dx + dy * dy) <= config.cooperation_radius * config.cooperation_radius
        near_eater = near_eater & self.alive[:, None] & ~on_food[:, None]
        self.cooperative_events += xp.sum(near_eater, axis=1).astype(xp.int32)

    def update_lifetime_learning(self, on_food) -> None:
        """Reinforce actions that produced food without modifying inherited weights."""

        if not self.config.lifetime_learning_enabled:
            return
        eater_count = int(self.to_numpy(self.xp.sum(on_food)))
        if eater_count == 0:
            return
        rows = self.xp.arange(self.config.population_size)
        self.action_bias[rows[on_food], self.previous_action[on_food]] += self.config.lifetime_learning_rate

    def write_cultural_events(self, on_food) -> None:
        """Store successful observations in the shared cultural bank."""

        if self.cultural_memory is None:
            return
        eater_count = int(self.to_numpy(self.xp.sum(on_food)))
        if eater_count == 0:
            return
        x_values = self.to_numpy(self.x[on_food])
        y_values = self.to_numpy(self.y[on_food])
        memory_dx_values = self.to_numpy(self.memory_dx[on_food]) * self.config.width
        memory_dy_values = self.to_numpy(self.memory_dy[on_food]) * self.config.height
        signal_values = self.to_numpy(self.signal[on_food])
        for x, y, dx, dy, signal in zip(
            x_values,
            y_values,
            memory_dx_values,
            memory_dy_values,
            signal_values,
        ):
            self.cultural_memory.write_food(
                int(x),
                int(y),
                float(dx),
                float(dy),
                float(signal),
                self.config.food_fitness,
                self.generation,
            )

    def run_generation(self):
        """Run one generation, record metrics, and evolve the population."""

        for _ in range(self.config.generation_steps):
            self.step()

        fitness = (
            self.food_eaten * self.config.food_fitness
            + self.cooperative_events * self.config.cooperation_fitness
            + self.age * self.config.survival_fitness
            + self.energy * self.config.remaining_energy_fitness
        )
        fitness_np = self.to_numpy(fitness)
        food_consumed = int(self.to_numpy(self.xp.sum(self.food_eaten)))
        diversity = self.population_diversity()
        extras = self.cultural_memory.snapshot_stats() if self.cultural_memory is not None else None
        metrics = self.metrics.record_values(self.generation, fitness_np, food_consumed, diversity, extras)
        self.evolve(fitness)
        if self.cultural_memory is not None:
            self.cultural_memory.mark_generation_transfer(self.config.population_size)
        self.generation += 1
        return metrics

    def evolve(self, fitness) -> None:
        """Select elites and refill the population with mutated elite clones."""

        xp = self.xp
        config = self.config
        population = config.population_size
        survivor_count = max(1, int(population * config.survivor_fraction))
        ranked = xp.argsort(fitness)[::-1]
        survivors = ranked[:survivor_count]

        parent_choices = self.rng.integers(0, survivor_count, size=population)
        parent_indices = survivors[parent_choices]
        parent_indices[:survivor_count] = survivors

        self.w1 = self.w1[parent_indices].copy()
        self.b1 = self.b1[parent_indices].copy()
        self.w2 = self.w2[parent_indices].copy()
        self.b2 = self.b2[parent_indices].copy()

        mutate_mask = xp.ones(population, dtype=bool)
        mutate_mask[:survivor_count] = False
        self.apply_mutation(self.w1, mutate_mask)
        self.apply_mutation(self.b1, mutate_mask)
        self.apply_mutation(self.w2, mutate_mask)
        self.apply_mutation(self.b2, mutate_mask)
        self.reset_lifetimes()

    def apply_mutation(self, parameter, mutate_agents) -> None:
        """Apply Gaussian mutation to selected offspring rows."""

        xp = self.xp
        row_shape = (self.config.population_size,) + (1,) * (parameter.ndim - 1)
        active_rows = mutate_agents.reshape(row_shape)
        parameter_mask = self.rng.random(parameter.shape) < self.config.mutation_rate
        noise = self.normal(0.0, self.config.mutation_strength, size=parameter.shape)
        parameter += active_rows * parameter_mask * noise

    def population_diversity(self) -> float:
        """Return mean parameter standard deviation across all brains."""

        xp = self.xp
        flattened = xp.concatenate(
            [
                self.w1.reshape(self.config.population_size, -1),
                self.b1.reshape(self.config.population_size, -1),
                self.w2.reshape(self.config.population_size, -1),
                self.b2.reshape(self.config.population_size, -1),
            ],
            axis=1,
        )
        return float(self.to_numpy(xp.mean(xp.std(flattened, axis=0))))

    def run(self, generations: int, report_every: int = 10) -> MetricsTracker:
        """Run a headless experiment and save outputs."""

        print(f"backend={self.backend.name} device={self.backend.device}")
        for _ in range(generations):
            metrics = self.run_generation()
            if metrics.generation == 0 or (metrics.generation + 1) % report_every == 0:
                print(
                    f"generation={metrics.generation + 1}/{generations} "
                    f"best={metrics.best_fitness:.2f} "
                    f"avg={metrics.average_fitness:.2f} "
                    f"median={metrics.median_fitness:.2f} "
                    f"food={metrics.food_consumed} "
                    f"diversity={metrics.diversity:.4f}"
                )

        self.metrics.save_csv(self.config.metrics_path)
        self.metrics.plot(self.config.plots_dir)
        return self.metrics

    def to_numpy(self, value):
        """Move a backend array or scalar to CPU numpy/Python."""

        if self.backend.name == "cupy":
            return self.xp.asnumpy(value)
        return value

    def normal(self, mean: float, standard_deviation: float, size: tuple[int, ...]):
        """Draw normal noise with an API that works for NumPy and CuPy generators."""

        if hasattr(self.rng, "normal"):
            return self.rng.normal(mean, standard_deviation, size=size)
        return self.rng.standard_normal(size=size) * standard_deviation + mean
