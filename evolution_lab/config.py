"""Configuration values for EvolutionLab."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(slots=True)
class SimulationConfig:
    """Runtime configuration for the evolutionary simulation."""

    width: int = 50
    height: int = 50
    food_count: int = 180
    population_size: int = 1000
    generation_steps: int = 250
    initial_energy: float = 50.0
    max_energy: float = 100.0
    move_energy_cost: float = 1.0
    stay_energy_cost: float = 0.2
    food_energy: float = 35.0
    food_fitness: float = 100.0
    survival_fitness: float = 1.0
    remaining_energy_fitness: float = 0.25
    hidden_size: int = 10
    communication_enabled: bool = True
    internal_memory_enabled: bool = True
    cooperation_enabled: bool = True
    cultural_memory_enabled: bool = False
    lifetime_learning_enabled: bool = True
    sensory_range: float = 15.0
    memory_decay: float = 0.92
    communication_radius: int = 4
    communication_cost: float = 0.05
    cooperation_radius: int = 3
    cooperation_fitness: float = 8.0
    cultural_memory_size: int = 500
    cultural_memory_radius: float = 18.0
    cultural_write_threshold: int = 1
    lifetime_learning_rate: float = 0.08
    action_bias_decay: float = 0.96
    survivor_fraction: float = 0.10
    mutation_rate: float = 0.08
    mutation_strength: float = 0.18
    seed: int | None = None
    metrics_path: Path = Path("metrics.csv")
    plots_dir: Path = Path("plots")


@dataclass(slots=True)
class VisualizationConfig:
    """Pygame visualization settings."""

    cell_size: int = 12
    fps: int = 60
    initial_steps_per_frame: int = 1
    min_steps_per_frame: int = 1
    max_steps_per_frame: int = 50
    background_color: tuple[int, int, int] = (18, 20, 24)
    grid_color: tuple[int, int, int] = (36, 39, 46)
    food_color: tuple[int, int, int] = (80, 210, 120)
    agent_color: tuple[int, int, int] = (80, 145, 245)
    best_agent_color: tuple[int, int, int] = (245, 210, 75)
    text_color: tuple[int, int, int] = (235, 235, 235)


@dataclass(slots=True)
class QLearningConfig:
    """Configuration for the single-agent Q-learning baseline."""

    episodes: int = 500
    learning_rate: float = 0.12
    discount_factor: float = 0.95
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_fraction: float = 0.80
    distance_bins: tuple[float, ...] = (1.0, 3.0, 6.0, 10.0, 15.0, 25.0, 50.0)
    energy_bins: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    delta_clip: int = 10
    metrics_path: Path = Path("q_learning_metrics.csv")
    plots_dir: Path = Path("q_learning_plots")


@dataclass(frozen=True, slots=True)
class ExperimentGroup:
    """Named experimental group and its enabled mechanisms."""

    name: str
    label: str
    q_learning: bool = False
    communication: bool = False
    internal_memory: bool = False
    cultural_memory: bool = False
    lifetime_learning: bool = False
    cooperation: bool = False


EXPERIMENT_GROUPS: dict[str, ExperimentGroup] = {
    "A": ExperimentGroup("A", "Q-learning", q_learning=True),
    "B": ExperimentGroup("B", "Pure evolution"),
    "C": ExperimentGroup("C", "Evolution + communication", communication=True),
    "D": ExperimentGroup(
        "D",
        "Evolution + communication + memory",
        communication=True,
        internal_memory=True,
        lifetime_learning=True,
    ),
    "E": ExperimentGroup(
        "E",
        "Evolution + communication + memory + cultural inheritance",
        communication=True,
        internal_memory=True,
        cultural_memory=True,
        lifetime_learning=True,
    ),
}


def config_for_group(base: SimulationConfig, group: ExperimentGroup) -> SimulationConfig:
    """Return a simulation config with only the group's mechanisms enabled."""

    return replace(
        base,
        communication_enabled=group.communication,
        internal_memory_enabled=group.internal_memory,
        cooperation_enabled=group.cooperation,
        cultural_memory_enabled=group.cultural_memory,
        lifetime_learning_enabled=group.lifetime_learning,
    )
