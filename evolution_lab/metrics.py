"""Metrics collection, CSV persistence, and plotting."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))

import matplotlib.pyplot as plt
import numpy as np

try:
    from .agent import Agent
except ImportError:
    from agent import Agent


@dataclass(slots=True)
class GenerationMetrics:
    """Summary statistics for one generation."""

    generation: int
    best_fitness: float
    average_fitness: float
    median_fitness: float
    food_consumed: int
    diversity: float
    cultural_entries: float = 0.0
    cultural_reads: float = 0.0
    cultural_writes: float = 0.0
    cultural_transfer_events: float = 0.0
    cultural_entry_uses: float = 0.0
    cultural_mean_value: float = 0.0


class MetricsTracker:
    """Collect, save, and plot simulation metrics."""

    def __init__(self) -> None:
        self.history: list[GenerationMetrics] = []

    def record(self, generation: int, population: list[Agent]) -> GenerationMetrics:
        """Record metrics for a completed generation."""

        fitnesses = np.array([agent.fitness for agent in population], dtype=float)
        food_consumed = int(sum(agent.food_eaten for agent in population))
        diversity = self.population_diversity(population)
        return self.record_values(generation, fitnesses, food_consumed, diversity)

    def record_values(
        self,
        generation: int,
        fitnesses: np.ndarray,
        food_consumed: int,
        diversity: float,
        extras: dict[str, float] | None = None,
    ) -> GenerationMetrics:
        """Record metrics from already-vectorized simulation arrays."""

        extras = extras or {}
        metrics = GenerationMetrics(
            generation=generation,
            best_fitness=float(np.max(fitnesses)),
            average_fitness=float(np.mean(fitnesses)),
            median_fitness=float(np.median(fitnesses)),
            food_consumed=food_consumed,
            diversity=diversity,
            cultural_entries=extras.get("cultural_entries", 0.0),
            cultural_reads=extras.get("cultural_reads", 0.0),
            cultural_writes=extras.get("cultural_writes", 0.0),
            cultural_transfer_events=extras.get("cultural_transfer_events", 0.0),
            cultural_entry_uses=extras.get("cultural_entry_uses", 0.0),
            cultural_mean_value=extras.get("cultural_mean_value", 0.0),
        )
        self.history.append(metrics)
        return metrics

    @staticmethod
    def population_diversity(population: list[Agent]) -> float:
        """Measure average parameter standard deviation across the population."""

        if len(population) < 2:
            return 0.0
        parameter_matrix = np.vstack([agent.brain.flattened_parameters() for agent in population])
        return float(np.mean(np.std(parameter_matrix, axis=0)))

    def save_csv(self, path: Path) -> None:
        """Save all collected metrics to CSV."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "generation",
                    "best_fitness",
                    "average_fitness",
                    "median_fitness",
                    "food_consumed",
                    "diversity",
                    "cultural_entries",
                    "cultural_reads",
                    "cultural_writes",
                    "cultural_transfer_events",
                    "cultural_entry_uses",
                    "cultural_mean_value",
                ]
            )
            for row in self.history:
                writer.writerow(
                    [
                        row.generation,
                        row.best_fitness,
                        row.average_fitness,
                        row.median_fitness,
                        row.food_consumed,
                        row.diversity,
                        row.cultural_entries,
                        row.cultural_reads,
                        row.cultural_writes,
                        row.cultural_transfer_events,
                        row.cultural_entry_uses,
                        row.cultural_mean_value,
                    ]
                )

    def plot(self, output_dir: Path) -> None:
        """Write matplotlib graphs for fitness and diversity."""

        if not self.history:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        generations = [row.generation for row in self.history]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, [row.best_fitness for row in self.history], label="Best fitness")
        plt.plot(generations, [row.average_fitness for row in self.history], label="Average fitness")
        plt.plot(generations, [row.median_fitness for row in self.history], label="Median fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness over generations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "fitness.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(generations, [row.diversity for row in self.history], color="tab:purple")
        plt.xlabel("Generation")
        plt.ylabel("Mean parameter standard deviation")
        plt.title("Population diversity over generations")
        plt.tight_layout()
        plt.savefig(output_dir / "diversity.png", dpi=150)
        plt.close()

        if any(row.cultural_entries or row.cultural_reads or row.cultural_writes for row in self.history):
            plt.figure(figsize=(10, 6))
            plt.plot(generations, [row.cultural_entries for row in self.history], label="Entries")
            plt.plot(generations, [row.cultural_reads for row in self.history], label="Reads")
            plt.plot(generations, [row.cultural_writes for row in self.history], label="Writes")
            plt.xlabel("Generation")
            plt.ylabel("Count")
            plt.title("Cultural memory activity")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "cultural_memory.png", dpi=150)
            plt.close()
