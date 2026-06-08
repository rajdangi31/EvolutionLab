"""Compare evolutionary population training against single-agent Q-learning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class ComparisonSummary:
    """High-level comparison result."""

    evolution_final_average: float
    evolution_final_best: float
    evolution_max_best: float
    q_learning_final_average: float
    q_learning_final_best: float
    winner_by_average: str
    winner_by_best: str


def read_csv(path: Path) -> list[dict[str, float]]:
    """Read a metrics CSV into numeric dictionaries."""

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(csv_file)]


def compare_runs(
    evolution_metrics_path: Path,
    q_learning_metrics_path: Path,
    output_dir: Path,
    window: int = 50,
) -> ComparisonSummary:
    """Generate comparison plots and return a concise summary."""

    evolution = read_csv(evolution_metrics_path)
    q_learning = read_csv(q_learning_metrics_path)
    if not evolution:
        raise ValueError(f"No evolution metrics found in {evolution_metrics_path}")
    if not q_learning:
        raise ValueError(f"No Q-learning metrics found in {q_learning_metrics_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    evolution_x = [row["generation"] for row in evolution]
    q_x = [row["episode"] for row in q_learning]
    q_rolling = rolling_average([row["fitness"] for row in q_learning], window)

    plt.figure(figsize=(11, 6))
    plt.plot(evolution_x, [row["best_fitness"] for row in evolution], label="Evolution best")
    plt.plot(evolution_x, [row["average_fitness"] for row in evolution], label="Evolution population average")
    plt.plot(q_x, q_rolling, label=f"Q-learning {window}-episode rolling average")
    plt.xlabel("Generation / Episode")
    plt.ylabel("Fitness")
    plt.title("Evolution vs single-agent Q-learning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "evolution_vs_q_learning_fitness.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.plot(evolution_x, [row["food_consumed"] for row in evolution], label="Evolution total population food")
    plt.plot(q_x, [row["food_consumed"] for row in q_learning], label="Q-learning single-agent food")
    plt.xlabel("Generation / Episode")
    plt.ylabel("Food consumed")
    plt.title("Food collection comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "evolution_vs_q_learning_food.png", dpi=150)
    plt.close()

    evolution_tail = evolution[-min(window, len(evolution)) :]
    q_tail = q_learning[-min(window, len(q_learning)) :]
    evolution_final_average = float(np.mean([row["average_fitness"] for row in evolution_tail]))
    evolution_final_best = float(np.mean([row["best_fitness"] for row in evolution_tail]))
    evolution_max_best = float(np.max([row["best_fitness"] for row in evolution]))
    q_learning_final_average = float(np.mean([row["fitness"] for row in q_tail]))
    q_learning_final_best = float(np.max([row["fitness"] for row in q_learning]))

    summary = ComparisonSummary(
        evolution_final_average=evolution_final_average,
        evolution_final_best=evolution_final_best,
        evolution_max_best=evolution_max_best,
        q_learning_final_average=q_learning_final_average,
        q_learning_final_best=q_learning_final_best,
        winner_by_average="q_learning"
        if q_learning_final_average > evolution_final_average
        else "evolution",
        winner_by_best="q_learning" if q_learning_final_best > evolution_max_best else "evolution",
    )
    write_summary(output_dir / "comparison_summary.txt", summary, window)
    return summary


def rolling_average(values: list[float], window: int) -> list[float]:
    """Return trailing rolling averages."""

    averages = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        averages.append(float(np.mean(values[start : index + 1])))
    return averages


def write_summary(path: Path, summary: ComparisonSummary, window: int) -> None:
    """Write a human-readable comparison summary."""

    path.write_text(
        "\n".join(
            [
                f"Comparison window: last {window} rows",
                f"Evolution final population average: {summary.evolution_final_average:.3f}",
                f"Evolution final best average: {summary.evolution_final_best:.3f}",
                f"Evolution best agent ever: {summary.evolution_max_best:.3f}",
                f"Q-learning final rolling average: {summary.q_learning_final_average:.3f}",
                f"Q-learning best episode: {summary.q_learning_final_best:.3f}",
                f"Winner by average: {summary.winner_by_average}",
                f"Winner by best: {summary.winner_by_best}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
