"""Reproducible multi-seed benchmarks for EvolutionLab experimental groups."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np

try:
    from .accelerated_simulation import BatchedSimulation
    from .config import EXPERIMENT_GROUPS, QLearningConfig, SimulationConfig, config_for_group
    from .q_learning import QLearningTrainer
except ImportError:
    from accelerated_simulation import BatchedSimulation
    from config import EXPERIMENT_GROUPS, QLearningConfig, SimulationConfig, config_for_group
    from q_learning import QLearningTrainer


def run_benchmark(
    base_config: SimulationConfig,
    generations: int,
    seeds: int,
    output_dir: Path,
    group_names: list[str],
    use_gpu: bool,
) -> None:
    """Run all requested groups and generate summary outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_values = list(range(1, seeds + 1))
    summary_rows: list[dict[str, float | str]] = []
    learning_curves: dict[str, list[list[float]]] = {}
    best_curves: dict[str, list[list[float]]] = {}
    cultural_curves: dict[str, list[list[float]]] = {}

    for group_name in group_names:
        group = EXPERIMENT_GROUPS[group_name]
        group_dir = output_dir / f"group_{group.name}"
        group_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Group {group.name}: {group.label} ===")

        final_scores: list[float] = []
        best_scores: list[float] = []
        competence_generations: list[float] = []
        learning_curves[group.name] = []
        best_curves[group.name] = []
        cultural_curves[group.name] = []

        for seed in seed_values:
            print(f"seed={seed}/{seeds}")
            if group.q_learning:
                metrics_path = group_dir / f"seed_{seed}_q_learning.csv"
                if metrics_complete(metrics_path, generations):
                    print(f"  reusing {metrics_path}")
                else:
                    run_q_learning_seed(base_config, generations, seed, metrics_path)
                rows = read_csv(metrics_path)
                curve = [row["rolling_average_fitness"] for row in rows]
                best_curve = [row["fitness"] for row in rows]
                final_score = curve[-1]
                best_score = max(best_curve)
                cultural_curve = [0.0 for _ in rows]
            else:
                metrics_path = group_dir / f"seed_{seed}_evolution.csv"
                memory_path = group_dir / f"seed_{seed}_cultural_memory.csv"
                if metrics_complete(metrics_path, generations):
                    print(f"  reusing {metrics_path}")
                else:
                    run_evolution_seed(base_config, group_name, generations, seed, metrics_path, memory_path, use_gpu)
                rows = read_csv(metrics_path)
                curve = [row["average_fitness"] for row in rows]
                best_curve = [row["best_fitness"] for row in rows]
                final_score = mean(curve[-min(50, len(curve)) :])
                best_score = max(best_curve)
                cultural_curve = [row.get("cultural_entries", 0.0) for row in rows]

            final_scores.append(final_score)
            best_scores.append(best_score)
            learning_curves[group.name].append(curve)
            best_curves[group.name].append(best_curve)
            cultural_curves[group.name].append(cultural_curve)
            competence_generations.append(first_competent_generation(curve))

        summary_rows.append(
            summarize_group(
                group.name,
                group.label,
                final_scores,
                best_scores,
                competence_generations,
            )
        )

    write_summary_csv(output_dir / "statistical_summary.csv", summary_rows)
    plot_learning_curves(output_dir, learning_curves, "mean_score_curves.png", "Mean score learning curves")
    plot_learning_curves(output_dir, best_curves, "best_score_curves.png", "Best score learning curves")
    plot_learning_curves(output_dir, cultural_curves, "cultural_memory_curves.png", "Cultural memory growth")
    write_report(output_dir / "BENCHMARK_REPORT.md", summary_rows)


def run_q_learning_seed(
    base_config: SimulationConfig,
    episodes: int,
    seed: int,
    metrics_path: Path,
) -> None:
    """Run one Q-learning seed."""

    config = replace(base_config, seed=seed)
    rl_config = QLearningConfig(
        episodes=episodes,
        metrics_path=metrics_path,
        plots_dir=metrics_path.parent / f"seed_{seed}_q_plots",
    )
    trainer = QLearningTrainer(config, rl_config)
    trainer.train()


def run_evolution_seed(
    base_config: SimulationConfig,
    group_name: str,
    generations: int,
    seed: int,
    metrics_path: Path,
    memory_path: Path,
    use_gpu: bool,
) -> None:
    """Run one evolutionary seed for a configured group."""

    group = EXPERIMENT_GROUPS[group_name]
    config = config_for_group(
        replace(
            base_config,
            seed=seed,
            metrics_path=metrics_path,
            plots_dir=metrics_path.parent / f"seed_{seed}_plots",
        ),
        group,
    )
    simulation = BatchedSimulation(config, use_gpu=use_gpu)
    simulation.run(generations, report_every=max(1, generations // 5))
    if simulation.cultural_memory is not None:
        simulation.cultural_memory.save_csv(memory_path)


def read_csv(path: Path) -> list[dict[str, float]]:
    """Read numeric CSV rows."""

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(csv_file)]


def metrics_complete(path: Path, expected_rows: int) -> bool:
    """Return true when a previous seed output can be reused."""

    if not path.exists():
        return False
    try:
        rows = read_csv(path)
    except (OSError, ValueError):
        return False
    return len(rows) >= expected_rows


def first_competent_generation(curve: list[float]) -> float:
    """Return first generation that reaches half of the run's final peak."""

    if not curve:
        return float("nan")
    threshold = max(curve) * 0.5
    for index, value in enumerate(curve):
        if value >= threshold:
            return float(index)
    return float(len(curve) - 1)


def summarize_group(
    group: str,
    label: str,
    final_scores: list[float],
    best_scores: list[float],
    competence_generations: list[float],
) -> dict[str, float | str]:
    """Create one statistical summary row."""

    return {
        "group": group,
        "label": label,
        "mean_score": mean(final_scores),
        "median_score": median(final_scores),
        "best_score": max(best_scores),
        "std_score": stdev(final_scores) if len(final_scores) > 1 else 0.0,
        "mean_best_score": mean(best_scores),
        "mean_competence_generation": mean(competence_generations),
    }


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Save aggregate statistics."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(
    output_dir: Path,
    curves_by_group: dict[str, list[list[float]]],
    filename: str,
    title: str,
) -> None:
    """Plot mean curves with one standard deviation bands."""

    plt.figure(figsize=(11, 6))
    for group_name, curves in curves_by_group.items():
        if not curves:
            continue
        min_length = min(len(curve) for curve in curves)
        matrix = np.array([curve[:min_length] for curve in curves], dtype=float)
        x_values = np.arange(min_length)
        means = np.mean(matrix, axis=0)
        stds = np.std(matrix, axis=0)
        label = f"{group_name}: {EXPERIMENT_GROUPS[group_name].label}"
        plt.plot(x_values, means, label=label)
        plt.fill_between(x_values, means - stds, means + stds, alpha=0.15)
    plt.xlabel("Generation / Episode")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def write_report(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Generate a Markdown benchmark report."""

    if not rows:
        path.write_text("# Benchmark Report\n\nNo rows generated.\n", encoding="utf-8")
        return

    by_average = max(rows, key=lambda row: float(row["mean_score"]))
    by_best = max(rows, key=lambda row: float(row["best_score"]))
    by_speed = min(rows, key=lambda row: float(row["mean_competence_generation"]))
    cultural = next((row for row in rows if row["group"] == "E"), None)
    memory = next((row for row in rows if row["group"] == "D"), None)

    lines = [
        "# EvolutionLab Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Best mean score: Group {by_average['group']} ({by_average['label']})",
        f"- Fastest competence: Group {by_speed['group']} ({by_speed['label']})",
        f"- Strongest individual: Group {by_best['group']} ({by_best['label']})",
        "",
        "## Statistical Summary",
        "",
        "| Group | Label | Mean | Median | Best | Std | Mean Best | Competence Gen |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['label']} | "
            f"{float(row['mean_score']):.3f} | "
            f"{float(row['median_score']):.3f} | "
            f"{float(row['best_score']):.3f} | "
            f"{float(row['std_score']):.3f} | "
            f"{float(row['mean_best_score']):.3f} | "
            f"{float(row['mean_competence_generation']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Research Questions",
            "",
            f"**Which approach performs best?** Group {by_average['group']} has the highest mean final score.",
            "",
            f"**Which reaches competence fastest?** Group {by_speed['group']} reaches half of its peak score earliest on average.",
            "",
            f"**Which discovers the strongest individual?** Group {by_best['group']} has the highest best observed score.",
            "",
            "**Does cultural inheritance create cumulative progress?** "
            + cultural_progress_sentence(cultural, memory),
            "",
            "**Does knowledge persist across generations?** Check `cultural_memory_curves.png` and Group E's per-seed `seed_*_cultural_memory.csv` files. Persistent nonzero entries and increasing read/use counts indicate persistence.",
            "",
            "## Output Files",
            "",
            "- `statistical_summary.csv`",
            "- `mean_score_curves.png`",
            "- `best_score_curves.png`",
            "- `cultural_memory_curves.png`",
            "- per-seed group metrics under `group_*` directories",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cultural_progress_sentence(
    cultural: dict[str, float | str] | None,
    memory: dict[str, float | str] | None,
) -> str:
    """Compare Group E against Group D when available."""

    if cultural is None or memory is None:
        return "Group D and Group E were not both run, so this cannot be answered from this benchmark."
    delta = float(cultural["mean_score"]) - float(memory["mean_score"])
    if delta > 0:
        return f"Group E outperformed Group D by {delta:.3f} mean score, supporting cumulative cultural benefit in this run."
    return f"Group E underperformed Group D by {abs(delta):.3f} mean score, so this run does not support a cultural benefit."
