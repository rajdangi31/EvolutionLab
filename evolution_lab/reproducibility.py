"""Reproducibility benchmark for the original high-scoring evolution run."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache")))
os.environ.setdefault("CUPY_CACHE_DIR", str(Path(".cupy-cache")))

import matplotlib.pyplot as plt
import numpy as np

try:
    from .accelerated_simulation import BatchedSimulation
    from .config import SimulationConfig
except ImportError:
    from accelerated_simulation import BatchedSimulation
    from config import SimulationConfig


ORIGINAL_SCORE = 9026.0
ORIGINAL_REPORTED_SCORE = 9026.7
THRESHOLDS = (5000.0, 7000.0, 8000.0, 9000.0)
OUTPUT_RESULTS = Path("reproducibility_results.csv")
OUTPUT_TOP_RUNS = Path("top_runs.csv")
OUTPUT_REPORT = Path("REPRODUCIBILITY_REPORT.md")
SEED_DIR = Path("reproducibility_seed_runs")


def run_reproducibility_benchmark(
    base_config: SimulationConfig,
    generations: int = 500,
    seeds: int = 100,
    use_gpu: bool = False,
    seed_start: int = 1,
) -> None:
    """Run independent original-system seeds and write validation outputs."""

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    top_run_details: list[dict[str, Any]] = []

    for offset in range(seeds):
        seed = seed_start + offset
        metrics_path = SEED_DIR / f"seed_{seed}_evolution.csv"
        print(f"\n=== reproducibility seed {offset + 1}/{seeds}: seed={seed} ===")
        row, top_detail = run_seed(base_config, generations, seed, metrics_path, use_gpu)
        rows.append(row)
        top_run_details.append(top_detail)
        write_results_csv(OUTPUT_RESULTS, rows)
        write_top_runs_csv(OUTPUT_TOP_RUNS, top_run_details)

    write_results_csv(OUTPUT_RESULTS, rows)
    write_top_runs_csv(OUTPUT_TOP_RUNS, top_run_details)
    write_plots(rows)
    report_config = replace(base_config, cultural_memory_enabled=False)
    write_report(OUTPUT_REPORT, report_config, generations, seeds, rows, top_run_details)


def run_seed(
    base_config: SimulationConfig,
    generations: int,
    seed: int,
    metrics_path: Path,
    use_gpu: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one complete independent seed while observing per-agent fitness."""

    config = replace(
        base_config,
        seed=seed,
        metrics_path=metrics_path,
        plots_dir=metrics_path.parent / f"seed_{seed}_plots",
        cultural_memory_enabled=False,
    )
    simulation = BatchedSimulation(config, use_gpu=use_gpu)
    print(f"backend={simulation.backend.name} device={simulation.backend.device}")

    best_score = -math.inf
    peak_generation = -1
    final_best_score = math.nan
    final_mean_score = math.nan
    threshold_counts = {threshold: 0 for threshold in THRESHOLDS}
    best_agent_parameters: dict[str, Any] = {}

    for generation in range(generations):
        fitness = evaluate_generation(simulation)
        fitness_np = np.asarray(simulation.to_numpy(fitness), dtype=float)
        best_index = int(np.argmax(fitness_np))
        generation_best = float(fitness_np[best_index])
        generation_mean = float(np.mean(fitness_np))

        for threshold in THRESHOLDS:
            threshold_counts[threshold] += int(np.sum(fitness_np > threshold))

        record_generation_metrics(simulation, fitness, fitness_np)

        if generation_best > best_score:
            best_score = generation_best
            peak_generation = generation
            best_agent_parameters = snapshot_agent_parameters(simulation, best_index)

        if generation == generations - 1:
            final_best_score = generation_best
            final_mean_score = generation_mean

        simulation.evolve(fitness)
        simulation.generation += 1

        if generation == 0 or (generation + 1) % max(1, generations // 10) == 0:
            print(
                f"generation={generation + 1}/{generations} "
                f"best={generation_best:.2f} avg={generation_mean:.2f}"
            )

    simulation.metrics.save_csv(metrics_path)

    row = {
        "seed": seed,
        "best_agent_score_ever": best_score,
        "final_generation_best_score": final_best_score,
        "final_generation_mean_score": final_mean_score,
        "generation_where_best_occurred": peak_generation,
        "agents_exceeding_5000": threshold_counts[5000.0],
        "agents_exceeding_7000": threshold_counts[7000.0],
        "agents_exceeding_8000": threshold_counts[8000.0],
        "agents_exceeding_9000": threshold_counts[9000.0],
    }
    top_detail = {
        "seed": seed,
        "peak_score": best_score,
        "generation_of_peak": peak_generation,
        "agent_parameters": json.dumps(best_agent_parameters, separators=(",", ":")),
        "lineage_information": "not_available_original_algorithm_does_not_track_parent_ids",
    }
    return row, top_detail


def evaluate_generation(simulation: BatchedSimulation):
    """Run all steps for the current generation and return original fitness."""

    for _ in range(simulation.config.generation_steps):
        simulation.step()
    return (
        simulation.food_eaten * simulation.config.food_fitness
        + simulation.cooperative_events * simulation.config.cooperation_fitness
        + simulation.age * simulation.config.survival_fitness
        + simulation.energy * simulation.config.remaining_energy_fitness
    )


def record_generation_metrics(simulation: BatchedSimulation, fitness, fitness_np: np.ndarray) -> None:
    """Record the same metrics as BatchedSimulation.run_generation."""

    food_consumed = int(simulation.to_numpy(simulation.xp.sum(simulation.food_eaten)))
    diversity = simulation.population_diversity()
    extras = simulation.cultural_memory.snapshot_stats() if simulation.cultural_memory is not None else None
    simulation.metrics.record_values(simulation.generation, fitness_np, food_consumed, diversity, extras)


def snapshot_agent_parameters(simulation: BatchedSimulation, index: int) -> dict[str, Any]:
    """Capture the peak agent brain without mutating the simulation."""

    return {
        "w1": np.asarray(simulation.to_numpy(simulation.w1[index])).round(10).tolist(),
        "b1": np.asarray(simulation.to_numpy(simulation.b1[index])).round(10).tolist(),
        "w2": np.asarray(simulation.to_numpy(simulation.w2[index])).round(10).tolist(),
        "b2": np.asarray(simulation.to_numpy(simulation.b2[index])).round(10).tolist(),
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save per-seed benchmark rows."""

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_top_runs_csv(path: Path, details: list[dict[str, Any]]) -> None:
    """Save top 10 run details with captured peak parameters."""

    top_details = sorted(details, key=lambda row: float(row["peak_score"]), reverse=True)[:10]
    if not top_details:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(top_details[0].keys()))
        writer.writeheader()
        writer.writerows(top_details)


def write_plots(rows: list[dict[str, Any]]) -> None:
    """Generate requested best-score and peak-generation plots."""

    best_scores = np.array([float(row["best_agent_score_ever"]) for row in rows], dtype=float)
    peak_generations = np.array([int(row["generation_where_best_occurred"]) for row in rows], dtype=int)

    plt.figure(figsize=(9, 6))
    plt.hist(best_scores, bins=20, color="tab:blue", edgecolor="black")
    plt.axvline(ORIGINAL_REPORTED_SCORE, color="tab:red", linestyle="--", label="Original 9026.7")
    plt.xlabel("Best score per seed")
    plt.ylabel("Run count")
    plt.title("Histogram of best scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_score_histogram.png", dpi=150)
    plt.close()

    sorted_scores = np.sort(best_scores)
    y_values = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.figure(figsize=(9, 6))
    plt.plot(sorted_scores, y_values, marker=".", linestyle="-")
    plt.axvline(ORIGINAL_REPORTED_SCORE, color="tab:red", linestyle="--", label="Original 9026.7")
    plt.xlabel("Best score per seed")
    plt.ylabel("Cumulative probability")
    plt.title("CDF of best scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_score_cdf.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.boxplot(best_scores, vert=True, labels=["Best scores"])
    plt.ylabel("Best score per seed")
    plt.title("Box plot of best scores")
    plt.tight_layout()
    plt.savefig("best_score_boxplot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist(peak_generations, bins=20, color="tab:green", edgecolor="black")
    plt.xlabel("Generation where peak occurred")
    plt.ylabel("Run count")
    plt.title("Distribution of peak generation")
    plt.tight_layout()
    plt.savefig("peak_generation_histogram.png", dpi=150)
    plt.close()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute requested summary statistics."""

    best_scores = [float(row["best_agent_score_ever"]) for row in rows]
    summary: dict[str, Any] = {
        "sample_size": len(best_scores),
        "mean_best_score": mean(best_scores),
        "median_best_score": median(best_scores),
        "std_best_score": stdev(best_scores) if len(best_scores) > 1 else 0.0,
        "min_best_score": min(best_scores),
        "max_best_score": max(best_scores),
        "mean_best_score_ci95": mean_ci(best_scores),
    }
    for threshold in THRESHOLDS:
        exceed = [score > threshold for score in best_scores]
        probability = sum(exceed) / len(exceed)
        summary[f"p_score_gt_{int(threshold)}"] = probability
        summary[f"p_score_gt_{int(threshold)}_ci95"] = wilson_ci(sum(exceed), len(exceed))
    summary["count_gt_9026"] = sum(score > ORIGINAL_SCORE for score in best_scores)
    summary["count_gt_8000"] = sum(score > 8000.0 for score in best_scores)
    summary["count_gt_7000"] = sum(score > 7000.0 for score in best_scores)
    summary["original_percentile"] = percentile_of_score(best_scores, ORIGINAL_REPORTED_SCORE)
    summary["outlier_category"] = outlier_category(summary["count_gt_9026"] / len(best_scores))
    return summary


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Approximate 95% confidence interval for a mean."""

    if len(values) < 2:
        return (values[0], values[0])
    standard_error = stdev(values) / math.sqrt(len(values))
    margin = 1.984 * standard_error
    return (mean(values) - margin, mean(values) + margin)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial probability."""

    if n == 0:
        return (math.nan, math.nan)
    phat = successes / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile_of_score(values: list[float], score: float) -> float:
    """Return empirical percentile rank using <= score."""

    return 100.0 * sum(value <= score for value in values) / len(values)


def outlier_category(probability_exceeding_original: float) -> str:
    """Classify reproducibility with explicit quantitative thresholds."""

    if probability_exceeding_original >= 0.25:
        return "Common"
    if probability_exceeding_original >= 0.05:
        return "Uncommon"
    if probability_exceeding_original >= 0.01:
        return "Rare"
    return "Extreme outlier"


def write_report(
    path: Path,
    config: SimulationConfig,
    generations: int,
    seeds: int,
    rows: list[dict[str, Any]],
    top_details: list[dict[str, Any]],
) -> None:
    """Write the requested Markdown reproducibility report."""

    stats = summarize(rows)
    mean_low, mean_high = stats["mean_best_score_ci95"]
    p7000_low, p7000_high = stats["p_score_gt_7000_ci95"]
    p8000_low, p8000_high = stats["p_score_gt_8000_ci95"]
    p9000_low, p9000_high = stats["p_score_gt_9000_ci95"]
    p9026 = stats["count_gt_9026"] / len(rows)
    conclusion = conclusion_sentence(stats)
    config_dict = asdict(config)
    config_dict["metrics_path"] = str(config_dict["metrics_path"])
    config_dict["plots_dir"] = str(config_dict["plots_dir"])

    lines = [
        "# Reproducibility Report",
        "",
        "## Experimental Setup",
        "",
        f"- Seeds: {seeds}",
        f"- Generations per seed: {generations}",
        f"- Original reference score: {ORIGINAL_REPORTED_SCORE:.1f}",
        "- System tested: original social evolutionary system that produced the 9026.7 score.",
        "- Enabled mechanisms retained from the original run: communication, internal memory, cooperation, lifetime action-bias learning.",
        "- Disabled mechanisms: cultural memory.",
        "- No new rewards, environment mechanics, communication channels, memory systems, or inheritance mechanisms were added.",
        f"- Seed metrics directory: `{SEED_DIR}`",
        "",
        "Configuration snapshot:",
        "",
        "```json",
        json.dumps(config_dict, indent=2, sort_keys=True),
        "```",
        "",
        "## Statistical Summary",
        "",
        f"- Mean best score: {stats['mean_best_score']:.3f}",
        f"- Median best score: {stats['median_best_score']:.3f}",
        f"- Standard deviation: {stats['std_best_score']:.3f}",
        f"- Minimum best score: {stats['min_best_score']:.3f}",
        f"- Maximum best score: {stats['max_best_score']:.3f}",
        "",
        f"- P(score > 5000): {stats['p_score_gt_5000']:.3f}",
        f"- P(score > 7000): {stats['p_score_gt_7000']:.3f}",
        f"- P(score > 8000): {stats['p_score_gt_8000']:.3f}",
        f"- P(score > 9000): {stats['p_score_gt_9000']:.3f}",
        "",
        "## Reproducibility Test",
        "",
        f"1. Runs exceeding 9026: {stats['count_gt_9026']} of {len(rows)}",
        f"2. Runs exceeding 8000: {stats['count_gt_8000']} of {len(rows)}",
        f"3. Runs exceeding 7000: {stats['count_gt_7000']} of {len(rows)}",
        f"4. Original 9026.7 percentile: {stats['original_percentile']:.1f}",
        f"5. Category: {stats['outlier_category']} "
        f"(thresholds: common >=25%, uncommon 5-24.999%, rare 1-4.999%, extreme outlier <1% probability of exceeding 9026).",
        "",
        "## Confidence Intervals",
        "",
        f"- Mean best score 95% CI: [{mean_low:.3f}, {mean_high:.3f}]",
        f"- P(score > 7000) 95% CI: [{p7000_low:.3f}, {p7000_high:.3f}]",
        f"- P(score > 8000) 95% CI: [{p8000_low:.3f}, {p8000_high:.3f}]",
        f"- P(score > 9000) 95% CI: [{p9000_low:.3f}, {p9000_high:.3f}]",
        "",
        "## Plots",
        "",
        "![Histogram of best scores](best_score_histogram.png)",
        "",
        "![CDF of best scores](best_score_cdf.png)",
        "",
        "![Box plot of best scores](best_score_boxplot.png)",
        "",
        "![Peak generation histogram](peak_generation_histogram.png)",
        "",
        "## Outlier Analysis",
        "",
        "Top 10 runs are saved in `top_runs.csv`. Lineage information is reported as unavailable because the original algorithm does not track parent IDs.",
        "",
        "| Rank | Seed | Peak Score | Peak Generation |",
        "|---:|---:|---:|---:|",
    ]
    for rank, detail in enumerate(sorted(top_details, key=lambda row: float(row["peak_score"]), reverse=True)[:10], 1):
        lines.append(
            f"| {rank} | {detail['seed']} | {float(detail['peak_score']):.3f} | "
            f"{int(detail['generation_of_peak'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct Conclusion",
            "",
            conclusion,
            "",
            f"Empirical probability of exceeding 9026: {p9026:.3f}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def conclusion_sentence(stats: dict[str, Any]) -> str:
    """Build the final hypothesis statement."""

    category = str(stats["outlier_category"]).lower()
    count = int(stats["count_gt_9026"])
    n = int(stats["sample_size"])
    if count == 0:
        return (
            f"The {n}-seed evidence rejects the hypothesis that a 9026-score agent is reproducibly obtainable "
            "as a common outcome under the original evolutionary system; in this benchmark it was not reproduced."
        )
    return (
        "The evidence supports reproducibility only at the observed frequency: "
        f"{count} of {n} runs exceeded 9026, making the original score {article_for(category)} {category} "
        "outcome by the stated thresholds."
    )


def article_for(word: str) -> str:
    """Return a simple English article for report text."""

    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
