"""Command-line entry point for EvolutionLab."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .accelerated_simulation import BatchedSimulation
    from .benchmark import run_benchmark
    from .comparison import compare_runs
    from .config import EXPERIMENT_GROUPS, QLearningConfig, SimulationConfig, VisualizationConfig
    from .q_learning import QLearningTrainer
    from .reproducibility import run_reproducibility_benchmark
    from .simulation import Simulation
    from .visualization import run_visualization
except ImportError:
    from accelerated_simulation import BatchedSimulation
    from benchmark import run_benchmark
    from comparison import compare_runs
    from config import EXPERIMENT_GROUPS, QLearningConfig, SimulationConfig, VisualizationConfig
    from q_learning import QLearningTrainer
    from reproducibility import run_reproducibility_benchmark
    from simulation import Simulation
    from visualization import run_visualization


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="EvolutionLab evolutionary food-seeking simulation")
    parser.add_argument("--headless", action="store_true", help="Run without pygame visualization")
    parser.add_argument("--rl", action="store_true", help="Run the single-agent Q-learning baseline")
    parser.add_argument("--compare", action="store_true", help="Compare evolution metrics with Q-learning metrics")
    parser.add_argument("--benchmark", action="store_true", help="Run the reproducible Group A-E benchmark")
    parser.add_argument(
        "--reproducibility",
        action="store_true",
        help="Run the 9026-score reproducibility benchmark for the original evolutionary system",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CUDA/CuPy accelerated headless simulation")
    parser.add_argument("--batched", action="store_true", help="Use faster vectorized CPU headless simulation")
    parser.add_argument("--generations", type=int, default=None, help="Number of generations to run")
    parser.add_argument("--population", type=int, default=1000, help="Population size")
    parser.add_argument("--steps", type=int, default=250, help="Simulation steps per generation")
    parser.add_argument("--width", type=int, default=50, help="Grid width")
    parser.add_argument("--height", type=int, default=50, help="Grid height")
    parser.add_argument("--food", type=int, default=180, help="Number of food items")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    parser.add_argument("--mutation-rate", type=float, default=0.08, help="Per-parameter mutation probability")
    parser.add_argument("--mutation-strength", type=float, default=0.18, help="Gaussian mutation standard deviation")
    parser.add_argument("--sensory-range", type=float, default=15.0, help="Distance at which agents directly sense food")
    parser.add_argument("--memory-decay", type=float, default=0.92, help="Per-step memory retention")
    parser.add_argument("--communication-radius", type=int, default=4, help="Radius for hearing other agents' signals")
    parser.add_argument("--cooperation-radius", type=int, default=3, help="Radius for shared food-discovery credit")
    parser.add_argument("--cooperation-fitness", type=float, default=8.0, help="Fitness bonus for nearby group success")
    parser.add_argument("--communication-cost", type=float, default=0.05, help="Energy cost multiplier for broadcasting")
    parser.add_argument("--disable-communication", action="store_true", help="Disable agent broadcast/receive signals")
    parser.add_argument("--disable-internal-memory", action="store_true", help="Disable lifetime spatial memory")
    parser.add_argument("--disable-cooperation", action="store_true", help="Disable cooperative fitness credit")
    parser.add_argument("--disable-lifetime-learning", action="store_true", help="Disable action-bias lifetime adaptation")
    parser.add_argument("--cultural-memory", action="store_true", help="Enable shared cultural memory bank")
    parser.add_argument("--cultural-memory-size", type=int, default=500, help="Maximum entries in the cultural memory bank")
    parser.add_argument("--cultural-memory-radius", type=float, default=18.0, help="Read radius for cultural memory entries")
    parser.add_argument("--lifetime-learning-rate", type=float, default=0.08, help="Action-bias reinforcement for food actions")
    parser.add_argument("--metrics", type=Path, default=Path("metrics.csv"), help="Metrics CSV path")
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"), help="Directory for generated plots")
    parser.add_argument("--rl-metrics", type=Path, default=Path("q_learning_metrics.csv"), help="Q-learning CSV path")
    parser.add_argument("--rl-plots-dir", type=Path, default=Path("q_learning_plots"), help="Q-learning plot directory")
    parser.add_argument("--comparison-dir", type=Path, default=Path("comparison_plots"), help="Comparison output directory")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark_results"), help="Benchmark output directory")
    parser.add_argument("--benchmark-seeds", type=int, default=30, help="Independent seeds for each benchmark group")
    parser.add_argument("--reproducibility-seeds", type=int, default=100, help="Independent seeds for reproducibility")
    parser.add_argument("--reproducibility-seed-start", type=int, default=1, help="First seed value for reproducibility")
    parser.add_argument(
        "--benchmark-groups",
        default="A,B,C,D,E",
        help="Comma-separated benchmark groups to run, e.g. A,B,E",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    """Create a simulation config from CLI arguments."""

    return SimulationConfig(
        width=args.width,
        height=args.height,
        food_count=args.food,
        population_size=args.population,
        generation_steps=args.steps,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        communication_enabled=not args.disable_communication,
        internal_memory_enabled=not args.disable_internal_memory,
        cooperation_enabled=not args.disable_cooperation,
        cultural_memory_enabled=args.cultural_memory,
        lifetime_learning_enabled=not args.disable_lifetime_learning,
        sensory_range=args.sensory_range,
        memory_decay=args.memory_decay,
        communication_radius=args.communication_radius,
        communication_cost=args.communication_cost,
        cooperation_radius=args.cooperation_radius,
        cooperation_fitness=args.cooperation_fitness,
        cultural_memory_size=args.cultural_memory_size,
        cultural_memory_radius=args.cultural_memory_radius,
        lifetime_learning_rate=args.lifetime_learning_rate,
        seed=args.seed,
        metrics_path=args.metrics,
        plots_dir=args.plots_dir,
    )


def run_headless(config: SimulationConfig, generations: int) -> None:
    """Run fast experiment mode and save metrics automatically."""

    simulation = Simulation(config)

    def report(generation: int, metrics, _population, _world) -> None:
        if generation == 0 or (generation + 1) % 10 == 0:
            print(
                f"generation={generation + 1}/{generations} "
                f"best={metrics.best_fitness:.2f} "
                f"avg={metrics.average_fitness:.2f} "
                f"median={metrics.median_fitness:.2f} "
                f"food={metrics.food_consumed} "
                f"diversity={metrics.diversity:.4f}"
            )

    simulation.run(generations, callback=report, autosave=True)
    print(f"metrics saved to {config.metrics_path}")
    print(f"plots saved to {config.plots_dir}")


def run_batched(config: SimulationConfig, generations: int, use_gpu: bool) -> None:
    """Run the vectorized CPU/GPU experiment path."""

    try:
        simulation = BatchedSimulation(config, use_gpu=use_gpu)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    simulation.run(generations)
    print(f"metrics saved to {config.metrics_path}")
    print(f"plots saved to {config.plots_dir}")


def run_q_learning(config: SimulationConfig, args: argparse.Namespace) -> None:
    """Run the single-agent reinforcement-learning baseline."""

    episodes = args.generations or 500
    rl_config = QLearningConfig(
        episodes=episodes,
        metrics_path=args.rl_metrics,
        plots_dir=args.rl_plots_dir,
    )
    trainer = QLearningTrainer(config, rl_config)
    trainer.train()
    print(f"Q-learning metrics saved to {rl_config.metrics_path}")
    print(f"Q-learning plots saved to {rl_config.plots_dir}")


def run_comparison(args: argparse.Namespace) -> None:
    """Compare saved evolution and Q-learning runs."""

    summary = compare_runs(args.metrics, args.rl_metrics, args.comparison_dir)
    print(f"comparison plots saved to {args.comparison_dir}")
    print(f"winner by final average: {summary.winner_by_average}")
    print(f"winner by best run: {summary.winner_by_best}")


def run_group_benchmark(config: SimulationConfig, args: argparse.Namespace) -> None:
    """Run the full reproducible experiment matrix."""

    group_names = [name.strip().upper() for name in args.benchmark_groups.split(",") if name.strip()]
    unknown_groups = [name for name in group_names if name not in EXPERIMENT_GROUPS]
    if unknown_groups:
        raise SystemExit(f"Unknown benchmark groups: {', '.join(unknown_groups)}")
    run_benchmark(
        config,
        generations=args.generations or 500,
        seeds=args.benchmark_seeds,
        output_dir=args.benchmark_dir,
        group_names=group_names,
        use_gpu=args.gpu,
    )
    print(f"benchmark outputs saved to {args.benchmark_dir}")


def run_reproducibility(config: SimulationConfig, args: argparse.Namespace) -> None:
    """Run the high-score reproducibility validation benchmark."""

    run_reproducibility_benchmark(
        config,
        generations=args.generations or 500,
        seeds=args.reproducibility_seeds,
        use_gpu=args.gpu,
        seed_start=args.reproducibility_seed_start,
    )
    print("reproducibility outputs saved to current directory")


def main() -> None:
    """Run EvolutionLab in visual or headless mode."""

    args = parse_args()
    config = build_config(args)

    if args.gpu and not (args.headless or args.benchmark or args.reproducibility):
        raise SystemExit("--gpu is only supported with --headless, --benchmark, or --reproducibility")

    if args.compare:
        run_comparison(args)
        return

    if args.benchmark:
        run_group_benchmark(config, args)
        return

    if args.reproducibility:
        run_reproducibility(config, args)
        return

    if args.rl:
        run_q_learning(config, args)
        return

    if args.headless:
        if args.gpu or args.batched:
            run_batched(config, args.generations or 500, use_gpu=args.gpu)
            return
        run_headless(config, args.generations or 500)
    else:
        if args.generations is not None:
            print("--generations is only used with --headless; visualization runs until closed.")
        run_visualization(config, VisualizationConfig())


if __name__ == "__main__":
    main()
