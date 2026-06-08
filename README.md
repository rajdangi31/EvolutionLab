# EvolutionLab

EvolutionLab is a small artificial-life and reinforcement-learning experiment for comparing population-level evolutionary search with a tabular Q-learning baseline in the same food-seeking grid world.

The evolutionary system trains a population of small neural-network agents through:

```text
Evaluation -> Selection -> Mutation
```

The Q-learning baseline trains a single agent by updating tabular action values from reward feedback. The project is intentionally compact so the comparison stays inspectable.

## Highlights

- 2D food-seeking world with energy, survival pressure, and respawning food.
- Evolutionary agents with small neural-network brains.
- Optional communication, internal memory, cooperation, lifetime action-bias learning, and cultural memory.
- Tabular Q-learning baseline in the same environment.
- Reproducibility runs across independent seeds.
- Generated plots and reports for benchmark comparisons.

## Core Fitness Function

The social evolutionary fitness score is:

```text
food eaten * 100
+ cooperative events * 8
+ survival steps * 1
+ remaining energy * 0.25
```

This is an accumulated per-generation score, not a normalized 0-100 rating.

## Key Results

### Social Evolution Reproducibility

The original social evolutionary run produced a best agent score of `9026.7`. A follow-up reproducibility test used:

- 100 independent seeds
- 500 generations per seed
- 1,000 agents per generation
- Top 10% selected each generation
- Communication, internal memory, cooperation, and lifetime action-bias learning enabled
- Cultural memory disabled

Summary:

| Metric | Value |
|---|---:|
| Mean best score | 6837.5 |
| Median best score | 6574.6 |
| Minimum best score | 5196.9 |
| Maximum best score | 12303.3 |
| Runs above 7000 | 22 / 100 |
| Runs above 8000 | 15 / 100 |
| Runs above 9000 | 5 / 100 |
| Runs above original 9026.7 | 5 / 100 |

![Best score histogram](assets/figures/reproducibility_best_score_histogram.png)

![Best score CDF](assets/figures/reproducibility_best_score_cdf.png)

![Reproducibility dashboard](assets/figures/reproducibility_dashboard.png)

### Evolution vs Q-Learning

In the direct social-evolution vs Q-learning comparison:

| Metric | Evolution | Q-learning |
|---|---:|---:|
| Late-window average | 4206.1 population average | 4742.5 rolling average |
| Late-window best average | 6414.9 | N/A |
| Best observed score | 9026.7 | 8074.8 |

That result is not a clean "evolution beats Q-learning" outcome. Q-learning was stronger by late rolling average, while evolution produced the highest single observed outlier in that comparison.

![Social evolution vs Q-learning fitness](assets/figures/social_evolution_vs_q_learning_fitness.png)

The broader 30-seed benchmark also favored Q-learning by average score and strongest individual among the tested groups. See [docs/RESULTS.md](docs/RESULTS.md) for details.

## Installation

Use Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For optional NVIDIA GPU acceleration, install a CuPy build that matches your CUDA setup, for example:

```bash
pip install "cupy-cuda12x[ctk]"
```

## Quick Start

Run a short headless evolutionary smoke test:

```bash
python -m evolution_lab.main --headless --batched --generations 5 --population 100 --steps 50
```

Run the default social evolutionary experiment:

```bash
python -m evolution_lab.main --headless --batched --generations 500
```

Run the Q-learning baseline:

```bash
python -m evolution_lab.main --rl --generations 500
```

Compare saved evolution and Q-learning metrics:

```bash
python -m evolution_lab.main --compare
```

Run the 100-seed reproducibility benchmark:

```bash
python -m evolution_lab.main --reproducibility --generations 500 --reproducibility-seeds 100
```

Run the A-E benchmark matrix:

```bash
python -m evolution_lab.main --benchmark --benchmark-seeds 30 --generations 500
```

## Repository Layout

```text
evolution_lab/             Source code
assets/figures/            Curated plots for the README and docs
docs/                      Experiment notes and result summaries
README.md                  Project overview
requirements.txt           Runtime dependencies
PROJECT_EXPLANATION.md     Longer project explanation
REPRODUCIBILITY_REPORT.md  Generated reproducibility report
```

## Documentation

- [Project explanation](PROJECT_EXPLANATION.md)
- [Experiment design](docs/EXPERIMENTS.md)
- [Result summary](docs/RESULTS.md)
- [Generated reproducibility report](REPRODUCIBILITY_REPORT.md)
- [Generated benchmark report](benchmark_results/BENCHMARK_REPORT.md)

## Interpretation

The current evidence supports a narrow conclusion:

Evolutionary selection can surface strong food-seeking behaviors in this controlled environment, and very high-scoring agents are reproducibly obtainable. They are uncommon rather than typical. Q-learning remains the stronger baseline by average performance in the benchmarked runs.

This should not be described as broad emergent intelligence. It is a small controlled result about selection pressure, mutation, and simple social mechanisms in a toy environment.
