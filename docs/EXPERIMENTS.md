# Experiments

EvolutionLab contains three main experiment paths.

## 1. Evolutionary Population Search

The evolutionary mode evaluates a population of neural-network agents in a food-seeking grid world. After each generation:

1. Agents are scored by fitness.
2. The top survivor fraction is retained.
3. The next generation is filled with mutated copies of survivors.

Default social evolutionary settings:

| Parameter | Value |
|---|---:|
| Population size | 1000 |
| Steps per generation | 250 |
| Food count | 180 |
| Survivor fraction | 0.10 |
| Mutation rate | 0.08 |
| Mutation strength | 0.18 |
| Sensory range | 15.0 |
| Communication radius | 4 |
| Cooperation radius | 3 |
| Cooperation reward | 8.0 |

The social configuration includes communication, internal memory, cooperation, and lifetime action-bias adaptation. It does not use backpropagation or gradient descent to train neural-network weights.

## 2. Q-Learning Baseline

The Q-learning baseline trains one agent across repeated episodes in the same kind of food-seeking world.

It uses:

- Tabular state-action values
- Epsilon-greedy exploration
- Bellman Q-value updates
- 500 default episodes

This baseline is useful because it directly updates action values from reward feedback, while the evolutionary population improves through selection and mutation.

## 3. Benchmark Groups

The benchmark matrix compares:

| Group | Description |
|---|---|
| A | Q-learning |
| B | Pure evolution |
| C | Evolution + communication |
| D | Evolution + communication + memory |
| E | Evolution + communication + memory + cultural inheritance |

The benchmark report is generated at `benchmark_results/BENCHMARK_REPORT.md`.

## Reproducibility Benchmark

The reproducibility benchmark retests the original high-scoring social evolutionary setup across many independent seeds.

Current published run:

| Setting | Value |
|---|---:|
| Seeds | 100 |
| Generations per seed | 500 |
| Agents per generation | 1000 |
| Original reference score | 9026.7 |
| Cultural memory | disabled |

This benchmark is summarized in [RESULTS.md](RESULTS.md) and in the generated [REPRODUCIBILITY_REPORT.md](../REPRODUCIBILITY_REPORT.md).
