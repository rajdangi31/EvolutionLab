# EvolutionLab

EvolutionLab is a Python 3.12+ grid-world experiment that tests whether evolutionary selection and mutation alone can produce increasingly competent food-seeking agents.

Agents move through a configurable 2D world containing respawning food. Each agent has a small feedforward neural-network brain implemented manually with `numpy`. There is no backpropagation, gradient descent, reinforcement learning algorithm, PyTorch, TensorFlow, or JAX. Improvement comes from ranking agents by fitness, preserving the best performers, and filling the next generation with mutated copies of successful brains.

## Installation

```bash
pip install numpy matplotlib pygame
```

## Run

From this directory:

```bash
cd evolution_lab
python main.py
```

Headless experiment mode runs faster and saves metrics plus plots automatically:

```bash
cd evolution_lab
python main.py --headless
```

The default headless run executes 500 generations. For a quick smoke test:

```bash
python main.py --headless --generations 5 --population 100 --steps 50
```

For a faster vectorized CPU run:

```bash
python main.py --headless --batched
```

For NVIDIA GPU acceleration, install CuPy for CUDA 12 and run:

```bash
pip install "cupy-cuda12x[ctk]"
python main.py --headless --gpu
```

GPU mode is headless-only and requires the NVIDIA driver to be visible. Check that this works before expecting GPU mode to run:

```bash
nvidia-smi
```

The visual `pygame` mode still renders on the CPU because the bottleneck there is drawing and event handling, not large neural-network training.

## Single-Agent Reinforcement Learning Baseline

EvolutionLab also includes a traditional single-agent reinforcement-learning baseline using tabular Q-learning:

```bash
python main.py --rl --generations 500
```

This trains one agent across repeated episodes. The agent updates a Q-table after each action using reward feedback. This is different from the evolutionary mode:

- Evolution trains a population by selection and mutation.
- Q-learning trains one agent by updating action values from experience.

Q-learning outputs:

- `q_learning_metrics.csv`
- `q_learning_plots/q_learning_fitness.png`
- `q_learning_plots/q_learning_food.png`

After running both evolution and Q-learning, compare them:

```bash
python main.py --compare
```

Comparison outputs are written to `comparison_plots/`.

## Cultural Inheritance Benchmark

EvolutionLab includes a reproducible benchmark for testing cumulative cultural inheritance across five groups:

- **Group A**: Q-learning
- **Group B**: Pure evolution
- **Group C**: Evolution + communication
- **Group D**: Evolution + communication + memory
- **Group E**: Evolution + communication + memory + cultural inheritance

Run the full 30-seed benchmark:

```bash
python main.py --benchmark --gpu --benchmark-seeds 30 --generations 500
```

For a quick smoke test:

```bash
python main.py --benchmark --benchmark-seeds 2 --generations 3 --population 30 --steps 15 --food 15
```

Benchmark outputs are written to `benchmark_results/` by default:

- `statistical_summary.csv`
- `BENCHMARK_REPORT.md`
- `mean_score_curves.png`
- `best_score_curves.png`
- `cultural_memory_curves.png`
- per-seed metrics under `group_A/` through `group_E/`
- Group E cultural memory snapshots as `seed_*_cultural_memory.csv`

The cultural memory bank stores bounded structured entries for successful food-related observations:

- location
- remembered direction
- communication signal
- value
- generation
- use count

Agents in Group E inherit access to the same cultural memory bank across generations. This lets the benchmark test whether knowledge persists and produces cumulative progress beyond ordinary genetic inheritance.

## Controls

- `Space`: pause or resume
- `+` / `Up`: speed up
- `-` / `Down`: slow down

## How It Works

Each agent observes:

- Relative `dx` to nearest food
- Relative `dy` to nearest food
- Normalized distance to nearest food
- Normalized current energy
- Previous action encoded as a scalar

The neural network maps those five inputs to five action probabilities:

- Stay still
- Move up
- Move down
- Move left
- Move right

The hidden layer uses `tanh`. The output layer uses `softmax`. An action is sampled from the output probability distribution.

## Evolutionary Mechanics

At the end of each generation, every agent receives a fitness score:

```text
food eaten * 100
+ cooperative events * 8
+ survival steps * 1
+ remaining energy * 0.25
```

The population is sorted by fitness. The top 10% survive as elites, and the rest of the next population is produced by cloning survivor brains and applying Gaussian mutation to random parameters. This is intentionally simple so the experiment isolates selection and mutation as the source of behavioral improvement.

## Cooperation, Communication, and Memory

The evolutionary agents now have three social/cognitive extensions:

- **Memory**: agents remember the last useful food direction they sensed. Memory fades each step according to `--memory-decay`.
- **Communication**: each brain emits a learned signal in `[-1, 1]`. Nearby agents hear the average signal within `--communication-radius`.
- **Cooperation**: when an agent eats food, nearby living agents receive cooperative credit within `--cooperation-radius`.

Direct food sensing is limited by `--sensory-range`, which makes memory and communication meaningful. If food is outside this range, the agent must rely on memory, local broadcast signals, energy, and previous action.

Useful tuning flags:

```bash
python main.py --headless --gpu \
  --sensory-range 12 \
  --memory-decay 0.95 \
  --communication-radius 5 \
  --cooperation-radius 4 \
  --cooperation-fitness 10
```

The vectorized neural-network brain uses 13 input slots and 6 outputs. The first 10 inputs cover food direction, energy, previous action, visibility, internal memory, and local communication. The last 3 inputs are cultural-memory fields and remain zero unless cultural memory is enabled. The first 5 outputs choose movement. The 6th output is the communication signal.

## Metrics

EvolutionLab tracks:

- Best fitness per generation
- Average fitness
- Median fitness
- Food consumed
- Population diversity

Metrics are saved to `metrics.csv`. Plots are saved to `plots/fitness.png` and `plots/diversity.png`.

Population diversity is measured as the mean standard deviation of neural-network parameters across the population.

The evolutionary score is an accumulated per-generation fitness value, not a
fixed 0-100 rating. In the current social evolutionary configuration:

```text
fitness =
    food eaten * 100
    + cooperative events * 8
    + survival steps * 1
    + remaining energy * 0.25
```

With the default 250-step generation, scores above 5000 indicate strong
food-seeking and survival behavior. Scores above 8000 or 9000 are high-end
outcomes in this environment.

## Reproducibility Result

The original social evolutionary experiment produced a best agent score of
approximately `9026.7`. To test whether this was reproducible, the project ran
100 independent seeds using the same original social evolutionary setup:

```text
100 seeds
500 generations per seed
1000 agents per generation
top 10% selected each generation
```

No new rewards, environment mechanics, cultural inheritance, extra memory, or
extra communication were added for this validation run.

Summary:

```text
mean best score: 6837.5
median best score: 6574.6
max observed: 12303.3
score > 7000: 22 / 100 runs
score > 8000: 15 / 100 runs
score > 9000: 5 / 100 runs
score > 9026: 5 / 100 runs
```

Conclusion:

> The 9026-score agent is reproducibly obtainable, but uncommon. It is not a
> typical run outcome, and it is not a one-in-a-thousand fluke.

Outputs:

```text
REPRODUCIBILITY_REPORT.md
reproducibility_results.csv
top_runs.csv
social_graphs/
```

Run the reproducibility benchmark:

```bash
python main.py --reproducibility --generations 500 --reproducibility-seeds 100
```

## Research Question

The project asks:

> Can evolutionary selection alone produce increasingly competent food-seeking behavior in a simple grid world?

The test is whether best and average fitness improve over generations without any learning during an agent lifetime. Agents do not receive gradients or direct instruction. They only inherit mutated policies from earlier generations.

## Expected Outcomes

In successful runs, best fitness should usually rise first as random policies discover useful movement biases. Average fitness may improve more slowly because mutation continues to introduce poor policies. Diversity may drop after strong selection and rise when mutation creates new variants.

Because the world is stochastic, individual runs can vary. Use seeds and multiple trials for stronger conclusions.

## Limitations

- The observation space is small and only points to the nearest food.
- Agents do not sense other agents.
- Agents do not reproduce during their lifetime.
- The environment has no predators, hazards, seasons, or scarcity dynamics beyond food placement.
- Neural networks have no memory.
- Selection pressure is hand-designed through the fitness function.
- This is not proof of general intelligence; it is a minimal evolutionary search experiment.

## Future Extensions

- Predators
- Multi-agent cooperation
- Reproduction costs
- Communication between agents
- Memory
- Cultural learning
- Agent societies

## File Overview

- `main.py`: command-line entry point
- `simulation.py`: generation and step loop
- `world.py`: grid and food logic
- `agent.py`: agent state, observations, actions, fitness
- `neural_network.py`: manual numpy neural network
- `evolution.py`: selection, cloning, mutation
- `visualization.py`: pygame renderer and controls
- `config.py`: runtime configuration
- `metrics.py`: CSV and matplotlib output
