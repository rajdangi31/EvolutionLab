# EvolutionLab Project Explanation

## 1. What This Project Is

EvolutionLab is a small artificial-life and reinforcement-learning research project.

It creates a simple 2D world where agents move around, search for food, consume energy, and are evaluated by how well they survive and collect food. The project then compares two different ways of producing intelligent behavior:

1. **Evolutionary search**
   - A population of agents is created.
   - Each agent has a small neural-network brain.
   - Neural-network weights are not updated by gradients during an agent's lifetime.
   - At the end of each generation, the best agents reproduce.
   - Their offspring inherit mutated versions of their brains.
   - Long-term improvement comes from selection and mutation.

2. **Single-agent reinforcement learning**
   - One agent repeatedly trains in the same type of world.
   - The agent updates a Q-table from reward feedback.
   - Improvement comes from direct learning through experience.

The purpose of the project is to ask:

> Can evolutionary selection and mutation discover effective food-seeking behavior, and how does that compare against a traditional reinforcement-learning approach?

The project started as a simple evolutionary simulation and was later extended with:

- GPU acceleration
- Q-learning comparison
- memory
- communication
- cooperation
- comparison plots and summaries

## 2. High-Level Idea

The core experiment is easy to describe:

```text
Create agents
Put them in a world with food
Let them move around
Score their behavior
Use the score to improve future behavior
Repeat many times
```

The interesting part is that the evolutionary agents are not directly told how to find food.

They receive observations from the world, pass those observations through a small neural network, and choose actions. If some agents happen to behave better, their brains are copied into the next generation. Random mutations create variation. Over many generations, useful behaviors can spread.

The Q-learning agent is different. It directly updates its action values after receiving rewards. It is a more traditional reinforcement-learning setup.

## 3. Project Structure

The main project code lives in the `evolution_lab/` directory.

```text
evolution_lab/
├── __init__.py
├── accelerated_simulation.py
├── agent.py
├── comparison.py
├── config.py
├── evolution.py
├── main.py
├── metrics.py
├── neural_network.py
├── q_learning.py
├── README.md
├── simulation.py
├── visualization.py
└── world.py
```

The root directory also contains this file:

```text
PROJECT_EXPLANATION.md
```

Generated experiment outputs may include:

```text
metrics.csv
plots/
q_learning_metrics.csv
q_learning_plots/
comparison_plots/
social_evolution_metrics.csv
social_evolution_plots/
social_vs_q_learning_comparison/
```

## 4. The World

The world is a 2D grid.

By default:

```text
width: 50
height: 50
food items: 180
```

The world contains:

- empty cells
- food cells
- agent positions

Food is randomly placed. When food is eaten, it respawns elsewhere.

The world logic is implemented in:

```text
evolution_lab/world.py
```

The `World` class handles:

- food placement
- food respawning
- nearest-food lookup
- food consumption

## 5. The Agents

Agents are implemented in:

```text
evolution_lab/agent.py
```

Each evolutionary agent has:

- `x` position
- `y` position
- energy
- age
- fitness score
- food eaten count
- cooperative event count
- previous action
- memory state
- communication signal
- local received signal
- neural-network brain
- alive/dead state

An agent can take one of five movement actions:

```text
0: stay still
1: move up
2: move down
3: move left
4: move right
```

Moving costs energy. Staying still also costs energy, but less.

If energy reaches zero, the agent dies for the rest of that generation.

## 6. The Original Evolutionary Brain

The first version of the project used a simple neural network with:

```text
5 inputs
5 outputs
```

The inputs were:

- relative `dx` to nearest food
- relative `dy` to nearest food
- normalized distance to nearest food
- normalized energy
- previous action

The outputs were probabilities for the five actions.

That version tested whether basic selection and mutation could evolve food-seeking behavior.

## 7. The Current Social Evolutionary Brain

The current evolutionary model is more advanced.

The vectorized social brain now has:

```text
13 input slots
6 outputs
```

The first 10 inputs are:

1. Observed or remembered `dx` to food
2. Observed or remembered `dy` to food
3. Normalized distance to food
4. Normalized current energy
5. Previous action
6. Whether food is currently visible
7. Memory `dx`
8. Memory `dy`
9. Memory strength
10. Local communication signal from nearby agents

The final 3 input slots are used for cultural-memory direction and confidence. They remain zero unless cultural memory is enabled.

The 6 outputs are:

1. Probability of staying still
2. Probability of moving up
3. Probability of moving down
4. Probability of moving left
5. Probability of moving right
6. Communication signal

The first five outputs are converted into action probabilities with softmax.

The sixth output is passed through `tanh`, producing a signal in:

```text
[-1, 1]
```

This signal is broadcast to nearby agents.

## 8. Neural Network Implementation

The neural network is implemented manually with `numpy` in:

```text
evolution_lab/neural_network.py
```

It does not use:

- PyTorch
- TensorFlow
- JAX
- scikit-learn
- backpropagation
- gradient descent

Architecture:

```text
input layer
hidden layer
output layer
```

Hidden activation:

```text
tanh
```

Action output activation:

```text
softmax
```

For evolution, the network weights are not trained by gradients. They are copied and randomly mutated.

## 9. Memory

Memory is part of the current evolutionary model.

Agents do not always directly sense all food. Food sensing is limited by:

```text
--sensory-range
```

Default:

```text
15.0
```

If food is within sensory range, the agent can observe its direction.

If food is outside sensory range, the agent relies on:

- remembered food direction
- memory strength
- communication from nearby agents
- current energy
- previous action

Memory fades over time using:

```text
--memory-decay
```

Default:

```text
0.92
```

This means memory retains 92% of its strength each step.

Memory makes the problem more interesting because agents cannot always see the target. They may need to keep moving based on remembered information.

## 10. Communication

Communication is also part of the current evolutionary model.

Each agent emits a learned signal from its neural-network brain.

Nearby agents hear the average signal from neighbors within:

```text
--communication-radius
```

Default:

```text
4
```

The received signal becomes one of the neural-network inputs.

The project does not manually define what the signal means.

That is important.

The signal is not hard-coded as:

```text
"food is north"
```

or:

```text
"follow me"
```

Instead, evolution can discover whether any signal is useful. If agents that communicate in some useful way perform better, their brains can become more common over generations.

## 11. Cooperation

Cooperation is implemented as shared local credit.

When an agent eats food, nearby living agents receive cooperative event credit if they are within:

```text
--cooperation-radius
```

Default:

```text
3
```

Each cooperative event contributes to fitness through:

```text
--cooperation-fitness
```

Default:

```text
8.0
```

The fitness function now includes:

```text
food eaten * food_fitness
+ cooperative_events * cooperation_fitness
+ survival_time * survival_fitness
+ remaining_energy * remaining_energy_fitness
```

This gives evolution a reason to favor group-compatible behavior, not only selfish individual food collection.

## 12. Fitness

Fitness is the score used to decide which agents reproduce.

The current evolutionary fitness formula is:

```text
fitness =
    food_eaten * 100
    + cooperative_events * 8
    + survival_time * 1
    + remaining_energy * 0.25
```

The exact values are configurable in `SimulationConfig`.

High fitness means an agent:

- found food
- survived
- retained energy
- was near successful group activity

The score is not normalized to a fixed 0-100 scale. It is an accumulated
per-generation fitness value. In the default social evolutionary run, each
agent lives for up to 250 simulation steps in a 50x50 world with 180 food
items.

Using the default social configuration:

- eating one food item contributes `100`
- each nearby cooperative success contributes `8`
- each step survived contributes `1`
- remaining energy contributes `0.25` per energy unit

So a score around `5000` means the agent found a substantial amount of food
and survived well during that generation. Scores above `8000` or `9000` are
high-end outcomes in this environment, not ordinary per-run guarantees.

## 13. Evolution

Evolution is implemented in:

```text
evolution_lab/evolution.py
```

At the end of each generation:

1. Every agent receives a fitness score.
2. Agents are sorted by fitness.
3. The top 10% survive.
4. The next generation is filled by cloning survivors.
5. Cloned brains receive random Gaussian mutations.
6. The new generation starts fresh in a reset world.

Default survivor fraction:

```text
10%
```

Default mutation rate:

```text
0.08
```

Default mutation strength:

```text
0.18
```

This is evolutionary search. Neural-network weights are not trained during an individual agent's lifetime. In the social configuration, agents can optionally use lifetime action-bias adaptation, but that bias resets between generations and is not backpropagation or Q-learning.

## 14. Visualization

Visualization is implemented with `pygame` in:

```text
evolution_lab/visualization.py
```

Run visual mode with:

```bash
.venv/bin/python evolution_lab/main.py
```

The display shows:

- agents
- food
- world grid
- current generation
- current step
- speed
- best/average fitness from the latest completed generation

Controls:

```text
Space: pause/resume
+ or Up: speed up
- or Down: slow down
```

Visual mode is useful for watching behavior, but it is not the fastest way to run experiments.

## 15. Headless Mode

Headless mode runs without graphics.

Run:

```bash
.venv/bin/python evolution_lab/main.py --headless
```

By default, headless mode runs:

```text
500 generations
```

Headless mode saves:

```text
metrics.csv
plots/fitness.png
plots/diversity.png
```

## 16. Batched CPU Mode

Batched CPU mode is a faster vectorized version of the evolutionary simulation.

Run:

```bash
.venv/bin/python evolution_lab/main.py --headless --batched
```

Instead of looping through Python `Agent` objects one by one, it stores the whole population in arrays.

This is implemented in:

```text
evolution_lab/accelerated_simulation.py
```

Batched mode is useful for faster experiments even without GPU.

## 17. GPU Mode

GPU mode uses CuPy to run the batched simulation on an NVIDIA GPU.

Run:

```bash
.venv/bin/python evolution_lab/main.py --headless --gpu
```

GPU support requires:

```bash
.venv/bin/python -m pip install "cupy-cuda12x[ctk]"
```

GPU mode was verified on:

```text
NVIDIA GeForce RTX 4080 Laptop GPU
```

When GPU mode works, the program prints something like:

```text
backend=cupy device=CUDA:0 NVIDIA GeForce RTX 4080 Laptop GPU
```

GPU mode is headless-only.

The visual `pygame` mode does not meaningfully benefit from GPU acceleration because the bottleneck there is rendering and event handling.

## 18. Q-Learning Baseline

The project also includes a traditional single-agent reinforcement-learning baseline.

It is implemented in:

```text
evolution_lab/q_learning.py
```

Run:

```bash
.venv/bin/python evolution_lab/main.py --rl --generations 500
```

This trains one agent across many episodes.

The agent uses tabular Q-learning.

Q-learning keeps a table:

```text
state -> action values
```

For each state, it stores estimated values for each possible action.

The agent chooses actions using epsilon-greedy exploration:

- sometimes it explores randomly
- otherwise it chooses the action with the best known Q-value

After each action, it updates the Q-value based on:

- immediate reward
- expected future reward
- learning rate
- discount factor

This is very different from evolution.

Evolution says:

```text
Try many agents, keep the best, mutate their children.
```

Q-learning says:

```text
One agent updates its behavior from reward feedback.
```

## 19. Q-Learning State Representation

The Q-learning agent does not use a neural network.

It uses a discretized state:

- bucketed `dx` to nearest food
- bucketed `dy` to nearest food
- distance bucket
- energy bucket
- previous action

This keeps the Q-table finite.

The Q-learning baseline is a strong comparison because the environment has discrete actions and can be represented with compact state buckets.

## 20. Metrics

Metrics are implemented in:

```text
evolution_lab/metrics.py
```

Evolutionary metrics include:

- generation
- best fitness
- average fitness
- median fitness
- food consumed
- population diversity

Q-learning metrics include:

- episode
- fitness
- rolling average fitness
- food consumed
- survival steps
- epsilon
- number of Q-table states

Population diversity measures how different the evolved neural-network parameters are across the population.

High diversity means the population contains many different brains.

Low diversity means the population has converged around similar brains.

## 21. Comparison System

Comparison logic is implemented in:

```text
evolution_lab/comparison.py
```

Run:

```bash
.venv/bin/python evolution_lab/main.py --compare
```

Or compare specific files:

```bash
.venv/bin/python evolution_lab/main.py --compare \
  --metrics social_evolution_metrics.csv \
  --rl-metrics q_learning_metrics.csv \
  --comparison-dir social_vs_q_learning_comparison
```

Comparison output includes:

```text
comparison_summary.txt
evolution_vs_q_learning_fitness.png
evolution_vs_q_learning_food.png
```

The summary reports:

- final evolutionary population average
- final evolutionary best-agent average
- best evolutionary agent ever
- final Q-learning rolling average
- best Q-learning episode
- winner by average
- winner by best

## 22. Latest Experiment Result

The most recent experiment compared:

1. Social evolution with:
   - memory
   - communication
   - cooperation
   - GPU batched execution
   - 500 generations

2. Q-learning with:
   - one training agent
   - 500 episodes

The comparison summary was:

```text
Comparison window: last 50 rows
Evolution final population average: 4206.120
Evolution final best average: 6414.899
Evolution best agent ever: 9026.700
Q-learning final rolling average: 4742.522
Q-learning best episode: 8074.750
Winner by average: q_learning
Winner by best: evolution
```

## 23. Interpretation of the Latest Result

The result is nuanced.

Q-learning had better stable final average performance:

```text
Q-learning final rolling average: 4742.522
Evolution final population average: 4206.120
```

That means Q-learning was more reliable near the end of training.

But social evolution produced the strongest individual result:

```text
Evolution best agent ever: 9026.700
Q-learning best episode: 8074.750
```

That means the evolutionary population found at least one strategy that outperformed the best Q-learning episode.

A reasonable interpretation is:

> Q-learning is more stable and sample-efficient in this environment, but social evolution can discover rarer high-performing strategies.

This is a good experimental observation, but not yet a final scientific conclusion.

## 24. Reproducibility Benchmark

After the original `9026.7` result, the project ran a reproducibility
benchmark to test whether that score was obtainable again under the same
original social evolutionary system.

The benchmark used:

```text
100 independent random seeds
500 generations per seed
1000 agents per generation
top 10% selected each generation
same original social evolutionary configuration
```

The benchmark did not add new rewards, new environment mechanics, cultural
inheritance, extra memory, or extra communication. It was designed as a
validation run, not an optimization run.

The result:

```text
mean best score: 6837.509
median best score: 6574.635
standard deviation: 1160.715
minimum best score: 5196.924
maximum best score: 12303.326
```

Threshold results:

```text
score > 5000: 100 / 100 runs
score > 7000: 22 / 100 runs
score > 8000: 15 / 100 runs
score > 9000: 5 / 100 runs
score > 9026: 5 / 100 runs
```

The original `9026.7` score landed at roughly the `95th percentile` of the
100-seed distribution.

The direct conclusion is:

> The 9026-score agent is reproducibly obtainable, but uncommon. It is not a
> typical run outcome, and it is not a one-in-a-thousand fluke.

Generated reproducibility outputs:

```text
reproducibility_results.csv
top_runs.csv
REPRODUCIBILITY_REPORT.md
best_score_histogram.png
best_score_cdf.png
best_score_boxplot.png
peak_generation_histogram.png
social_graphs/
```

Run the benchmark:

```bash
.venv/bin/python -m evolution_lab.main \
  --reproducibility \
  --generations 500 \
  --reproducibility-seeds 100
```

## 25. Important Caveats

The evolutionary and Q-learning systems are not perfectly equivalent.

Evolution uses a population of agents.

Q-learning uses one agent trained across episodes.

Evolution optimizes neural-network weights.

Q-learning updates a table of state-action values.

Evolution can discover collective/social dynamics.

Q-learning is currently a single-agent learner and does not model communication or cooperation.

Also, food-consumption numbers are not directly comparable in every case:

- evolution's `food_consumed` is population-wide
- Q-learning's `food_consumed` is single-agent per episode

Fitness is more comparable than raw food count, but even fitness should be interpreted carefully.

## 26. Useful Commands

Run visual evolution:

```bash
.venv/bin/python evolution_lab/main.py
```

Run normal headless evolution:

```bash
.venv/bin/python evolution_lab/main.py --headless
```

Run batched CPU evolution:

```bash
.venv/bin/python evolution_lab/main.py --headless --batched
```

Run GPU evolution:

```bash
.venv/bin/python evolution_lab/main.py --headless --gpu
```

Run social evolution with custom parameters:

```bash
.venv/bin/python evolution_lab/main.py --headless --gpu \
  --generations 500 \
  --sensory-range 12 \
  --memory-decay 0.95 \
  --communication-radius 5 \
  --cooperation-radius 4 \
  --cooperation-fitness 10
```

Run Q-learning:

```bash
.venv/bin/python evolution_lab/main.py --rl --generations 500
```

Compare default files:

```bash
.venv/bin/python evolution_lab/main.py --compare
```

Compare social evolution against Q-learning:

```bash
.venv/bin/python evolution_lab/main.py --compare \
  --metrics social_evolution_metrics.csv \
  --rl-metrics q_learning_metrics.csv \
  --comparison-dir social_vs_q_learning_comparison
```

## 27. Recommended Next Experiments

The most important next experiment is multi-seed benchmarking.

Suggested design:

```text
seeds: 1 through 20
evolution generations: 500
Q-learning episodes: 500
same world size
same food count
same step limit
```

Collect:

- evolution final population average
- evolution best-ever fitness
- Q-learning final rolling average
- Q-learning best-ever fitness
- diversity trend
- food consumed

Then ask:

```text
Does Q-learning usually win by average?
Does social evolution usually win by best-ever agent?
Does cooperation improve evolution over non-social evolution?
Does communication radius matter?
Does memory decay matter?
```

Other useful experiments:

- disable cooperation and keep memory/communication
- disable communication and keep cooperation/memory
- disable memory and keep cooperation/communication
- compare social evolution against original evolution
- increase scarcity by reducing food count
- add predators
- add agent reproduction costs
- add communication channels with more than one signal
- add recurrent memory
- add separate species
- add multi-agent Q-learning baseline

## 28. What The Project Currently Suggests

The current evidence suggests:

1. Evolution alone can discover food-seeking behavior.
2. Social features can improve evolutionary search.
3. Q-learning is more stable in this simple environment.
4. Social evolution can produce unusually strong individual agents.
5. In the 100-seed reproducibility run, very high scores were real but uncommon.

The most compact summary is:

> EvolutionLab shows that selection pressure can produce meaningful behavioral improvement in a population of small agents, while the strongest behaviors appear as uncommon high-end outcomes rather than reliable guarantees.

## 29. Cultural Inheritance Extension

The project now includes a dedicated cumulative cultural inheritance benchmark.

It adds a global bounded cultural memory bank. Agents can write successful food-related observations during life, and future generations can read from the same bank.

The memory bank stores structured entries:

```text
kind
x
y
dx
dy
signal
value
generation
uses
```

This lets the project ask whether useful knowledge can persist outside any single agent's neural weights.

The benchmark groups are:

```text
Group A: Q-learning
Group B: Pure evolution
Group C: Evolution + communication
Group D: Evolution + communication + memory
Group E: Evolution + communication + memory + cultural inheritance
```

Run the full benchmark:

```bash
.venv/bin/python evolution_lab/main.py --benchmark --gpu --benchmark-seeds 30 --generations 500
```

Run a quick smoke test:

```bash
.venv/bin/python evolution_lab/main.py --benchmark \
  --benchmark-seeds 2 \
  --generations 3 \
  --population 30 \
  --steps 15 \
  --food 15
```

Outputs include:

```text
benchmark_results/statistical_summary.csv
benchmark_results/BENCHMARK_REPORT.md
benchmark_results/mean_score_curves.png
benchmark_results/best_score_curves.png
benchmark_results/cultural_memory_curves.png
benchmark_results/group_E/seed_*_cultural_memory.csv
```

The report answers:

- Which approach performs best?
- Which reaches competence fastest?
- Which discovers the strongest individual?
- Does cultural inheritance create cumulative progress?
- Does knowledge persist across generations?

The implementation also adds visualization overlays:

- communication signal rings
- memory direction traces
- cultural-memory read markers
