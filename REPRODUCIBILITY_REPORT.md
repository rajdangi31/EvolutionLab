# Reproducibility Report

## Experimental Setup

- Seeds: 100
- Generations per seed: 500
- Original reference score: 9026.7
- System tested: original social evolutionary system that produced the 9026.7 score.
- Enabled mechanisms retained from the original run: communication, internal memory, cooperation, lifetime action-bias learning.
- Disabled mechanisms: cultural memory.
- No new rewards, environment mechanics, communication channels, memory systems, or inheritance mechanisms were added.
- Seed metrics directory: `reproducibility_seed_runs`

Configuration snapshot:

```json
{
  "action_bias_decay": 0.96,
  "communication_cost": 0.05,
  "communication_enabled": true,
  "communication_radius": 4,
  "cooperation_enabled": true,
  "cooperation_fitness": 8.0,
  "cooperation_radius": 3,
  "cultural_memory_enabled": false,
  "cultural_memory_radius": 18.0,
  "cultural_memory_size": 500,
  "cultural_write_threshold": 1,
  "food_count": 180,
  "food_energy": 35.0,
  "food_fitness": 100.0,
  "generation_steps": 250,
  "height": 50,
  "hidden_size": 10,
  "initial_energy": 50.0,
  "internal_memory_enabled": true,
  "lifetime_learning_enabled": true,
  "lifetime_learning_rate": 0.08,
  "max_energy": 100.0,
  "memory_decay": 0.92,
  "metrics_path": "metrics.csv",
  "move_energy_cost": 1.0,
  "mutation_rate": 0.08,
  "mutation_strength": 0.18,
  "plots_dir": "plots",
  "population_size": 1000,
  "remaining_energy_fitness": 0.25,
  "seed": null,
  "sensory_range": 15.0,
  "stay_energy_cost": 0.2,
  "survival_fitness": 1.0,
  "survivor_fraction": 0.1,
  "width": 50
}
```

## Statistical Summary

- Mean best score: 6837.509
- Median best score: 6574.635
- Standard deviation: 1160.715
- Minimum best score: 5196.924
- Maximum best score: 12303.326

- P(score > 5000): 1.000
- P(score > 7000): 0.220
- P(score > 8000): 0.150
- P(score > 9000): 0.050

## Reproducibility Test

1. Runs exceeding 9026: 5 of 100
2. Runs exceeding 8000: 15 of 100
3. Runs exceeding 7000: 22 of 100
4. Original 9026.7 percentile: 95.0
5. Category: Uncommon (thresholds: common >=25%, uncommon 5-24.999%, rare 1-4.999%, extreme outlier <1% probability of exceeding 9026).

## Confidence Intervals

- Mean best score 95% CI: [6607.223, 7067.795]
- P(score > 7000) 95% CI: [0.150, 0.311]
- P(score > 8000) 95% CI: [0.093, 0.233]
- P(score > 9000) 95% CI: [0.022, 0.112]

## Plots

![Histogram of best scores](best_score_histogram.png)

![CDF of best scores](best_score_cdf.png)

![Box plot of best scores](best_score_boxplot.png)

![Peak generation histogram](peak_generation_histogram.png)

## Outlier Analysis

Top 10 runs are saved in `top_runs.csv`. Lineage information is reported as unavailable because the original algorithm does not track parent IDs.

| Rank | Seed | Peak Score | Peak Generation |
|---:|---:|---:|---:|
| 1 | 86 | 12303.326 | 492 |
| 2 | 34 | 11405.708 | 486 |
| 3 | 90 | 9788.000 | 469 |
| 4 | 19 | 9616.822 | 447 |
| 5 | 48 | 9300.131 | 428 |
| 6 | 67 | 8878.413 | 467 |
| 7 | 59 | 8812.545 | 489 |
| 8 | 5 | 8769.144 | 462 |
| 9 | 66 | 8763.814 | 455 |
| 10 | 85 | 8684.386 | 475 |

## Direct Conclusion

The evidence supports reproducibility only at the observed frequency: 5 of 100 runs exceeded 9026, making the original score an uncommon outcome by the stated thresholds.

Empirical probability of exceeding 9026: 0.050.
