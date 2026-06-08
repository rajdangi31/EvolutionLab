# EvolutionLab Benchmark Report

## Summary

- Best mean score: Group A (Q-learning)
- Fastest competence: Group C (Evolution + communication)
- Strongest individual: Group A (Q-learning)

## Statistical Summary

| Group | Label | Mean | Median | Best | Std | Mean Best | Competence Gen |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Q-learning | 4522.975 | 4499.709 | 8474.750 | 294.411 | 7674.572 | 181.333 |
| B | Pure evolution | 1905.005 | 1916.334 | 5075.000 | 169.179 | 4647.358 | 0.033 |
| C | Evolution + communication | 1791.117 | 1825.956 | 5174.746 | 135.523 | 4574.194 | 0.000 |
| D | Evolution + communication + memory | 2076.296 | 2069.812 | 5575.000 | 140.177 | 4927.814 | 1.133 |
| E | Evolution + communication + memory + cultural inheritance | 1950.512 | 1959.937 | 5473.704 | 223.011 | 4823.995 | 0.633 |

## Research Questions

**Which approach performs best?** Group A has the highest mean final score.

**Which reaches competence fastest?** Group C reaches half of its peak score earliest on average.

**Which discovers the strongest individual?** Group A has the highest best observed score.

**Does cultural inheritance create cumulative progress?** Group E underperformed Group D by 125.784 mean score, so this run does not support a cultural benefit.

**Does knowledge persist across generations?** Check `cultural_memory_curves.png` and Group E's per-seed `seed_*_cultural_memory.csv` files. Persistent nonzero entries and increasing read/use counts indicate persistence.

## Output Files

- `statistical_summary.csv`
- `mean_score_curves.png`
- `best_score_curves.png`
- `cultural_memory_curves.png`
- per-seed group metrics under `group_*` directories
