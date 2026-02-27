# genetic-engine

A clean, controllable genetic algorithm library for Python.

Built because existing libraries (DEAP et al.) are hard to control, leak values outside defined
gene ranges, and require excessive boilerplate. This library stays small, readable, and predictable.

---

## Install

```bash
pip install genetic-engine
```

Or from source:

```bash
git clone https://github.com/yourname/genetic-engine
cd genetic-engine
pip install -e .
```

---

## Quick Start

```python
from genetic_engine import GeneticAlgorithm, GeneBuilder, FloatRange, IntRange, ChoiceList

# 1. Define genes
genes = GeneBuilder()
genes.add("threshold",  FloatRange(0.0, 1.0))
genes.add("lookback",   IntRange(3, 50))
genes.add("ma_type",    ChoiceList(["sma", "ema", "wma"]))

# 2. Define fitness function — return a float, higher = better
def fitness(individual: dict) -> float:
    return run_my_strategy(
        threshold = individual["threshold"],
        lookback  = individual["lookback"],
        ma_type   = individual["ma_type"],
    )

# 3. Run
ga = GeneticAlgorithm(
    gene_builder     = genes,
    fitness_function = fitness,
    population_size  = 100,
    generations      = 100,
    seed             = 42,
    patience         = 20,        # stop early if no improvement
    log_path         = "run.json" # optional structured log
)

best, score, history = ga.run()
print(best)   # {'threshold': 0.42, 'lookback': 14, 'ma_type': 'ema'}
```

---

## Gene Types

| Type | Description | Example |
|---|---|---|
| `FloatRange(low, high, sigma)` | Continuous float, Gaussian mutation | `FloatRange(0.0, 1.0)` |
| `IntRange(low, high)` | Integer, ±step mutation | `IntRange(3, 50)` |
| `ChoiceList(options)` | Categorical, picks different item | `ChoiceList(["sma", "ema"])` |

Individuals are plain Python `dict`s — no special classes, no magic.

---

## Key Features

- **Named genes** — access values by name, not index: `ind["lookback"]` not `ind[2]`
- **Strict bounds** — mutation never produces values outside defined ranges
- **Early stopping** — `patience` parameter stops the run when score stagnates
- **Generation history** — full `best_score` / `avg_score` per generation, always returned
- **Structured JSON logging** — machine- and AI-readable run logs with convergence analysis
- **Reproducible** — `seed` parameter gives identical runs every time
- **Multiprocessing** — `use_multiprocessing=True` parallelises fitness evaluation across all cores
- **Extensible** — add custom gene types by subclassing `GeneSpec`

---

## Why Not DEAP?

DEAP is the most widely known Python GA library, but has well-documented problems:

- Values drift outside defined gene ranges
- Global mutable state (`creator.create`) breaks distributed computing
- No named genes — individuals are anonymous lists
- No early stopping, no history, no structured logging
- ~8000 lines, complex API, 30–50 lines of boilerplate before any EA logic
- 237 open issues, DEAP 2.0 stalled at ~27% since roughly 2014

See [deap_comparison.md](deap_comparison.md) for a full breakdown with GitHub issue references.

---

## Structured Logs for AI Analysis

Pass `log_path` to write a JSON log that AI agents (Claude, GPT, etc.) can read directly
to explain what happened and what to tune:

```json
{
  "config": { "population_size": 100, "mutation_rate": 0.1, ... },
  "genes":  { "threshold": {"type": "FloatRange", "low": 0, "high": 1}, ... },
  "result": { "best_score": 1.87, "early_stopped": true, "convergence_gen": 23 },
  "analysis": {
    "convergence_pattern": "converged_early",
    "notes": ["Algorithm converged well before the generation limit. If the result is good, parameters are well-tuned. If not, increase mutation_rate to escape local optima."]
  },
  "history": [...]
}
```

---

## Full Documentation

See [features.md](features.md) for the complete parameter reference and examples.

---

## Roadmap

See [ideas.md](ideas.md) for planned features including:
- Tournament and rank-based selection
- Arithmetic crossover for floats
- Adaptive mutation rate
- Callback hooks
- Island model (parallel populations)
- Multi-objective / Pareto support

---

## License

MIT
