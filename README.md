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
from genetic_engine import (
    GeneticAlgorithm, GeneBuilder,
    FloatRange, IntRange, ChoiceList,
)

# 1. Define genes
genes = GeneBuilder()
genes.add("threshold",  FloatRange(0.0, 1.0))
genes.add("lookback",   IntRange(3, 50))
genes.add("ma_type",    ChoiceList(["sma", "ema", "wma"]))

# 2. Define fitness function
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
    patience         = 20,
    log_path         = "run.json",
)

best, score, history = ga.run()
print(best)   # {'threshold': 0.42, 'lookback': 14, 'ma_type': 'ema'}
```

---

## Gene Types

| Type | Description | Example |
|---|---|---|
| `FloatRange(low, high, sigma=0.1)` | Continuous float, Gaussian mutation | `FloatRange(0.0, 1.0)` |
| `IntRange(low, high, sigma=0.05)` | Integer, jump scaled to range width | `IntRange(3, 200)` |
| `ChoiceList(options)` | Categorical, picks a different item | `ChoiceList(["sma", "ema"])` |

All gene types accept an optional `mutation_rate=` argument to override the global rate.
Individuals are plain Python `dict`s — no special classes, no magic.

---

## Key Features

**Genes & search space**
- **Named genes** — access by name: `ind["lookback"]` not `ind[2]`
- **Strict bounds** — mutation never produces values outside defined ranges
- **Mixed types** — float, int, and categorical genes in the same individual
- **Per-gene mutation rate** — each gene can override the global `mutation_rate`

**Optimization control**
- **Maximize or minimize** — `mode='minimize'` for loss/error functions, no negation needed
- **Early stopping** — `patience` stops the run when score stagnates
- **Stagnation restart** — inject fresh individuals to escape local optima
- **Adaptive mutation** — auto-tune mutation rate based on progress
- **Reproducible** — `seed` gives identical results every run (seeds both `random` and `numpy.random`)

**Multiple population architectures**
- **Single GA** — `GeneticAlgorithm` for standard single-objective problems
- **Island model** — `IslandModel` for multiple diverse sub-populations with migration
- **Multi-objective** — `MultiObjectiveGA` for Pareto-front optimization (NSGA-II style)

**Pluggable strategies**
- **Selection:** `RouletteSelection`, `TournamentSelection(k)`, `RankSelection`
- **Crossover:** `UniformCrossover`, `ArithmeticCrossover`, `SinglePointCrossover`

**Observability**
- **Generation history** — `best_score`, `avg_score`, `diversity`, `mutation_rate`, `restarted` per generation
- **Population diversity metric** — tracks how spread out the population is (0.0–1.0) each generation
- **Structured JSON logging** — machine- and AI-readable run logs with convergence analysis
- **Callback hook** — `on_generation(gen, best_score, avg_score, best_individual)` for live plots
- **Checkpoint / resume** — save state to disk, resume interrupted runs

**Testing**
- **207 tests** covering every feature
- **Property-based tests** via `hypothesis` — invariants hold for any valid input

---

## Pluggable Selection Strategies

```python
from genetic_engine import TournamentSelection, RankSelection, RouletteSelection

# Tournament — robust, no score normalization needed (recommended default)
ga = GeneticAlgorithm(..., selection=TournamentSelection(k=4))

# Rank-based — steady pressure, prevents one individual dominating
ga = GeneticAlgorithm(..., selection=RankSelection())

# Roulette — fitness-proportionate (default if not specified)
ga = GeneticAlgorithm(..., selection=RouletteSelection())
```

---

## Pluggable Crossover Strategies

```python
from genetic_engine import ArithmeticCrossover, SinglePointCrossover, UniformCrossover

# Arithmetic — blends float genes between parents (best for continuous problems)
ga = GeneticAlgorithm(..., crossover=ArithmeticCrossover())

# Single-point — split index, preserves gene co-dependencies
ga = GeneticAlgorithm(..., crossover=SinglePointCrossover())

# Uniform — 50/50 per gene (default if not specified)
ga = GeneticAlgorithm(..., crossover=UniformCrossover())
```

---

## Minimize Mode

Pass `mode='minimize'` when optimizing loss functions, error rates, or drawdown.
No need to negate your fitness function:

```python
def fitness(ind):
    return mean_squared_error(ind["window"], ind["threshold"])  # just return the error

ga = GeneticAlgorithm(..., mode='minimize')
best, score, history = ga.run()
# score is 0.003, not -0.003
# history shows real error values decreasing over time
```

---

## Stagnation Restart

When stuck in a local optimum, inject fresh random individuals:

```python
ga = GeneticAlgorithm(
    ...,
    restart_after    = 20,   # inject after 20 gens without improvement
    restart_fraction = 0.3,  # replace 30% of population
)
```

Elites are preserved. Restarts fire every `restart_after` stagnant generations.
`history[gen]['restarted']` records when it fired.

---

## Checkpoint / Resume

For long runs (e.g. stock backtests across many tickers):

```python
ga = GeneticAlgorithm(
    ...,
    checkpoint_path  = "checkpoint.json",
    checkpoint_every = 10,
)
best, score, history = ga.run()

# Later — resume after crash or interruption:
best, score, history = ga.run(resume_from="checkpoint.json")
```

---

## Population Diversity

Every generation history entry includes a `diversity` metric (0.0 = fully converged, 1.0 = fully spread):

```python
_, _, history = ga.run()
for h in history:
    print(f"Gen {h['gen']:3d} | diversity={h['diversity']:.3f} | best={h['best_score']:.4f}")
```

Use this to diagnose premature convergence and decide when to increase `mutation_rate`
or enable `restart_after`.

---

## Island Model

Multiple independent populations with migration — better diversity and escape from local optima:

```python
from genetic_engine import IslandModel

im = IslandModel(
    gene_builder       = genes,
    fitness_function   = fitness,
    n_islands          = 4,
    island_population  = 50,
    generations        = 100,
    migration_interval = 10,   # exchange top individuals every 10 gens
    migration_size     = 2,
    seed               = 42,
)

best, score, history = im.run()
# history[gen]['island_bests'] — best score on each island that generation
```

---

## Multi-Objective Optimization

When you have multiple conflicting objectives (e.g. Sharpe ratio vs. max drawdown):

```python
from genetic_engine import MultiObjectiveGA

def fitness(ind: dict) -> list[float]:
    bt = run_backtest(**ind)
    return [bt.sharpe_ratio, bt.max_drawdown]  # maximize sharpe, minimize drawdown

ga = MultiObjectiveGA(
    gene_builder     = genes,
    fitness_function = fitness,
    n_objectives     = 2,
    objectives       = ['maximize', 'minimize'],
    population_size  = 100,
    generations      = 50,
    seed             = 42,
)

pareto_front, history = ga.run()
# pareto_front: [{'individual': {...}, 'scores': [sharpe, drawdown]}, ...]
# Pick your preferred trade-off from the non-dominated solutions
```

Uses NSGA-II style Pareto ranking + crowding distance for diverse Pareto fronts.

---

## Per-Gene Mutation Rate

Each gene can override the global mutation rate:

```python
genes = GeneBuilder()
genes.add("fast_ma",  IntRange(3, 30, mutation_rate=0.5))   # mutates aggressively
genes.add("slow_ma",  IntRange(20, 200, mutation_rate=0.1)) # standard rate
genes.add("ma_type",  ChoiceList(["sma", "ema"], mutation_rate=0.02)) # rarely changes
```

---

## Live Plot with Callback

```python
import matplotlib.pyplot as plt

gens, bests, avgs = [], [], []
fig, ax = plt.subplots()
line_best, = ax.plot([], [], label='best')
line_avg,  = ax.plot([], [], label='avg')
ax.legend()

def on_gen(gen, best_score, avg_score, best_ind):
    gens.append(gen)
    bests.append(best_score)
    avgs.append(avg_score)
    line_best.set_data(gens, bests)
    line_avg.set_data(gens, avgs)
    ax.relim(); ax.autoscale_view()
    plt.pause(0.01)

ga = GeneticAlgorithm(..., on_generation=on_gen)
ga.run()
plt.show()
```

See `examples/live_plot.py` for a full two-panel live visualization.

---

## Structured Logs for AI Analysis

Pass `log_path` to write a JSON log that AI agents (Claude, GPT, etc.) can read directly
to explain what happened and what to tune:

```json
{
  "config": { "population_size": 100, "mutation_rate": 0.1, "mode": "minimize", ... },
  "genes":  { "threshold": {"type": "FloatRange", "low": 0, "high": 1}, ... },
  "result": { "best_score": 0.003, "early_stopped": true, "convergence_gen": 23 },
  "analysis": {
    "convergence_pattern": "converged_early",
    "notes": ["Algorithm converged well before the generation limit..."]
  },
  "history": [{"gen": 1, "best_score": 0.12, "diversity": 0.94, ...}, ...]
}
```

---

## Why Not DEAP?

DEAP is the most widely known Python GA library but has well-documented problems:

- Values drift outside defined gene ranges
- Global mutable state (`creator.create`) breaks distributed computing and notebooks
- No named genes — individuals are anonymous lists
- No early stopping, no history, no structured logging, no callback API
- No minimize mode — always maximize, users must negate manually
- ~8000 lines, 30–50 lines of boilerplate before any EA logic
- 237 open issues, DEAP 2.0 stalled at ~27% since roughly 2014

See [deap_comparison.md](deap_comparison.md) for a full breakdown with GitHub issue references.

---

## Full Documentation

See [features.md](features.md) for the complete parameter reference and examples.

---

## Roadmap

See [ideas.md](ideas.md) for ideas under consideration:
- CMA-ES hybrid (optimal for pure float search spaces)
- GitHub Actions CI

---

## License

MIT
