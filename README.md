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

# 1. Define what you're optimizing
genes = GeneBuilder()
genes.add("threshold",  FloatRange(0.0, 1.0))
genes.add("lookback",   IntRange(3, 50))
genes.add("ma_type",    ChoiceList(["sma", "ema", "wma"]))

# 2. Define how to score a candidate
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

## Which class should I use?

| Situation | Use |
|---|---|
| One score to optimize (Sharpe, accuracy, MSE...) | `GeneticAlgorithm` |
| All genes are `FloatRange` and speed matters | `CMAESOptimizer` |
| Getting stuck in local optima; need more diversity | `IslandModel` |
| Two competing goals (e.g. return vs. drawdown) | `MultiObjectiveGA` |

---

## Gene Types

| Type | Description | Example |
|---|---|---|
| `FloatRange(low, high, sigma=0.1)` | Continuous float, Gaussian mutation | `FloatRange(0.0, 1.0)` |
| `IntRange(low, high, sigma=0.05)` | Integer, jump scaled to range width | `IntRange(3, 200)` |
| `ChoiceList(options)` | Categorical, always picks a different item | `ChoiceList(["sma", "ema"])` |

All gene types accept an optional `mutation_rate=` to override the global rate for that gene.
Individuals are plain Python `dict`s — no special classes, no magic.

---

## Key Features

**Genes & search space**
- **Named genes** — access by name: `ind["lookback"]` not `ind[2]`
- **Strict bounds** — mutation never produces values outside defined ranges, ever
- **Mixed types** — float, int, and categorical genes in the same individual
- **Per-gene mutation rate** — each gene can have its own rate independent of the global one

**Optimization control**
- **Maximize or minimize** — `mode='minimize'` for loss/error functions; no negation needed
- **Early stopping** — `patience` stops when score stagnates
- **Stagnation restart** — inject fresh individuals to escape local optima automatically
- **Adaptive mutation** — rate auto-adjusts: decreases on improvement, increases on stagnation
- **Reproducible** — `seed` gives identical results every run (seeds both `random` and `numpy.random`)

**Population architectures**
- **`GeneticAlgorithm`** — standard single-objective GA
- **`IslandModel`** — N isolated populations with periodic migration; better diversity
- **`MultiObjectiveGA`** — NSGA-II Pareto optimization; finds the full trade-off frontier

**Pluggable strategies**
- **Selection:** `RouletteSelection`, `TournamentSelection(k)`, `RankSelection`
- **Crossover:** `UniformCrossover`, `ArithmeticCrossover`, `SinglePointCrossover`

**Observability**
- **Generation history** — `best_score`, `avg_score`, `diversity`, `mutation_rate`, `restarted` per generation
- **Diversity metric** — tracks how spread-out the population is each generation (0.0–1.0)
- **Structured JSON logging** — machine- and AI-readable logs with convergence analysis
- **Callback hook** — `on_generation` for live plots, progress bars, custom logging
- **Checkpoint / resume** — save state, resume interrupted runs (essential for long backtests)

**Testing**
- **207 tests** across 4 test files
- **Property-based tests** via `hypothesis` — invariants verified against hundreds of random inputs

---

## Choosing selection and crossover strategies

**Selection** controls which individuals get to reproduce.

```python
from genetic_engine import TournamentSelection, RankSelection, RouletteSelection

# TournamentSelection — recommended for most problems
# k controls pressure: k=2 gentle, k=7+ aggressive
ga = GeneticAlgorithm(..., selection=TournamentSelection(k=4))

# RankSelection — when fitness values span wildly different magnitudes
ga = GeneticAlgorithm(..., selection=RankSelection())

# RouletteSelection — simple, default; can struggle with dominant individuals
ga = GeneticAlgorithm(..., selection=RouletteSelection())
```

**Crossover** controls how two parents produce a child.

```python
from genetic_engine import ArithmeticCrossover, SinglePointCrossover, UniformCrossover

# ArithmeticCrossover — blends float values between parents; best for continuous problems
ga = GeneticAlgorithm(..., crossover=ArithmeticCrossover())

# SinglePointCrossover — preserves co-dependent genes that go together
ga = GeneticAlgorithm(..., crossover=SinglePointCrossover())

# UniformCrossover — 50/50 per gene; general purpose default
ga = GeneticAlgorithm(..., crossover=UniformCrossover())
```

---

## Minimize Mode

Return the value to minimize directly — no negation needed:

```python
def fitness(ind):
    return mean_squared_error(ind["window"], ind["threshold"])

ga = GeneticAlgorithm(..., mode='minimize')
best, score, history = ga.run()
# score is 0.003, not -0.003
# history shows real values decreasing over generations
```

---

## Stagnation Restart

Automatically inject fresh individuals when the population gets stuck:

```python
ga = GeneticAlgorithm(
    ...,
    restart_after    = 20,   # after 20 gens without improvement
    restart_fraction = 0.3,  # replace 30% of the population
)
```

Elites are always preserved. `history[gen]['restarted']` records when it happened.

---

## Checkpoint / Resume

For long runs (stock backtests, large search spaces):

```python
ga = GeneticAlgorithm(..., checkpoint_path="checkpoint.json", checkpoint_every=10)
best, score, history = ga.run()

# Resume after crash / interruption:
best, score, history = ga.run(resume_from="checkpoint.json")
```

The returned history includes all generations — both pre-crash and resumed.

---

## Population Diversity

Every history entry includes a `diversity` metric (0.0 = fully converged, 1.0 = fully spread).
Use it to diagnose problems:

- **Collapses quickly** → mutation rate too low, or population too small
- **Stays high, no improvement** → mutation rate too high, random walk
- **Falls gradually while score improves** → healthy convergence

```python
_, _, history = ga.run()
for h in history:
    print(f"Gen {h['gen']:3d} | diversity={h['diversity']:.3f} | best={h['best_score']:.4f}")
```

---

## Island Model

Multiple independent populations exploring different regions, with periodic migration:

```python
from genetic_engine import IslandModel

im = IslandModel(
    gene_builder       = genes,
    fitness_function   = fitness,
    n_islands          = 4,          # 4 independent populations
    island_population  = 50,         # 50 individuals each (200 total)
    generations        = 100,
    migration_interval = 10,         # share top individuals every 10 gens
    migration_size     = 2,
    seed               = 42,
)

best, score, history = im.run()
# history[gen]['island_bests'] — best per island each generation
```

Islands evolve independently and periodically exchange their best individuals (ring topology).
This maintains diversity while still converging — better than one large population on complex problems.

---

## Multi-Objective Optimization

When you have competing goals that can't be collapsed into one score:

```python
from genetic_engine import MultiObjectiveGA

def fitness(ind: dict) -> list[float]:
    bt = run_backtest(**ind)
    return [bt.sharpe_ratio, bt.max_drawdown]

ga = MultiObjectiveGA(
    gene_builder     = genes,
    fitness_function = fitness,
    n_objectives     = 2,
    objectives       = ['maximize', 'minimize'],  # Sharpe up, drawdown down
    population_size  = 100,
    generations      = 50,
    seed             = 42,
)

pareto_front, history = ga.run()
# pareto_front: list of non-dominated solutions — pick your preferred trade-off
for solution in pareto_front:
    sharpe, drawdown = solution['scores']
    print(f"Sharpe={sharpe:.2f}  Drawdown={drawdown:.2f}  Params={solution['individual']}")
```

Returns the Pareto front — all solutions where improving one objective would require
worsening another. You choose the trade-off that fits your risk tolerance.

---

## CMA-ES: Faster Convergence on Float Problems

If all your genes are `FloatRange`, `CMAESOptimizer` is typically 10–100× faster than
a standard GA because it **learns the shape of the fitness landscape** rather than
searching blindly.

```python
from genetic_engine import CMAESOptimizer

opt = CMAESOptimizer(
    gene_builder     = genes,   # FloatRange genes only
    fitness_function = fitness,
    sigma0           = 0.3,     # initial step size (fraction of gene range)
    generations      = 200,
    patience         = 30,
    mode             = 'maximize',
    seed             = 42,
    log_path         = "cmaes_run.json",
)

best, score, history = opt.run()
# Same return shape as GeneticAlgorithm.run()
```

**Requires numpy** (`pip install numpy`). Raises `ValueError` at construction if any
gene is not `FloatRange`. Returns the same `(best_individual, best_score, history)`
tuple — the result is a plain dict, same as always.

See [features.md](features.md#cma-es-optimizer) for the full parameter reference,
sigma tuning guide, history format, and when to use CMA-ES vs GeneticAlgorithm.

---

## Per-Gene Mutation Rate

When genes have very different sensitivities:

```python
genes = GeneBuilder()
genes.add("fast_ma",  IntRange(3, 30, mutation_rate=0.5))     # explore aggressively
genes.add("slow_ma",  IntRange(20, 200))                       # use global rate
genes.add("ma_type",  ChoiceList(["sma", "ema"], mutation_rate=0.02))  # rarely change
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

Pass `log_path` to write a JSON log that AI agents (Claude, GPT, etc.) can read
to explain what happened and suggest what to tune:

```json
{
  "config": { "population_size": 100, "mutation_rate": 0.1, "mode": "maximize", ... },
  "genes":  { "threshold": {"type": "FloatRange", "low": 0, "high": 1}, ... },
  "result": { "best_score": 1.84, "early_stopped": true, "convergence_gen": 43 },
  "analysis": {
    "convergence_pattern": "converged_early",
    "notes": ["Algorithm converged well before the generation limit..."]
  },
  "history": [{"gen": 1, "best_score": 0.3, "diversity": 0.94, ...}, ...]
}
```

Paste the log to an AI agent with "explain what happened" — the `analysis` section
and `history` give it everything it needs to diagnose the run.

---

## Property-Based Tests

Beyond unit tests, the library includes property-based tests using
[hypothesis](https://hypothesis.readthedocs.io/) — a library that automatically generates
hundreds of random inputs to verify invariants that must hold for any valid input.

For example, instead of testing "does mutate(0.5) stay in [0, 1]?", hypothesis tests
"does mutate(x) stay in [low, high] for *any* valid low, high, and x?" — including
edge cases like very small ranges, negative values, and extreme sigmas.

```bash
pip install hypothesis
pytest tests/test_property.py
```

Invariants verified: gene bounds, mutation rate zero means no change, history keys always
present, best score never decreases, gen counter sequential, diversity in [0, 1].

---

## Why Not DEAP?

DEAP is the most widely known Python GA library but has well-documented problems:

- Values drift outside defined gene ranges (the original reason this library was built)
- Global mutable state (`creator.create`) breaks distributed computing and notebooks
- No named genes — individuals are anonymous lists
- No early stopping, no history, no structured logging, no callback API
- No minimize mode — always maximize, users must negate manually
- ~8000 lines, 30–50 lines of boilerplate before any EA logic
- 237 open issues, DEAP 2.0 stalled at ~27% since roughly 2014

See [deap_comparison.md](deap_comparison.md) for a full breakdown with GitHub issue references.

---

## Design Philosophy

See [PRINCIPLES.md](PRINCIPLES.md) for the vision and design principles behind the library —
including the strategic goal of making logs + documentation work together so an AI agent
can read a run log, diagnose what went wrong, and suggest parameter changes automatically.

---

## Full Documentation

See [features.md](features.md) for:
- Complete parameter reference for all classes
- Detailed strategy comparison and selection guide
- Tuning guide and troubleshooting checklist
- How to read the JSON log and convergence patterns
- Property-based testing explanation
- Custom gene type guide

---

## License

MIT
