# evogine

A clean, controllable evolutionary optimization library — named genes, strict bounds,
structured logging, and minimal boilerplate.

---

## Install

```bash
pip install evogine
```

Or from source:

```bash
git clone https://github.com/yourname/evogine
cd evogine
pip install -e .
```

---

## Quick Start

```python
from evogine import (
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
| All genes are `FloatRange`, mild multimodality | `DEOptimizer` |
| Getting stuck in local optima; need more diversity | `IslandModel` |
| Two competing goals (e.g. return vs. drawdown) | `MultiObjectiveGA` |
| Explore a diverse set of high-quality solutions | `MAPElites` |
| Not sure which optimizer to use | `landscape_analysis()` |

---

## Gene Types

| Type | Description | Example |
|---|---|---|
| `FloatRange(low, high, sigma=0.1)` | Continuous float, Gaussian or Levy mutation | `FloatRange(0.0, 1.0)` |
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
- **Levy flight mutation** — heavier-tailed exploration for `FloatRange` (`mutation_dist='levy'`)

**Optimization control**
- **Maximize or minimize** — `mode='minimize'` for loss/error functions; no negation needed
- **Early stopping** — `patience` stops when score stagnates
- **Stagnation restart** — inject fresh individuals to escape local optima automatically
- **Adaptive mutation** — rate auto-adjusts: decreases on improvement, increases on stagnation
- **Constraint handling** — enforce feasibility rules; infeasible individuals never win
- **Linear population reduction** — shrink population over time (L-SHADE style)
- **Reproducible** — `seed` gives identical results every run (seeds both `random` and `numpy.random`)

**Population architectures**
- **`GeneticAlgorithm`** — standard single-objective GA
- **`IslandModel`** — N isolated populations with periodic migration; ring, fully-connected, or star topology
- **`MultiObjectiveGA`** — NSGA-II or NSGA-III Pareto optimization; finds the full trade-off frontier
- **`CMAESOptimizer`** — covariance matrix adaptation; fastest on smooth float-only landscapes
- **`DEOptimizer`** — SHADE differential evolution with adaptive F/CR; great for continuous search spaces
- **`MAPElites`** — quality-diversity archive; finds the best solution for every behavioral niche

**Pluggable strategies**
- **Selection:** `RouletteSelection`, `TournamentSelection(k)`, `RankSelection`
- **Crossover:** `UniformCrossover`, `ArithmeticCrossover`, `SinglePointCrossover`, `LLMCrossover`

**Landscape analysis**
- **`landscape_analysis()`** — samples fitness landscape, measures ruggedness/neutrality/modes,
  recommends the best optimizer for the problem

**Observability**
- **Generation history** — `best_score`, `avg_score`, `diversity`, `mutation_rate`, `restarted` per generation
- **Diversity metric** — tracks how spread-out the population is each generation (0.0–1.0)
- **Structured JSON logging** — machine- and AI-readable logs with convergence analysis
- **Callback hook** — `on_generation` for live plots, progress bars, custom logging
- **Checkpoint / resume** — save state, resume interrupted runs (essential for long backtests)

**Testing**
- **376 tests** across 8 test files
- **Property-based tests** via `hypothesis` — invariants verified against hundreds of random inputs

---

## Levy Flight Mutation

For problems where the landscape has long-range structure, Levy flight mutations take
larger exploratory jumps than Gaussian, helping escape local optima:

```python
genes = GeneBuilder()
genes.add("x", FloatRange(0.0, 1.0, mutation_dist='levy'))
genes.add("y", FloatRange(-5.0, 5.0, mutation_dist='levy'))

# All other GA/DE/CMA-ES usage is identical
ga = GeneticAlgorithm(genes, fitness, ...)
```

Default is `'gaussian'`. No external dependencies — uses a pure Python
Chambers-Mallows-Stuck approximation.

---

## Constraint Handling

Enforce hard constraints on individuals. Infeasible individuals are always ranked below
any feasible individual, regardless of score (Deb's feasibility rules):

```python
ga = GeneticAlgorithm(
    genes, fitness,
    constraints=[
        lambda ind: ind['fast_ma'] < ind['slow_ma'],   # crossover requires fast < slow
        lambda ind: ind['stop_loss'] < ind['take_profit'],
    ],
)
```

Each constraint is `fn(individual) -> bool`. Multiple violations stack — fewer violations
ranks higher among infeasible individuals.

---

## Choosing selection and crossover strategies

**Selection** controls which individuals get to reproduce.

```python
from evogine import TournamentSelection, RankSelection, RouletteSelection

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
from evogine import ArithmeticCrossover, SinglePointCrossover, UniformCrossover, LLMCrossover

# ArithmeticCrossover — blends float values between parents; best for continuous problems
ga = GeneticAlgorithm(..., crossover=ArithmeticCrossover())

# SinglePointCrossover — preserves co-dependent genes that go together
ga = GeneticAlgorithm(..., crossover=SinglePointCrossover())

# UniformCrossover — 50/50 per gene; general purpose default
ga = GeneticAlgorithm(..., crossover=UniformCrossover())

# LLMCrossover — delegate crossover logic to an LLM or custom function
ga = GeneticAlgorithm(..., crossover=LLMCrossover(llm_fn=my_api_call))
```

---

## LLM Crossover

Let an LLM (or any custom function) combine two parent individuals:

```python
from evogine import LLMCrossover

def my_llm_fn(parent1: dict, parent2: dict) -> dict:
    # call your LLM, return a child dict
    response = call_claude(f"Combine these two strategies: {parent1}, {parent2}")
    return parse_response(response)

ga = GeneticAlgorithm(
    genes, fitness,
    crossover=LLMCrossover(my_llm_fn, raise_on_failure=False),
)
```

- If the function returns an invalid result, it silently falls back to `UniformCrossover`
  and increments `crossover.fallback_count`
- Set `raise_on_failure=True` to surface errors immediately instead

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

## Linear Population Reduction

Shrink the population size linearly over generations (L-SHADE style). Starts with
a large diverse population and ends with a focused, small one:

```python
ga = GeneticAlgorithm(
    genes, fitness,
    population_size      = 100,
    generations          = 200,
    linear_pop_reduction = True,
    min_population       = 4,    # never shrinks below this
)
```

Also available in `DEOptimizer` as the full L-SHADE variant.

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

Multiple independent populations with periodic migration. Supports three topologies:

```python
from evogine import IslandModel

im = IslandModel(
    gene_builder       = genes,
    fitness_function   = fitness,
    n_islands          = 4,
    island_population  = 50,
    generations        = 100,
    migration_interval = 10,
    migration_size     = 2,
    topology           = 'ring',          # 'ring' | 'fully_connected' | 'star'
    seed               = 42,
)

best, score, history = im.run()
```

| Topology | Migration pattern |
|---|---|
| `'ring'` (default) | Each island sends to the next; good balance of diversity and speed |
| `'fully_connected'` | Every island sends to every other; maximum information sharing |
| `'star'` | All islands send to a hub island which redistributes; centralized |

---

## Multi-Objective Optimization

### NSGA-II (default)

```python
from evogine import MultiObjectiveGA

def fitness(ind: dict) -> list[float]:
    bt = run_backtest(**ind)
    return [bt.sharpe_ratio, bt.max_drawdown]

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
for solution in pareto_front:
    sharpe, drawdown = solution['scores']
    print(f"Sharpe={sharpe:.2f}  Drawdown={drawdown:.2f}  Params={solution['individual']}")
```

### NSGA-III (for 3+ objectives)

For three or more competing objectives, NSGA-III uses reference-point niching instead of
crowding distance, which maintains better diversity across a high-dimensional Pareto front:

```python
ga = MultiObjectiveGA(
    genes, fitness,
    n_objectives              = 4,
    objectives                = ['maximize'] * 4,
    algorithm                 = 'nsga3',
    reference_point_divisions = 3,   # Das-Dennis lattice divisions per objective
    population_size           = 100,
    generations               = 100,
)
```

---

## DEOptimizer (Differential Evolution)

SHADE variant with adaptive F and CR parameters. Works on `FloatRange` genes only,
requires at least 2 genes. Often outperforms a standard GA on continuous search spaces:

```python
from evogine import DEOptimizer

de = DEOptimizer(
    gene_builder     = genes,   # FloatRange only
    fitness_function = fitness,
    population_size  = 50,
    generations      = 200,
    strategy         = 'current_to_best',  # or 'rand1'
    memory_size      = 6,                  # SHADE history memory size
    linear_pop_reduction = False,          # set True for L-SHADE variant
    mode             = 'maximize',
    patience         = 30,
    seed             = 42,
    log_path         = "de_run.json",
)

best, score, history = de.run()
# history includes F_mean, CR_mean, pop_size per generation
```

| Key | Description |
|---|---|
| `strategy='current_to_best'` | Biases mutation toward current best; faster convergence |
| `strategy='rand1'` | Fully random base vector; more exploration |
| `memory_size` | SHADE archive size; larger = more stable adaptation |
| `linear_pop_reduction=True` | L-SHADE: shrinks from `population_size` to 4 over generations |

---

## CMA-ES: Fastest on Float Problems

If all your genes are `FloatRange`, `CMAESOptimizer` typically converges faster than a
standard GA or DE because it learns the shape of the fitness landscape:

```python
from evogine import CMAESOptimizer

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
```

Raises `ValueError` at construction if any gene is not `FloatRange`.

---

## MAPElites: Quality-Diversity Optimization

Instead of finding one best solution, MAPElites fills a behavioral grid with the best
solution found in each niche. Useful when you want to understand the trade-off between
performance and behavior across a whole space:

```python
from evogine import MAPElites

genes = GeneBuilder()
genes.add('x', FloatRange(0.0, 1.0))
genes.add('y', FloatRange(0.0, 1.0))

def fitness(ind):
    return -(ind['x']**2 + ind['y']**2)   # maximize toward origin

def behavior(ind):
    return (ind['x'], ind['y'])            # 2D behavior space; values should be in [0, 1]

me = MAPElites(
    gene_builder       = genes,
    fitness_function   = fitness,
    behavior_fn        = behavior,
    grid_shape         = (20, 20),         # 20×20 grid = 400 cells
    initial_population = 200,             # random seeding phase
    generations        = 1000,
    seed               = 42,
)

archive, history = me.run()
# archive: {(i, j): {'individual': dict, 'score': float, 'behavior': tuple}}
# history: [{gen, archive_size, best_score, coverage}, ...]
print(f"Filled {history[-1]['coverage']*100:.1f}% of the grid")
```

`behavior_fn` maps an individual to a tuple of coordinates, each ideally in `[0, 1]`.
The grid discretizes those coordinates into cells. Each cell keeps only its best-scoring
individual. Over time the archive fills, giving you a map of best-in-class solutions
across the whole behavioral space.

---

## Landscape Analysis

Not sure which optimizer to use? Sample the fitness landscape first:

```python
from evogine import landscape_analysis

report = landscape_analysis(
    gene_builder     = genes,
    fitness_function = fitness,
    n_samples        = 500,
    seed             = 42,
)

print(report['recommendation'])   # e.g. 'DEOptimizer'
print(report['reason'])           # plain English explanation
print(f"Ruggedness: {report['ruggedness']:.3f}")
print(f"Neutrality: {report['neutrality']:.3f}")
print(f"Estimated modes: {report['estimated_modes']}")
```

| Key | Description |
|---|---|
| `ruggedness` | 0 = smooth, 1 = maximally jagged |
| `neutrality` | Fraction of neighbor pairs with nearly identical fitness |
| `estimated_modes` | Estimated number of local optima |
| `float_only` | True if all genes are FloatRange |
| `recommendation` | `'CMAESOptimizer'`, `'DEOptimizer'`, `'IslandModel'`, or `'GeneticAlgorithm'` |
| `reason` | Plain English explanation of the recommendation |
| `sample_best` | Best score found during sampling |
| `sample_best_individual` | Individual that achieved `sample_best` |

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
[hypothesis](https://hypothesis.readthedocs.io/) — automatically generates hundreds of
random inputs to verify invariants that must hold for any valid input.

```bash
pip install hypothesis
pytest tests/test_property.py
```

Invariants verified: gene bounds, mutation rate zero means no change, history keys always
present, best score never decreases, gen counter sequential, diversity in [0, 1].

---

## Examples

See the `examples/` directory:

- **`stock_strategy_optimization.md`** — end-to-end guide for optimizing trading strategy
  parameters against a backtesting engine, including multi-objective trade-offs and
  overfitting prevention

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
