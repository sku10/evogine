# Genetic Engine — Features Reference

Complete parameter reference and examples for every feature.

---

## Gene Types

### `FloatRange(low, high, sigma=0.1)`
Continuous float gene. Mutation adds Gaussian noise scaled to the range width.
`sigma` controls mutation aggressiveness: `0.1` = 10% of range width per step.

```python
from genetic_engine import FloatRange

genes.add("threshold", FloatRange(0.0, 1.0))            # default sigma
genes.add("weight",    FloatRange(-5.0, 5.0, sigma=0.05)) # finer steps
genes.add("factor",    FloatRange(0.0, 100.0, sigma=0.2)) # wider jumps
```

### `IntRange(low, high, sigma=0.05)`
Integer gene. Jump size scales with range width: `jump = max(1, round(sigma * (high - low)))`.

```python
from genetic_engine import IntRange

genes.add("lookback",  IntRange(3, 50))           # jump up to ±2
genes.add("ma_period", IntRange(5, 200))          # jump up to ±10
genes.add("window",    IntRange(0, 500, sigma=0.02)) # finer control on wide range
```

Default `sigma=0.05` (5% of range). Use smaller sigma for fine-tuning, larger for wide exploration.
Previously this was hardcoded to ±1, which made wide integer ranges impractically slow to explore.

### `ChoiceList(options)`
Categorical gene. Mutation always picks a *different* item from the list.
Safe with single-item lists — returns the same value, no crash.

```python
from genetic_engine import ChoiceList

genes.add("ma_type",   ChoiceList(["sma", "ema", "wma"]))
genes.add("fib_level", ChoiceList([3, 5, 8, 13, 21, 34]))
genes.add("enabled",   ChoiceList([True, False]))
```

---

## GeneBuilder

Composes named genes into a genome. Individuals are plain Python `dict`s.

```python
from genetic_engine import GeneBuilder, FloatRange, IntRange, ChoiceList

genes = GeneBuilder()
genes.add("entry_threshold", FloatRange(0.0, 1.0))
genes.add("exit_window",     IntRange(3, 30))
genes.add("signal_type",     ChoiceList(["rsi", "macd", "bb"]))

ind = genes.sample()
# → {'entry_threshold': 0.423, 'exit_window': 17, 'signal_type': 'rsi'}
```

Custom gene types: subclass `GeneSpec` and implement `sample()`, `mutate()`, `describe()`.

---

## GeneticAlgorithm — Full Parameter Reference

```python
from genetic_engine import GeneticAlgorithm

ga = GeneticAlgorithm(
    gene_builder        = genes,         # GeneBuilder (required)
    fitness_function    = my_fitness,    # fn(dict) -> float (required)
    population_size     = 100,           # individuals per generation
    generations         = 50,            # maximum generations
    mutation_rate       = 0.1,           # probability per gene of mutating
    crossover_rate      = 0.5,           # probability of crossover vs. cloning
    elitism             = 2,             # top N carried unchanged each generation
    use_multiprocessing = False,         # parallel fitness via multiprocessing.Pool
    seed                = 42,            # random seed for reproducibility
    patience            = 20,            # early stop after N gens with no improvement
    min_delta           = 1e-6,          # minimum score change to count as improvement
    log_path            = "run.json",    # write structured JSON log (optional)
    selection           = None,          # SelectionStrategy (default: RouletteSelection)
    crossover           = None,          # CrossoverStrategy (default: UniformCrossover)
    on_generation       = None,          # callback fn(gen, best_score, avg_score, best_ind)
    mode                = 'maximize',    # 'maximize' or 'minimize'
)
```

### Running

```python
best_individual, best_score, history = ga.run()
```

Returns:
- `best_individual` — dict of gene name → value for the best solution found
- `best_score` — fitness score of that individual (real value in user's units)
- `history` — list of dicts, one per generation

---

## Fitness Function

Must accept a `dict` of gene values and return a single `float`.

```python
def fitness(individual: dict) -> float:
    return run_backtest(
        ma_period = individual["ma_period"],
        threshold = individual["entry_threshold"],
    ).sharpe_ratio
```

Use `mode='maximize'` (default) when higher = better.
Use `mode='minimize'` when lower = better — see Minimize Mode below.

---

## Minimize Mode

Return the natural value to minimize — no negation needed:

```python
def fitness(ind):
    return mean_squared_error(ind["window"], ind["threshold"])

ga = GeneticAlgorithm(..., mode='minimize')
best, score, history = ga.run()
# score is the real error value e.g. 0.003, not -0.003
# history shows real values decreasing toward the minimum
```

Works with all features: early stopping, logging, callback, history all show real values.
Invalid mode raises `ValueError`.

---

## Selection Strategies

Pass via `selection=` parameter. Controls how parents are chosen each generation.

### `RouletteSelection()` — default
Fitness-proportionate selection. Higher fitness = higher chance of being chosen.
Scores are shifted so even the worst individual has a nonzero chance.

```python
from genetic_engine import RouletteSelection
ga = GeneticAlgorithm(..., selection=RouletteSelection())
```

**When to use:** Simple problems. Can struggle when one individual dominates early.

### `TournamentSelection(k=3)`
Randomly picks `k` individuals and returns the best. Repeat for each parent.
No score normalization needed — works natively with negative fitness.

```python
from genetic_engine import TournamentSelection
ga = GeneticAlgorithm(..., selection=TournamentSelection(k=4))
```

**`k` controls selection pressure:** `k=2` is gentle, `k=7+` is aggressive.
**When to use:** Recommended for most problems. Robust and predictable.

### `RankSelection()`
Assigns weights by rank position (best = N, worst = 1) rather than raw score.
Prevents one superstar dominating when fitness values vary wildly in magnitude.

```python
from genetic_engine import RankSelection
ga = GeneticAlgorithm(..., selection=RankSelection())
```

**When to use:** When fitness values span very different magnitudes across generations.

---

## Crossover Strategies

Pass via `crossover=` parameter. Controls how two parents produce a child.

### `UniformCrossover()` — default
Each gene independently 50/50 from either parent.

```python
from genetic_engine import UniformCrossover
ga = GeneticAlgorithm(..., crossover=UniformCrossover())
```

**When to use:** General purpose. Fast and simple.

### `ArithmeticCrossover()`
Blends float genes: `child = t * p1 + (1-t) * p2` where `t` is random in [0,1].
Non-float genes (IntRange, ChoiceList) fall back to uniform selection.

```python
from genetic_engine import ArithmeticCrossover
ga = GeneticAlgorithm(..., crossover=ArithmeticCrossover())
```

**When to use:** Continuous float-heavy problems. Explores the space between parents
rather than making hard jumps.

### `SinglePointCrossover()`
Picks a random split index; genes before it come from p1, genes after from p2.
Preserves gene co-dependencies — genes that work well together stay together.

```python
from genetic_engine import SinglePointCrossover
ga = GeneticAlgorithm(..., crossover=SinglePointCrossover())
```

**When to use:** When gene order is meaningful (e.g. a sequence of thresholds that
build on each other).

---

## Early Stopping

```python
ga = GeneticAlgorithm(
    ...,
    patience  = 20,     # stop after 20 gens without improvement
    min_delta = 1e-6,   # improvement must exceed this to count
)
```

`patience=None` (default) disables early stopping — always runs all `generations`.

---

## Generation History

Always returned as the third value from `run()`:

```python
best_ind, best_score, history = ga.run()
```

Each entry:
```python
{
    'gen':                      int,    # generation number (1-based)
    'best_score':               float,  # best score this generation (real value)
    'avg_score':                float,  # population average (real value)
    'improved':                 bool,   # did best_score improve vs. previous best?
    'gens_without_improvement': int,    # consecutive gens with no improvement
}
```

Plot convergence:
```python
import matplotlib.pyplot as plt
gens  = [h['gen'] for h in history]
bests = [h['best_score'] for h in history]
avgs  = [h['avg_score'] for h in history]
plt.plot(gens, bests, label='best')
plt.plot(gens, avgs,  label='avg')
plt.legend(); plt.show()
```

---

## Callback Hook

Called after every generation. Use for live plots, progress bars, custom logging.

```python
def on_gen(gen: int, best_score: float, avg_score: float, best_individual: dict):
    print(f"Gen {gen:04d} | best={best_score:.4f} | avg={avg_score:.4f}")

ga = GeneticAlgorithm(..., on_generation=on_gen)
```

Scores passed to the callback are always real values (un-negated in minimize mode).
The callback fires on every generation including the final one before early stopping.

Live plot example — see `examples/live_plot.py`.

---

## Structured JSON Logging

```python
ga = GeneticAlgorithm(..., log_path="logs/run_001.json")
```

Log structure:
- **`run`** — timestamp, elapsed seconds
- **`config`** — all parameters: population size, rates, mode, selection/crossover strategy names
- **`genes`** — full gene definitions (type, ranges, options, sigma)
- **`result`** — best individual, best score (real value), early_stopped, convergence_gen
- **`analysis`** — pre-computed AI-readable observations:
  - `convergence_pattern` — one of: `converged_early`, `converged_midway`, `still_improving`, `no_progress_after_gen1`, `converged_at_end`, `too_short_to_assess`
  - `notes` — plain-English notes on what happened and what to adjust
- **`history`** — full per-generation data

AI agents (Claude, GPT, etc.) can read this log and explain whether the run was
successful, whether parameters need tuning, and what to try next.

---

## Reproducibility

```python
ga = GeneticAlgorithm(..., seed=42)
```

Seed is applied at the start of `run()`, not at construction. This means:
- Calling `ga.run()` twice produces identical results
- Creating two instances with the same seed produces identical results regardless of construction order

---

## Multiprocessing

```python
ga = GeneticAlgorithm(..., use_multiprocessing=True)
```

Uses all available CPU cores via `multiprocessing.Pool`.

**Requirements:** Fitness function must be defined at module level (not a lambda or
nested function) to be picklable across processes.

---

## Custom Gene Types

Subclass `GeneSpec` and implement three methods:

```python
from genetic_engine import GeneSpec

class LogRange(GeneSpec):
    """Float gene sampled on a log scale — useful for learning rates, etc."""
    def __init__(self, low: float, high: float):
        import math
        self.low = math.log10(low)
        self.high = math.log10(high)

    def sample(self):
        import random, math
        return 10 ** random.uniform(self.low, self.high)

    def mutate(self, value, mutation_rate):
        import random, math
        if random.random() < mutation_rate:
            log_val = math.log10(value) + random.gauss(0, 0.1 * (self.high - self.low))
            log_val = max(min(log_val, self.high), self.low)
            return 10 ** log_val
        return value

    def describe(self):
        return {'type': 'LogRange', 'low': 10**self.low, 'high': 10**self.high}

genes.add("learning_rate", LogRange(1e-5, 1e-1))
```

---

## Complete Example

```python
from genetic_engine import (
    GeneticAlgorithm, GeneBuilder,
    FloatRange, IntRange, ChoiceList,
    TournamentSelection, ArithmeticCrossover,
)

genes = GeneBuilder()
genes.add("fast_ma",   IntRange(3, 30))
genes.add("slow_ma",   IntRange(20, 200))
genes.add("threshold", FloatRange(0.0, 1.0))
genes.add("ma_type",   ChoiceList(["sma", "ema", "wma"]))

def fitness(ind: dict) -> float:
    return run_backtest(**ind).sharpe_ratio

ga = GeneticAlgorithm(
    gene_builder        = genes,
    fitness_function    = fitness,
    population_size     = 100,
    generations         = 200,
    mutation_rate       = 0.15,
    crossover_rate      = 0.7,
    elitism             = 3,
    seed                = 42,
    patience            = 30,
    mode                = 'maximize',
    selection           = TournamentSelection(k=4),
    crossover           = ArithmeticCrossover(),
    use_multiprocessing = True,
    log_path            = "backtest_run.json",
)

best, score, history = ga.run()

print(f"Best parameters: {best}")
print(f"Sharpe ratio: {score:.4f}")
print(f"Converged at generation {history[-1]['gen']}")
```
