# Genetic Engine — Features Reference

A quick guide to everything the library supports, with examples.

---

## Gene Types

### `FloatRange(low, high, sigma=0.1)`
A continuous float gene. Mutation adds Gaussian noise scaled to the range width.
`sigma` controls how aggressive mutation is (default 0.1 = 10% of range width per step).

```python
from genetic_engine import FloatRange
genes.add("threshold", FloatRange(0.0, 1.0))
genes.add("weight",    FloatRange(-5.0, 5.0, sigma=0.05))  # finer mutation
```

### `IntRange(low, high)`
An integer gene. Mutation steps ±1 per event.

```python
from genetic_engine import IntRange
genes.add("lookback", IntRange(3, 50))
genes.add("ma_period", IntRange(5, 200))
```

### `ChoiceList(options)`
A categorical gene. Mutation picks a different item from the list at random.
Safe with single-item lists (no mutation possible, returns same value).

```python
from genetic_engine import ChoiceList
genes.add("ma_type",   ChoiceList(["sma", "ema", "wma"]))
genes.add("fib_level", ChoiceList([3, 5, 8, 13, 21, 34]))
```

---

## GeneBuilder

Composes named genes into a genome. Individuals are plain Python dicts.

```python
from genetic_engine import GeneBuilder, FloatRange, IntRange, ChoiceList

genes = GeneBuilder()
genes.add("entry_threshold", FloatRange(0.0, 1.0))
genes.add("exit_window",     IntRange(3, 30))
genes.add("signal_type",     ChoiceList(["rsi", "macd", "bb"]))

# Sample a random individual
ind = genes.sample()
# → {'entry_threshold': 0.423, 'exit_window': 17, 'signal_type': 'rsi'}
```

---

## GeneticAlgorithm

### Full parameter reference

```python
from genetic_engine import GeneticAlgorithm

ga = GeneticAlgorithm(
    gene_builder      = genes,        # GeneBuilder instance (required)
    fitness_function  = my_fitness,   # fn(dict) -> float (required)
    population_size   = 100,          # number of individuals per generation
    generations       = 50,           # maximum generations to run
    mutation_rate     = 0.1,          # probability per gene of mutating
    crossover_rate    = 0.5,          # probability of crossover vs. cloning
    elitism           = 2,            # top N individuals carried unchanged
    use_multiprocessing = False,      # parallel fitness evaluation
    seed              = 42,           # random seed for reproducibility
    patience          = 20,           # early stop after N gens no improvement
    min_delta         = 1e-6,         # minimum improvement to count as progress
    log_path          = "run.json",   # write structured JSON log (optional)
)
```

### Running

```python
best_individual, best_score, history = ga.run()
```

Returns:
- `best_individual` — dict of gene name → value for the best solution found
- `best_score` — fitness score of that individual
- `history` — list of dicts, one per generation (see below)

---

## Fitness Function

Must accept a `dict` of gene values and return a single `float`.
Higher is better (maximization). To minimize something, negate it.

```python
def fitness(individual: dict) -> float:
    x = individual["x"]
    y = individual["y"]
    return -((x - 3.14)**2 + (y - 2.72)**2)  # maximize = minimize distance
```

For stock/trading use cases:
```python
def fitness(individual: dict) -> float:
    result = run_backtest(
        ma_period  = individual["ma_period"],
        threshold  = individual["entry_threshold"],
    )
    return result.sharpe_ratio  # or total_return, profit_factor, etc.
```

---

## Early Stopping

Stop automatically when no meaningful improvement is seen for `patience` generations.

```python
ga = GeneticAlgorithm(
    ...,
    patience  = 20,     # stop after 20 gens without improvement
    min_delta = 1e-6,   # improvement must exceed this to count
)
```

`patience=None` (default) disables early stopping — runs all `generations`.

---

## Generation History

`ga.run()` always returns a history list regardless of whether logging is enabled.

```python
best_ind, best_score, history = ga.run()

for entry in history:
    print(entry['gen'], entry['best_score'], entry['avg_score'], entry['improved'])
```

Each entry:
```python
{
    'gen':                      int,    # generation number
    'best_score':               float,  # best fitness in this generation
    'avg_score':                float,  # population average fitness
    'improved':                 bool,   # did best_score improve this gen?
    'gens_without_improvement': int,    # consecutive gens with no improvement
}
```

Plot convergence with matplotlib:
```python
import matplotlib.pyplot as plt
gens   = [h['gen'] for h in history]
bests  = [h['best_score'] for h in history]
avgs   = [h['avg_score'] for h in history]
plt.plot(gens, bests, label='best')
plt.plot(gens, avgs,  label='avg')
plt.legend()
plt.show()
```

---

## Structured JSON Logging

Pass `log_path` to write a machine- and AI-readable log of the full run.

```python
ga = GeneticAlgorithm(..., log_path="logs/run_001.json")
```

The log contains:
- **`run`** — timestamp, elapsed seconds
- **`config`** — all GA parameters used
- **`genes`** — full gene definitions (type, ranges, options)
- **`result`** — best individual, best score, whether it stopped early, which generation converged
- **`analysis`** — pre-computed observations for AI/human review:
  - `convergence_pattern` — one of: `converged_early`, `converged_midway`, `still_improving`, `no_progress_after_gen1`, `converged_at_end`, `too_short_to_assess`
  - `notes` — plain-English observations about what happened and what to try next
- **`history`** — full per-generation data

Example `analysis` block from a log:
```json
"analysis": {
  "score_initial": -23.4,
  "score_final": -0.002,
  "score_improvement_total": 23.398,
  "improvement_events": 8,
  "convergence_pattern": "converged_early",
  "notes": [
    "Algorithm converged well before the generation limit. If the result is good, parameters are well-tuned. If not, increase mutation_rate to escape local optima."
  ]
}
```

AI agents (Claude, GPT, etc.) can read this log directly and explain to you whether the run was successful, whether parameters need tuning, and what to try next.

---

## Reproducibility

Set `seed` for fully reproducible runs:

```python
ga = GeneticAlgorithm(..., seed=42)
```

Same seed + same fitness function = identical results every run.

---

## Multiprocessing

Enable for fitness functions that are slow (backtests, simulations):

```python
ga = GeneticAlgorithm(..., use_multiprocessing=True)
```

Uses all available CPU cores via `multiprocessing.Pool`.

**Important:** Your fitness function must be picklable (defined at module level, not a lambda or nested function) for multiprocessing to work.

---

## Minimal Example

```python
from genetic_engine import GeneticAlgorithm, GeneBuilder, FloatRange

genes = GeneBuilder()
genes.add("x", FloatRange(0, 10))
genes.add("y", FloatRange(0, 10))

def fitness(ind):
    return -((ind["x"] - 3.14)**2 + (ind["y"] - 2.72)**2)

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
print(f"Best: {best}")
print(f"Score: {score:.6f}")
print(f"Converged at generation: {history[-1]['gen']}")
```
