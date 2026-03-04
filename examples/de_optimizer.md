# DEOptimizer — SHADE / L-SHADE Example Guide

`DEOptimizer` implements SHADE (Success-History based Adaptive Differential Evolution) and its L-SHADE variant. It is one of the most effective single-population optimizers for continuous problems and requires almost no manual tuning.

---

## When to Use

- All genes must be `FloatRange`. At least 2 genes are required.
- Best fit for **mild-to-moderate multimodality** — problems with a few local optima or complex fitness landscapes that are still continuous.
- Think of it as the middle ground in the evogine lineup:
  - `CMAESOptimizer` — smooth, unimodal or near-unimodal landscapes, fast convergence
  - `DEOptimizer` — continuous with some local optima, robust and self-tuning
  - `IslandModel` — highly multimodal, rugged landscapes, needs population diversity

If CMA-ES stagnates early and IslandModel feels like overkill, reach for `DEOptimizer`.

---

## Minimal Working Example

```python
from evogine import GeneBuilder, FloatRange, DEOptimizer

gb = GeneBuilder()
gb.add("x", FloatRange(-5.0, 5.0))
gb.add("y", FloatRange(-5.0, 5.0))

def fitness(ind):
    x, y = ind["x"], ind["y"]
    return -(x**2 + y**2)  # maximize negative sphere = minimize sphere

best_ind, best_score, history = DEOptimizer(
    gene_builder=gb,
    fitness_function=fitness,
    population_size=50,
    generations=200,
    seed=42,
).run()

print(best_ind)   # {"x": ~0.0, "y": ~0.0}
print(best_score) # ~0.0
```

---

## How SHADE Works

Standard Differential Evolution requires you to pick `F` (scale factor) and `CR` (crossover rate) and stick with them. Bad choices cost you convergence or diversity.

SHADE replaces fixed `F` and `CR` with **adaptive history**. Each generation:

1. `F` and `CR` are sampled from Cauchy and Normal distributions centred on values stored in a success history of size `H` (`memory_size`).
2. Whenever a trial vector beats its parent, the `F` and `CR` values that produced it are recorded.
3. The history archive is updated with a weighted Lehmer mean of successful values, giving more weight to larger fitness improvements.

The result: `F` and `CR` drift toward values that actually work on your problem. You do not need to tune them.

---

## Strategy: `current_to_best` vs `rand1`

Control the mutation strategy with the `strategy` parameter.

### `current_to_best` (default)

```
mutant = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
```

Biases mutation toward the current best individual. Converges faster, better on smoother landscapes. Use this first.

### `rand1`

```
mutant = x_r1 + F * (x_r2 - x_r3)
```

Fully random donor vector. More exploratory, less greedy. Slower convergence but better at escaping shallow local optima. Switch to this if `current_to_best` stagnates prematurely on your problem.

```python
DEOptimizer(
    gene_builder=gb,
    fitness_function=fitness,
    strategy="rand1",
    population_size=80,
    generations=500,
).run()
```

---

## L-SHADE: Linear Population Reduction

Set `linear_pop_reduction=True` to enable the L-SHADE variant.

L-SHADE starts with a larger effective population and linearly shrinks it down to a minimum (4 individuals) over the run. The idea: early generations explore broadly with many individuals; late generations exploit with a tight, converged set.

This is particularly effective under a **fixed evaluation budget** — you get the exploration benefit of a large population without paying for it in every generation.

```python
best_ind, best_score, history = DEOptimizer(
    gene_builder=gb,
    fitness_function=fitness,
    population_size=100,   # starting population, will shrink to 4
    generations=300,
    linear_pop_reduction=True,
    seed=42,
).run()
```

Watch `pop_size` in history to see the reduction in action.

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gene_builder` | `GeneBuilder` | required | FloatRange genes only, >= 2 |
| `fitness_function` | callable | required | `fn(dict) -> float` |
| `population_size` | int | `50` | Initial population size |
| `generations` | int | `200` | Maximum generations |
| `strategy` | str | `'current_to_best'` | `'current_to_best'` or `'rand1'` |
| `memory_size` | int | `6` | SHADE history size H; larger = slower adaptation |
| `linear_pop_reduction` | bool | `False` | `True` enables L-SHADE |
| `patience` | int or None | `None` | Stop after N generations without improvement |
| `min_delta` | float | `1e-6` | Minimum improvement to count as progress |
| `mode` | str | `'maximize'` | `'maximize'` or `'minimize'` |
| `seed` | int or None | `None` | Random seed for reproducibility |
| `log_path` | str or None | `None` | Write per-generation log to this CSV path |
| `on_generation` | callable or None | `None` | `fn(gen, best_score, avg_score, best_individual)` |

---

## Full Example: L-SHADE with Patience and Callback

This example optimizes the Rastrigin function in 6 dimensions — a classic multimodal benchmark with many local optima on a periodic landscape.

```python
import math
from evogine import GeneBuilder, FloatRange, DEOptimizer

# 6D Rastrigin: global minimum at origin, value = 0
def rastrigin(ind):
    n = len(ind)
    total = 10 * n + sum(
        v**2 - 10 * math.cos(2 * math.pi * v)
        for v in ind.values()
    )
    return -total  # negate to maximize

gb = GeneBuilder()
for i in range(6):
    gb.add(f"x{i}", FloatRange(-5.12, 5.12))

def on_gen(gen, best_score, avg_score, best_ind):
    # F_mean and CR_mean are available via history, not the callback signature.
    # Use this callback for lightweight progress reporting.
    if gen % 50 == 0:
        print(f"gen={gen:3d}  best={best_score:.4f}  avg={avg_score:.4f}")

best_ind, best_score, history = DEOptimizer(
    gene_builder=gb,
    fitness_function=rastrigin,
    population_size=120,
    generations=500,
    strategy="current_to_best",
    memory_size=10,
    linear_pop_reduction=True,
    patience=80,
    min_delta=1e-6,
    mode="maximize",
    seed=7,
    on_generation=on_gen,
).run()

print("\nBest individual:", best_ind)
print("Best score (negated Rastrigin):", best_score)

# Inspect adaptive parameters from history
last = history[-1]
print(f"\nFinal generation: {last['gen']}")
print(f"F_mean:  {last['F_mean']:.4f}")
print(f"CR_mean: {last['CR_mean']:.4f}")
print(f"pop_size: {last['pop_size']}")
print(f"Stop reason: {last['stop_reason']}")
```

---

## Interpreting Output

The `history` list contains one dict per generation. Key fields to watch:

| Field | What it tells you |
|---|---|
| `best_score` | All-time best score up to this generation (non-decreasing) |
| `avg_score` | Mean score of the current population; gap between this and `best_score` shows diversity |
| `F_mean` | Mean scale factor drawn from history this generation; converging toward ~0.5 is typical |
| `CR_mean` | Mean crossover rate; high CR (~0.9) = aggressive crossover; low CR (~0.1) = more conservative |
| `pop_size` | Current population size; only changes when `linear_pop_reduction=True` |
| `improved` | `True` if best score improved this generation |
| `gens_without_improvement` | Counts up; triggers patience stop when it reaches `patience` |
| `stop_reason` | `None` during run; `"max_generations"`, `"patience"`, etc. on final entry |

**Watching adaptation:**

- If `F_mean` drops very low (< 0.2) early, the optimizer is being overly exploitative — consider increasing `memory_size` or switching to `rand1`.
- If `CR_mean` oscillates wildly throughout the run without settling, the fitness landscape may be separable — `rand1` often handles separable problems better.
- If `pop_size` reaches 4 (L-SHADE floor) while `best_score` has barely improved, you may need more generations or a larger initial `population_size`.

---

## Validation Errors

`DEOptimizer` raises `ValueError` at construction for invalid configurations:

**Non-FloatRange gene:**
```python
from evogine import GeneBuilder, FloatRange, IntRange, DEOptimizer

gb = GeneBuilder()
gb.add("x", FloatRange(-1.0, 1.0))
gb.add("n", IntRange(0, 10))  # not allowed

DEOptimizer(gb, fitness_fn)
# ValueError: DEOptimizer requires all genes to be FloatRange. Found non-FloatRange gene: 'n'
```

**Fewer than 2 genes:**
```python
gb = GeneBuilder()
gb.add("x", FloatRange(-1.0, 1.0))  # only one gene

DEOptimizer(gb, fitness_fn)
# ValueError: DEOptimizer requires at least 2 genes. Got 1.
```

---

## See Also

- `CMAESOptimizer` — faster convergence on smooth, low-dimensional problems
- `IslandModel` — parallel islands with migration for highly multimodal problems
- `GeneticAlgorithm` — general purpose; supports IntRange and ChoiceList genes
- `features.md` — full parameter reference for all optimizers
