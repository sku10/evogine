# CMA-ES Optimizer Guide

`CMAESOptimizer` implements the Covariance Matrix Adaptation Evolution Strategy — a
state-of-the-art algorithm for continuous optimization. It learns the shape of the
fitness landscape by adapting an internal covariance matrix, then steers sampling toward
the most promising directions. For the right problems it converges 10–100x faster than
a standard genetic algorithm.

---

## When to use CMA-ES

**Use `CMAESOptimizer` when:**

- Every gene is `FloatRange` — no `IntRange`, no `ChoiceList`
- The fitness landscape is smooth and continuous (differentiable, or nearly so)
- You want the fastest convergence and fewest fitness evaluations
- You have at least 2 genes (required — use `GeneticAlgorithm` for 1-dimensional problems)

**Do not use `CMAESOptimizer` when:**

- Any gene is `IntRange` or `ChoiceList` — it will raise `ValueError` at construction
- The landscape is highly discrete or discontinuous (noisy categorical jumps)
- You suspect many isolated local optima and need broad parallel search — use `IslandModel`

---

## Minimal working example

```python
from evogine import CMAESOptimizer, FloatRange, GeneBuilder

gb = GeneBuilder()
gb.add('x', FloatRange(-5.0, 5.0))
gb.add('y', FloatRange(-5.0, 5.0))

def fitness(params):
    # Negative Rosenbrock — minimum at (1, 1), value 0
    x, y = params['x'], params['y']
    return -(100 * (y - x**2)**2 + (1 - x)**2)

opt = CMAESOptimizer(gb, fitness, seed=42)
best, score, history = opt.run()

print(best)   # {'x': ~1.0, 'y': ~1.0}
print(score)  # ~0.0
```

---

## sigma0 tuning guide

`sigma0` is the initial step size expressed as a **fraction of each gene's range**. It
controls how broadly CMA-ES explores before it starts converging.

| Situation | Recommended sigma0 |
|---|---|
| No prior knowledge of where the optimum is | `0.3` (default) |
| Safe, general-purpose starting point | `0.2`–`0.4` |
| You have a good warm-start guess (mean is close) | `0.1`–`0.15` |
| Very wide search space, highly uncertain | `0.4`–`0.5` |
| CMA-ES stagnates early (stuck in a poor region) | Increase — try `0.4` or `0.5` |
| CMA-ES diverges or never settles | Decrease — try `0.15` or `0.2` |

The internal working space is always normalized to `[0, 1]^n`, so `sigma0=0.3` means the
initial step size is 30% of each gene's range regardless of the actual units.

**Rule of thumb:** if you have no starting point information, `0.3` is almost always fine.
Only tune it if you observe pathological behavior (see Tips section below).

---

## Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gene_builder` | `GeneBuilder` | required | FloatRange genes only — raises `ValueError` otherwise |
| `fitness_function` | `Callable` | required | `dict -> float`, same interface as `GeneticAlgorithm` |
| `sigma0` | `float` | `0.3` | Initial step size as a fraction of each gene's range |
| `population_size` | `int \| None` | `None` | Lambda (sample size per generation). Default: `4 + floor(3 * ln(n_genes))`. Rarely needs changing |
| `generations` | `int` | `200` | Maximum number of generations |
| `patience` | `int \| None` | `None` | Stop after this many generations without improvement |
| `min_delta` | `float` | `1e-6` | Minimum score improvement to reset the patience counter |
| `mode` | `str` | `'maximize'` | `'maximize'` or `'minimize'` |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `log_path` | `str \| None` | `None` | Path to write a JSON log file |
| `on_generation` | `Callable \| None` | `None` | Called each generation: `fn(gen, best_score, avg_score, best_individual)` |

Internal stopping conditions (rarely need adjustment):

| Parameter | Default | Meaning |
|---|---|---|
| `tolx` | `1e-8` | Stop when step size times max eigenvalue falls below this — the distribution has collapsed |
| `tolfun` | `1e-10` | Stop when the best score has not changed meaningfully over a rolling window of recent generations |

---

## Full example with patience, log_path, and on_generation

```python
from evogine import CMAESOptimizer, FloatRange, GeneBuilder

# 5-dimensional Ackley function (highly multimodal, global min at origin = 0)
import math

def ackley(params):
    vals = list(params.values())
    n = len(vals)
    a, b, c = 20, 0.2, 2 * math.pi
    sum_sq   = sum(v**2 for v in vals) / n
    sum_cos  = sum(math.cos(c * v) for v in vals) / n
    result   = -a * math.exp(-b * math.sqrt(sum_sq)) - math.exp(sum_cos) + a + math.e
    return -result  # minimize Ackley = maximize negative Ackley

gb = GeneBuilder()
for name in ['x1', 'x2', 'x3', 'x4', 'x5']:
    gb.add(name, FloatRange(-5.0, 5.0))

def on_gen(gen, best_score, avg_score, best_individual):
    if gen % 10 == 0:
        print(f"  gen {gen:3d} | best={best_score:.6f} | avg={avg_score:.6f}")

opt = CMAESOptimizer(
    gb,
    ackley,
    sigma0=0.35,
    generations=500,
    patience=50,        # stop if no improvement for 50 generations
    min_delta=1e-6,
    mode='maximize',    # we negated Ackley, so maximize
    seed=7,
    log_path='/tmp/cmaes_ackley_run.json',
    on_generation=on_gen,
)

best, score, history = opt.run()

print(f"\nBest score : {score:.8f}")
print(f"Best params: {best}")
print(f"Generations: {len(history)}")
print(f"Stop reason: {history[-1]['stop_reason']}")
```

---

## Interpreting the output

`run()` returns a tuple `(best_individual, best_score, history)` — the same shape as
`GeneticAlgorithm.run()` and `IslandModel.run()`.

```python
best, score, history = opt.run()
```

**`best_individual`** — `dict` mapping gene names to their optimal values:

```python
{'x1': 0.0003, 'x2': -0.0001, 'x3': 0.0002, 'x4': -0.0001, 'x5': 0.0000}
```

**`best_score`** — the all-time best fitness value seen during the run (non-decreasing,
matches the GA's history convention).

**`history`** — list of one dict per generation:

```python
{
    'gen':                       42,
    'best_score':                -0.00031,   # running best (your fitness units)
    'avg_score':                 -0.18450,   # generation average
    'sigma':                      0.04213,   # current step size
    'improved':                   True,      # did this gen set a new best?
    'gens_without_improvement':   0,
    'stop_reason':                None,      # set on the final entry if early stop
}
```

`stop_reason` on the last history entry tells you why the run ended:

| Value | Meaning |
|---|---|
| `None` | Hit the `generations` limit |
| `'patience'` | No improvement for `patience` generations |
| `'tolx'` | Step size collapsed — distribution converged to a point |
| `'tolfun'` | Score plateau — no meaningful change in recent generations |

**`sigma` in history** is particularly useful for diagnosing convergence. A healthy run
shows sigma decreasing steadily. If sigma never decreases, the landscape may be too
noisy. If it collapses to near zero in the first 10 generations, `sigma0` was too small.

---

## What happens with non-FloatRange genes

`CMAESOptimizer` raises `ValueError` at construction if any gene is not `FloatRange`:

```python
from evogine import CMAESOptimizer, FloatRange, IntRange, GeneBuilder

gb = GeneBuilder()
gb.add('learning_rate', FloatRange(0.0001, 0.1))
gb.add('n_layers',      IntRange(1, 8))          # not a FloatRange

opt = CMAESOptimizer(gb, fitness)
# ValueError: CMAESOptimizer only supports FloatRange genes.
# Gene 'n_layers' is IntRange. Use GeneticAlgorithm for mixed gene types.
```

The error is raised immediately — you do not have to wait for `run()` to discover the
incompatibility. Switch to `GeneticAlgorithm` or `IslandModel` for problems with
`IntRange` or `ChoiceList` genes.

---

## CMA-ES vs DEOptimizer vs GeneticAlgorithm

| | CMAESOptimizer | DEOptimizer | GeneticAlgorithm |
|---|---|---|---|
| **Gene types** | FloatRange only | FloatRange only | FloatRange, IntRange, ChoiceList |
| **Landscape** | Smooth, continuous | Smooth-to-rugged | Any |
| **Convergence speed** | Fastest on smooth problems | Fast, robust | Moderate |
| **Multimodal handling** | Fair (single population) | Fair | Good (use IslandModel for best) |
| **Correlation awareness** | Yes — adapts covariance matrix | No | No |
| **Population size** | Auto-set, small (`~4 + 3 ln n`) | Larger (`~10 * n`) | User-set |
| **Requires numpy** | Yes | Yes | No |

**When CMA-ES wins:** few genes (2–30), smooth differentiable-ish landscape, want the
fewest fitness evaluations to converge.

**When DEOptimizer wins:** moderate-to-high gene count, rugged but still continuous
landscape, CMA-ES stagnates due to early covariance collapse.

**When GeneticAlgorithm wins:** mixed gene types, discrete choices, or when
interpretability and fine-grained control over selection and crossover matter more than
raw speed.

For highly multimodal problems, wrap any of these in `IslandModel` or run CMA-ES
multiple times from different seeds and take the best result.

---

## Tips

**If CMA-ES stagnates early** (score barely improves in the first 20–30 generations):

- Increase `sigma0` — try `0.4` or `0.5`. The initial distribution may be too narrow to
  reach the basin of attraction.
- Increase `population_size` — a larger lambda helps on multimodal surfaces.
- Check whether the fitness function is nearly flat in most of the search space; if so,
  CMA-ES may need a warm start or a reformulated fitness.

**If CMA-ES diverges or oscillates** (score worsens after initial progress):

- Decrease `sigma0` — try `0.15` or `0.1`. The step size may be overshooting the optimum.
- Verify gene ranges are reasonable — extremely wide ranges relative to the solution
  region force CMA-ES to work in a very sparse space.

**If `stop_reason` is `tolx` very early** (within the first 10–20 generations):

- `sigma0` is too small. The distribution collapsed before meaningful search happened.
  Increase it.

**Reproducibility:** set `seed` to any integer. The seed is applied at the start of
`run()`, not `__init__`, so you can construct the optimizer once and call `run()`
multiple times with different seeds if needed (though that requires a new instance since
the seed is fixed at construction).

**Logging:** use `log_path` to save a full JSON record of the run. The log format is
identical to `GeneticAlgorithm` logs, so the same analysis tools apply.

---

## See also

- `GeneticAlgorithm` — the general-purpose optimizer; handles all gene types
- `IslandModel` — parallel island GA; best for multimodal landscapes
- `MultiObjectiveGA` — NSGA-II for problems with multiple competing objectives
- `DEOptimizer` — Differential Evolution; strong on rugged continuous landscapes
- `features.md` — full parameter reference for all optimizers
- `examples/stock_strategy_optimization.md` — a real-world walkthrough choosing between optimizers
