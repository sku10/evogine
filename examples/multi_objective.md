# Multi-Objective Optimization with evogine

`MultiObjectiveGA` finds a set of trade-off solutions — the **Pareto front** — instead of a
single best answer. Use it when you have 2–4 competing objectives that cannot be collapsed
into one score without losing information.

---

## When to Use This

Use `MultiObjectiveGA` when:

- You have multiple goals that pull in opposite directions (high return vs. low risk, accuracy vs. speed)
- You cannot justify a weighting scheme upfront — you want to see the full trade-off curve first
- You are presenting results to stakeholders who need to choose a preferred balance
- A single-objective score would hide important variation (e.g., two strategies with the same Sharpe but wildly different drawdowns)

Do not use it when:

- You already know the right trade-off and can express it as a weighted sum — use `GeneticAlgorithm` instead
- You have more than 4 objectives — see the tips section below

---

## Pareto Front: the Core Concept

A solution A **dominates** solution B if A is at least as good as B on every objective and
strictly better on at least one. The **Pareto front** is the set of solutions that no other
solution dominates — the genuine trade-off frontier.

Example: you are optimizing a trading strategy on Sharpe ratio (higher is better) and
maximum drawdown (lower is better).

| Strategy | Sharpe | Max Drawdown |
|----------|--------|--------------|
| A        | 2.1    | 8%           |
| B        | 1.7    | 4%           |
| C        | 1.9    | 12%          |
| D        | 1.5    | 9%           |

Strategy C is dominated by A (A beats C on both objectives). Strategy D is dominated by B.
Only A and B are non-dominated — they form the Pareto front. There is no objectively correct
choice between A and B; it depends on how much drawdown you are willing to accept. That is
exactly the decision `MultiObjectiveGA` hands back to you.

---

## Minimal Working Example (2 Objectives)

Optimize a trading strategy for **Sharpe ratio** and **win rate** simultaneously.

```python
from evogine import MultiObjectiveGA, GeneBuilder, FloatRange, IntRange, ChoiceList

# Define the search space
gb = GeneBuilder()
gb.add('fast_ma',    IntRange(5, 50))
gb.add('slow_ma',    IntRange(20, 200))
gb.add('stop_loss',  FloatRange(0.01, 0.10))
gb.add('indicator',  ChoiceList(['sma', 'ema', 'wma']))

# Fitness function returns one value per objective
def fitness(params):
    result = backtest(params)               # your backtest engine
    return [
        result['sharpe_ratio'],             # objective 0: maximize
        result['win_rate'],                 # objective 1: maximize
    ]

ga = MultiObjectiveGA(
    gene_builder=gb,
    fitness_function=fitness,
    n_objectives=2,
    objectives=['maximize', 'maximize'],
    population_size=100,
    generations=100,
    seed=42,
)

pareto_front, history = ga.run()

print(f"Found {len(pareto_front)} non-dominated strategies")
for entry in pareto_front:
    s = entry['scores']
    print(f"  Sharpe={s[0]:.3f}  WinRate={s[1]:.1%}  params={entry['individual']}")
```

**Minimizing an objective:** pass the real value and set the objective direction to
`'minimize'`. Do not negate manually — `MultiObjectiveGA` handles it internally and
returns un-negated scores in `pareto_front`.

```python
def fitness(params):
    result = backtest(params)
    return [
        result['sharpe_ratio'],         # maximize
        result['max_drawdown'],         # minimize (pass raw positive value)
    ]

ga = MultiObjectiveGA(
    ...
    n_objectives=2,
    objectives=['maximize', 'minimize'],
)
```

---

## NSGA-II vs NSGA-III

### NSGA-II (default)

The default algorithm. Uses crowding distance to maintain diversity along the Pareto front.
Well-suited for **2–3 objectives**.

```python
ga = MultiObjectiveGA(..., algorithm='nsga2')
```

### NSGA-III

Replaces crowding distance with structured **reference points** on a unit simplex (the
Das-Dennis lattice). This distributes solutions more evenly across the front and handles
**3+ objectives** significantly better than NSGA-II.

Switch to NSGA-III when:

- You have 3 or more objectives
- The NSGA-II front looks clustered in one region rather than spread across the trade-off surface

```python
ga = MultiObjectiveGA(
    ...
    algorithm='nsga3',
    reference_point_divisions=6,    # controls density of reference points on the simplex
)
```

**Choosing `reference_point_divisions`:**

| Objectives | Divisions | Reference points |
|------------|-----------|-----------------|
| 2          | 12        | 13              |
| 3          | 6         | 28              |
| 3          | 8         | 45              |
| 4          | 4         | 35              |
| 4          | 5         | 56              |

More divisions = denser reference point lattice = more diverse front, but the population
needs to be large enough to cover all reference points. A rough rule: `population_size`
should be at least as large as the number of reference points.

If the default `reference_point_divisions` (auto-computed as `max(1, 12 // n_objectives)`)
does not suit your problem, provide your own reference points:

```python
my_ref_points = [
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.0, 0.0, 1.0],
]

ga = MultiObjectiveGA(
    ...
    algorithm='nsga3',
    reference_points=my_ref_points,
)
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `gene_builder` | required | `GeneBuilder` defining the search space |
| `fitness_function` | required | `fn(dict) -> list[float]`; length must equal `n_objectives` |
| `n_objectives` | required | Number of objectives |
| `objectives` | all `'maximize'` | List of `'maximize'` or `'minimize'` per objective |
| `algorithm` | `'nsga2'` | `'nsga2'` or `'nsga3'` |
| `reference_point_divisions` | auto | Das-Dennis lattice divisions (NSGA-III only) |
| `reference_points` | `None` | User-supplied reference point list (NSGA-III only) |
| `population_size` | `100` | Number of individuals; scale up with more objectives |
| `generations` | `50` | Maximum generations to run |
| `mutation_rate` | `0.1` | Probability of mutating each gene |
| `crossover_rate` | `0.5` | Probability of crossover vs. cloning a parent |
| `patience` | `None` | Stop after N generations without hypervolume improvement |
| `min_delta` | `1e-6` | Minimum improvement to reset patience counter |
| `seed` | `None` | Integer seed for reproducibility |
| `log_path` | `None` | Write JSON run log to this file path |
| `on_generation` | `None` | Callback `fn(gen, pareto_size, hv_proxy, pareto_front)` called each generation |

---

## Full 3-Objective Example with NSGA-III

Optimize an ML model for **accuracy**, **model size** (number of parameters), and
**inference time** simultaneously.

```python
from evogine import MultiObjectiveGA, GeneBuilder, IntRange, FloatRange, ChoiceList
import time

gb = GeneBuilder()
gb.add('n_layers',        IntRange(1, 8))
gb.add('hidden_size',     IntRange(32, 512))
gb.add('dropout',         FloatRange(0.0, 0.5))
gb.add('learning_rate',   FloatRange(1e-4, 1e-1))
gb.add('activation',      ChoiceList(['relu', 'tanh', 'gelu']))
gb.add('batch_norm',      ChoiceList([True, False]))

def fitness(params):
    model = build_model(params)
    accuracy = train_and_evaluate(model)        # e.g. 0.0 – 1.0

    n_params = count_parameters(model)          # e.g. 50_000 – 5_000_000
    size_score = n_params / 1_000_000           # convert to millions

    x = get_sample_batch()
    t0 = time.perf_counter()
    model.predict(x)
    latency_ms = (time.perf_counter() - t0) * 1000

    return [accuracy, size_score, latency_ms]

def on_gen(gen, pareto_size, hv_proxy, pareto_front):
    if gen % 10 == 0:
        print(f"Gen {gen:03d} | Pareto size: {pareto_size} | HV proxy: {hv_proxy:.4f}")

ga = MultiObjectiveGA(
    gene_builder=gb,
    fitness_function=fitness,
    n_objectives=3,
    objectives=['maximize', 'minimize', 'minimize'],
    algorithm='nsga3',
    reference_point_divisions=6,    # 28 reference points for 3 objectives
    population_size=100,
    generations=200,
    mutation_rate=0.12,
    patience=40,
    seed=7,
    on_generation=on_gen,
    log_path='ml_pareto_run.json',
)

pareto_front, history = ga.run()
```

---

## Interpreting the Output

### `pareto_front`

A list of dicts. Each entry is a non-dominated solution:

```python
{
    'individual': {'n_layers': 4, 'hidden_size': 256, 'dropout': 0.2, ...},
    'scores':     [0.923, 1.4, 12.3],   # [accuracy, size_MB, latency_ms]
}
```

Scores are always in their natural direction — accuracy is the actual accuracy, size is the
actual size in millions of parameters, latency is the actual milliseconds. No sign flipping.

### Sorting and filtering

```python
# Sort by accuracy descending — pick the most accurate model you can afford
by_accuracy = sorted(pareto_front, key=lambda x: -x['scores'][0])
print("Most accurate:", by_accuracy[0])

# Filter: only models under 2M parameters and under 20ms latency
affordable = [
    e for e in pareto_front
    if e['scores'][1] < 2.0 and e['scores'][2] < 20.0
]
print(f"{len(affordable)} models meet the budget constraints")

# Pick the one with the best accuracy from the affordable set
best = max(affordable, key=lambda x: x['scores'][0])
print("Best affordable model:", best['individual'])
print(f"  Accuracy:  {best['scores'][0]:.3f}")
print(f"  Size:      {best['scores'][1]:.1f}M params")
print(f"  Latency:   {best['scores'][2]:.1f}ms")
```

### `history`

A list of per-generation dicts:

```python
{
    'gen': 42,
    'pareto_size': 17,
    'hypervolume_proxy': 1.2834,    # mean objective sum on the front; higher is better
    'improved': True,
    'gens_without_improvement': 0,
}
```

`hypervolume_proxy` is a scalar summary of the front quality used for patience tracking. It
is not the true hypervolume (which requires a reference point), but it is directionally
correct and useful for early stopping decisions.

```python
# Plot front quality over time
import matplotlib.pyplot as plt

gens = [h['gen'] for h in history]
hvs  = [h['hypervolume_proxy'] for h in history]
plt.plot(gens, hvs)
plt.xlabel('Generation')
plt.ylabel('Hypervolume proxy')
plt.title('Front quality over time')
plt.show()
```

---

## Tips

**n_objectives > 4 is very hard.** The Pareto front grows exponentially with the number of
objectives. With 5+ objectives, nearly every solution in the population is non-dominated
after just a few generations, so selection pressure collapses. Before going that route,
consider whether some objectives can be combined (e.g., combine latency and throughput into
a single efficiency score) or expressed as constraints (e.g., reject any model over 2M
params outright in the fitness function).

**Scale population_size with objectives.** A rough guide:

| Objectives | Minimum population |
|---|---|
| 2 | 50 |
| 3 | 100 |
| 4 | 200 |

For NSGA-III, ensure `population_size` is at least as large as the number of reference
points; otherwise some reference points have no candidates and diversity is lost.

**Use `patience` to avoid wasted compute.** The Pareto front often stabilizes well before
the generation limit. Set `patience=30` or `patience=50` and let early stopping handle it.

**Reproducibility.** Set `seed` for deterministic results. The seed is applied at the start
of `run()`, not at `__init__`, so you can reuse the same `MultiObjectiveGA` object with
different seeds across experiments.

**Multiprocessing.** If your fitness function is expensive, pass `use_multiprocessing=True`.
The population is evaluated in parallel using all available CPU cores. Ensure your fitness
function and any state it references are picklable.

---

## See Also

- `GeneticAlgorithm` — single-objective GA; use when you can express everything as one score
- `IslandModel` — runs N independent GAs in parallel with migration; good for avoiding local optima
- `CMAESOptimizer` — fast convergence for continuous float-only problems
- `features.md` — full parameter reference for all optimizers
- `examples/stock_strategy_optimization.md` — single-objective trading strategy example
