# GeneticAlgorithm — Example Guide

## When to use

Use `GeneticAlgorithm` when you need to search a mixed (continuous + discrete + categorical) parameter space and don't have gradient information.

**Ideal scenarios:**
- Hyperparameter tuning for ML models (learning rate, layer sizes, dropout, optimizer choice)
- Trading strategy parameter search (lookback windows, thresholds, position sizing rules)
- Combinatorial configuration problems where parameters interact in non-linear ways
- Any black-box optimization where you can define a scalar fitness score
- Problems where CMA-ES is ruled out because you have integer or categorical genes

**Not ideal when:** you have a purely continuous, convex problem and can compute gradients — use scipy or CMA-ES instead.

---

## Minimal working example

Tune hyperparameters for a simple trading strategy: find the best RSI period, threshold, and position size.

```python
from evogine import GeneticAlgorithm, GeneBuilder, FloatRange, IntRange

# Define the search space
genes = (
    GeneBuilder()
    .add("rsi_period", IntRange(5, 50))
    .add("rsi_threshold", FloatRange(20.0, 45.0))
    .add("position_size", FloatRange(0.05, 0.50))
)

# Fitness function — receives a dict with your gene names as keys
def backtest_fitness(individual):
    period = individual["rsi_period"]
    threshold = individual["rsi_threshold"]
    size = individual["position_size"]
    # Replace with your actual backtest logic
    sharpe = run_backtest(period, threshold, size)
    return sharpe

best_individual, best_score, history = GeneticAlgorithm(
    gene_builder=genes,
    fitness_function=backtest_fitness,
    population_size=80,
    generations=150,
    seed=42,
).run()

print(best_individual)
# {'rsi_period': 14, 'rsi_threshold': 30.2, 'position_size': 0.18}
print(f"Best Sharpe: {best_score:.4f}")
```

---

## Key parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gene_builder` | `GeneBuilder` | required | Defines the genes and their types/ranges |
| `fitness_function` | `callable` | required | `fn(dict) -> float`. Higher is better (in maximize mode) |
| `population_size` | `int` | `100` | Number of individuals per generation. Start at 50–150 |
| `generations` | `int` | `100` | Maximum generations to run |
| `mutation_rate` | `float` | `0.1` | Probability of mutating each gene. Start at `0.05`–`0.2` |
| `elitism` | `int` | `2` | Top N individuals carried unchanged to the next generation |
| `selection` | `Selection` | `RouletteSelection()` | Parent selection strategy |
| `crossover` | `Crossover` | `UniformCrossover()` | How two parents produce offspring |
| `patience` | `int \| None` | `None` | Stop early if best score doesn't improve by `min_delta` for this many generations |
| `min_delta` | `float` | `1e-6` | Minimum improvement to count as progress (used with `patience`) |
| `mode` | `str` | `'maximize'` | `'maximize'` or `'minimize'` |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `log_path` | `str \| None` | `None` | Path to write per-generation JSONL log |
| `on_generation` | `callable \| None` | `None` | `fn(gen, best_score, avg_score, best_individual)` called each generation |
| `restart_after` | `int \| None` | `None` | Trigger a stagnation restart if no improvement for N generations |
| `restart_fraction` | `float` | `0.3` | Fraction of population preserved across a stagnation restart |
| `adaptive_mutation` | `bool` | `False` | Automatically increase mutation rate when diversity drops |
| `linear_pop_reduction` | `bool` | `False` | Linearly shrink population from `population_size` down to `min_population` |
| `min_population` | `int` | `4` | Floor for `linear_pop_reduction` |
| `constraints` | `list[callable] \| None` | `None` | List of `fn(dict) -> bool`. Individuals violating any constraint are penalized |
| `checkpoint_path` | `str \| None` | `None` | File path to save checkpoints |
| `checkpoint_every` | `int` | `10` | Save a checkpoint every N generations |

**Selection options:**
- `RouletteSelection()` — fitness-proportional; fast, but can converge early on dominated landscapes
- `TournamentSelection(k=3)` — pick best of k random candidates; more selection pressure, less sensitive to fitness scale
- `RankSelection()` — rank-based; stable on noisy fitness functions

**Crossover options:**
- `UniformCrossover()` — randomly picks each gene from either parent; good default
- `ArithmeticCrossover()` — blends parent values; best for continuous genes
- `SinglePointCrossover()` — splits the genome at one point; preserves gene blocks

---

## Full kitchen sink example

ML hyperparameter search with every major option enabled.

```python
import json
from evogine import (
    GeneticAlgorithm, GeneBuilder,
    FloatRange, IntRange, ChoiceList,
    TournamentSelection, ArithmeticCrossover,
)

# --- Search space ---
genes = (
    GeneBuilder()
    .add("learning_rate", FloatRange(1e-5, 1e-1, sigma=0.3, mutation_dist='levy'))
    .add("n_layers",      IntRange(1, 6))
    .add("hidden_size",   IntRange(32, 512))
    .add("dropout",       FloatRange(0.0, 0.6))
    .add("optimizer",     ChoiceList(["adam", "sgd", "rmsprop"]))
    .add("batch_size",    ChoiceList([16, 32, 64, 128, 256]))
    # Per-gene mutation rate: keep learning_rate stable once found
    .add("weight_decay",  FloatRange(1e-6, 1e-2, mutation_rate=0.03))
)

# --- Fitness ---
def train_and_eval(individual):
    """Return validation accuracy (higher is better)."""
    val_acc = run_training(
        lr=individual["learning_rate"],
        n_layers=individual["n_layers"],
        hidden_size=individual["hidden_size"],
        dropout=individual["dropout"],
        optimizer=individual["optimizer"],
        batch_size=individual["batch_size"],
        weight_decay=individual["weight_decay"],
    )
    return val_acc

# --- Constraint: hidden_size must be a multiple of n_layers ---
def architecture_constraint(ind):
    return ind["hidden_size"] % ind["n_layers"] == 0

# --- Progress callback ---
def on_gen(gen, best_score, avg_score, best_individual):
    if gen % 10 == 0:
        print(f"[gen {gen:3d}] best={best_score:.4f}  avg={avg_score:.4f}  "
              f"lr={best_individual['learning_rate']:.2e}  "
              f"opt={best_individual['optimizer']}")

# --- Run ---
best, score, history = GeneticAlgorithm(
    gene_builder=genes,
    fitness_function=train_and_eval,
    population_size=120,
    generations=200,
    mutation_rate=0.12,
    elitism=4,
    selection=TournamentSelection(k=5),
    crossover=ArithmeticCrossover(),
    patience=30,
    min_delta=1e-4,
    mode='maximize',
    seed=99,
    log_path="runs/hp_search.jsonl",
    on_generation=on_gen,
    restart_after=20,
    restart_fraction=0.25,
    adaptive_mutation=True,
    linear_pop_reduction=True,
    min_population=20,
    constraints=[architecture_constraint],
    checkpoint_path="runs/hp_checkpoint.json",
    checkpoint_every=10,
).run()

print("\n--- Best configuration ---")
for k, v in best.items():
    print(f"  {k}: {v}")
print(f"  validation accuracy: {score:.4f}")
```

---

## Interpreting the output

`run()` returns a 3-tuple: `(best_individual, best_score, history)`.

### `best_individual`

A plain `dict` mapping each gene name to its optimized value. Use it directly:

```python
model = build_model(
    n_layers=best["n_layers"],
    hidden_size=best["hidden_size"],
    dropout=best["dropout"],
)
```

### `best_score`

The fitness value of `best_individual`. This is the **all-time running best** — it never decreases across generations (even if a stagnation restart temporarily reduces quality).

In `minimize` mode the score is the raw minimized value (e.g. loss). Smaller is better.

### `history`

A list of dicts, one per generation. Each dict has:

| Key | Type | Description |
|---|---|---|
| `gen` | `int` | Generation number (1-indexed) |
| `best_score` | `float` | All-time best score as of this generation |
| `avg_score` | `float` | Mean fitness across the current population |
| `diversity` | `float` | Population diversity metric (0 = converged, 1 = maximally diverse) |
| `mutation_rate` | `float` | Effective mutation rate this generation (changes when `adaptive_mutation=True`) |
| `restarted` | `bool` | `True` if a stagnation restart fired this generation |
| `stop_reason` | `str \| None` | Non-None on the final entry: `'patience'`, `'generations'`, or `'manual'` |

**Example: plot learning curves**

```python
import matplotlib.pyplot as plt

gens        = [h["gen"] for h in history]
best_scores = [h["best_score"] for h in history]
avg_scores  = [h["avg_score"] for h in history]

plt.plot(gens, best_scores, label="best")
plt.plot(gens, avg_scores,  label="avg", alpha=0.6)
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
```

**Example: check why it stopped**

```python
stop = history[-1]["stop_reason"]
print(f"Stopped after {len(history)} generations: {stop}")
```

**Example: count restarts**

```python
n_restarts = sum(1 for h in history if h["restarted"])
print(f"Stagnation restarts: {n_restarts}")
```

---

## Common mistakes and tips

**Fitness function must be deterministic (or nearly so).**
If `fitness_function` has high noise (e.g. a short backtest), the GA will chase noise. Either increase the evaluation window or average over multiple seeds.

**Don't set `population_size` too small.**
With fewer than ~30 individuals the population converges before exploring the space. Start at 50–150 and reduce only if evaluation is expensive.

**`patience` counts generations without improvement, not wall time.**
If fitness improves by less than `min_delta`, the patience counter increments. Set `min_delta` to match the meaningful precision of your fitness function — for accuracy use `1e-4`, for Sharpe ratios use `0.01`.

**Constraints do not hard-reject individuals — they penalize them.**
Individuals violating a constraint receive the worst seen fitness score, so they survive but rarely reproduce. If all constraints are violated in the initial population the run still proceeds.

**`adaptive_mutation` can mask a badly shaped search space.**
If the GA keeps bumping mutation rate and never converges, your fitness landscape may be deceptive. Try `TournamentSelection(k)` with higher `k`, or simplify the search space first.

**Reproducing a run requires fixing the seed before the call.**
The seed is applied at the start of `run()`, not `__init__`, so constructing the GA and calling `run()` later is safe.

```python
ga = GeneticAlgorithm(gene_builder=genes, fitness_function=f, seed=42)
# ... other setup ...
best, score, history = ga.run()  # seed applied here
```

**Log files let you resume analysis without rerunning.**
With `log_path` set, each generation writes one JSON line. You can tail the file during a long run:

```bash
tail -f runs/hp_search.jsonl | python -c "import sys,json; [print(json.loads(l)['best_score']) for l in sys.stdin]"
```

**Use `elitism >= 1` in almost all cases.**
Without elitism the best individual can be lost to crossover. `elitism=2` is a safe default; increase to 4–6 for noisy fitness functions.

---

## See also

- `IslandModel` — run multiple GA populations in parallel with periodic migration; better for expensive fitness functions and large search spaces.
- `MultiObjectiveGA` — NSGA-II when you have two or more competing objectives (e.g. maximize return and minimize drawdown simultaneously).
- `CMAESOptimizer` — faster convergence on purely continuous (`FloatRange`-only) problems; not suitable for integer or categorical genes.
