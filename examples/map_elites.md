# MAP-Elites: Quality-Diversity Optimization

## When to Use

Use `MAPElites` when a single best solution is not enough. It is the right tool when:

- You want a **diverse portfolio of solutions** — not just the global optimum, but the best solution in each behavioral niche.
- You want to **explore trade-offs manually** after the run, rather than encoding them as a weighted objective.
- The problem has a **meaningful behavioral space** — e.g., trading strategies that vary from fully conservative to highly aggressive, or robot gaits that range from slow-and-stable to fast-and-risky.
- You expect **multiple valid strategies** to exist and want to understand the landscape of possibilities.

If you only care about a single best answer, use `GeneticAlgorithm` or `CMAESOptimizer` instead.

---

## Concept

MAP-Elites maintains a **behavior grid** — a multi-dimensional archive where each cell represents a distinct behavioral niche. When a solution is evaluated, it is mapped to a cell by a `behavior_fn`. If the cell is empty, the solution is placed there. If it already contains a solution, the new one replaces it only if it scores higher. After the run, the archive holds **the best solution found for each behavioral niche**, giving you a complete map of quality across the behavioral space.

---

## Minimal Working Example (2D Grid)

```python
from evogine import GeneBuilder, FloatRange, MAPElites

genes = GeneBuilder()
genes.add("speed",    FloatRange(0.0, 10.0))
genes.add("accuracy", FloatRange(0.0, 1.0))

def fitness(ind):
    return ind["speed"] * ind["accuracy"]

def behavior(ind):
    # Map to a 2D behavioral space: (speed bucket, accuracy bucket)
    speed_norm    = ind["speed"] / 10.0       # already in [0, 1]
    accuracy_norm = ind["accuracy"]            # already in [0, 1]
    return (speed_norm, accuracy_norm)

archive, history = MAPElites(
    gene_builder=genes,
    fitness_function=fitness,
    behavior_fn=behavior,
    grid_shape=(10, 10),
    initial_population=200,
    generations=500,
    seed=42,
).run()

print(f"Cells filled: {len(archive)} / 100")
best_cell = max(archive, key=lambda k: archive[k]["score"])
print(f"Best cell: {best_cell}, score: {archive[best_cell]['score']:.4f}")
```

---

## Designing the `behavior_fn`

The `behavior_fn` is the most important decision in a MAP-Elites run. It determines what "behavioral diversity" means for your problem.

**Values should be in `[0, 1]`.** Values outside this range are clamped to the boundary cells — they will not cause errors, but they reduce effective grid coverage. Normalize your descriptors explicitly.

### What makes a good behavior descriptor

A good descriptor captures **something structurally different** about the solution — not just a proxy for fitness. Ask: "If two solutions score the same, what makes them meaningfully different?"

**Example 1 — Robot locomotion:**
```python
def behavior(ind):
    stride_length = ind["step_size"] / MAX_STEP   # [0, 1]: short vs long strides
    symmetry      = 1.0 - abs(ind["left_power"] - ind["right_power"])  # [0, 1]
    return (stride_length, symmetry)
```

**Example 2 — Neural network hyperparameters:**
```python
def behavior(ind):
    depth_norm = (ind["num_layers"] - 1) / 9.0   # 1–10 layers → [0, 1]
    reg_norm   = ind["dropout"] / 0.9             # 0.0–0.9 → [0, 1]
    return (depth_norm, reg_norm)
```

**Example 3 — Trading strategy:**
```python
def behavior(ind):
    aggression      = ind["position_size"] / MAX_POSITION   # [0, 1]
    diversification = 1.0 - ind["concentration"]            # [0, 1]: 0=single asset, 1=spread
    return (aggression, diversification)
```

Avoid descriptors that are directly correlated with fitness — the grid will collapse into a narrow band and coverage will be poor.

---

## Choosing `grid_shape`

`grid_shape` sets the resolution of the archive in each behavioral dimension.

| Shape | Cells | Use when |
|---|---|---|
| `(20,)` | 20 | One behavioral dimension matters |
| `(10, 10)` | 100 | Two dimensions, standard starting point |
| `(20, 20)` | 400 | Two dimensions, finer resolution needed |
| `(5, 5, 5)` | 125 | Three dimensions, keep cells manageable |
| `(10, 10, 10)` | 1000 | Three dimensions, requires large population |

**Resolution vs compute:** More cells require more evaluations to fill. A `(20, 20)` grid with 400 cells needs at least `initial_population` large enough to seed most of them — 400–1000 is reasonable. A `(5, 5)` grid can be well-covered with 100–200 initial solutions.

**1D grids** are useful when there is a single spectrum you want to explore (e.g., risk tolerance from 0 to 1).

**3D+ grids** are possible but quickly become sparse unless you use very large populations and many generations.

---

## Key Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gene_builder` | `GeneBuilder` | required | Defines the genome. |
| `fitness_function` | `callable` | required | `fn(individual: dict) -> float` |
| `behavior_fn` | `callable` | required | `fn(individual: dict) -> tuple[float, ...]` — values in `[0, 1]` |
| `grid_shape` | `tuple[int, ...]` | required | Number of cells per behavioral dimension. |
| `initial_population` | `int` | `200` | Random individuals evaluated in the seeding phase. |
| `generations` | `int` | `1000` | Mutation/selection cycles after seeding. |
| `mutation_rate` | `float` | `0.1` | Per-gene mutation probability. |
| `mode` | `str` | `'maximize'` | `'maximize'` or `'minimize'`. |
| `seed` | `int \| None` | `None` | RNG seed for reproducibility. |
| `log_path` | `str \| None` | `None` | File path for JSONL generation log. |
| `on_generation` | `callable \| None` | `None` | `fn(gen, archive_size, best_score, coverage)` called each generation. |

---

## Full Example: Trading Strategy Diversity Map

This example finds the best trading strategy in each (aggression, diversification) niche — giving a portfolio manager a complete map of risk/return trade-offs.

```python
from evogine import GeneBuilder, FloatRange, IntRange, MAPElites

MAX_POSITION = 0.5   # max fraction of portfolio per trade

genes = GeneBuilder()
genes.add("position_size",   FloatRange(0.01, MAX_POSITION))  # trade size
genes.add("stop_loss",       FloatRange(0.01, 0.20))          # max loss per trade
genes.add("take_profit",     FloatRange(0.02, 0.50))          # target gain per trade
genes.add("concentration",   FloatRange(0.0,  1.0))           # 0=diversified, 1=concentrated
genes.add("rebalance_freq",  IntRange(1, 30))                 # days between rebalancing


def simulate_returns(ind):
    """Toy fitness: reward high reward/risk ratio with some rebalancing benefit."""
    rr_ratio      = ind["take_profit"] / ind["stop_loss"]
    risk_penalty  = ind["position_size"] * ind["concentration"]
    rebal_bonus   = 1.0 / ind["rebalance_freq"]
    return rr_ratio - risk_penalty + rebal_bonus * 0.5


def trading_behavior(ind):
    """
    Two behavioral axes:
      aggression      — how large and concentrated the bets are
      diversification — inverse of concentration, normalized
    """
    aggression      = ind["position_size"] / MAX_POSITION          # [0, 1]
    diversification = 1.0 - ind["concentration"]                   # [0, 1]
    return (aggression, diversification)


archive, history = MAPElites(
    gene_builder=genes,
    fitness_function=simulate_returns,
    behavior_fn=trading_behavior,
    grid_shape=(10, 10),
    initial_population=500,
    generations=2000,
    mutation_rate=0.12,
    mode="maximize",
    seed=7,
    on_generation=lambda gen, sz, best, cov: (
        print(f"gen={gen:4d}  cells={sz:3d}  best={best:.4f}  coverage={cov:.1%}")
        if gen % 200 == 0 else None
    ),
).run()

# Show final coverage
final = history[-1]
print(f"\nFinal archive: {final['archive_size']} / 100 cells filled")
print(f"Coverage: {final['coverage']:.1%}")
print(f"Best score across all niches: {final['best_score']:.4f}")
```

---

## Interpreting the Archive

`run()` returns `(archive, history)`.

### Archive structure

```python
# archive is a dict: cell_coords -> entry
# cell_coords: tuple of ints, e.g. (3, 7) for a 2D grid
# entry: {'individual': dict, 'score': float, 'behavior': tuple}

entry = archive[(3, 7)]
print(entry["score"])       # fitness value for this cell's champion
print(entry["individual"])  # the gene dict for that champion
print(entry["behavior"])    # the raw behavior tuple that placed it here
```

### Find the best solution overall

```python
best_cell  = max(archive, key=lambda k: archive[k]["score"])
best_entry = archive[best_cell]
print(f"Global best in cell {best_cell}: score={best_entry['score']:.4f}")
print(best_entry["individual"])
```

### Query a specific niche

```python
# Find the best conservative, well-diversified strategy
# Grid is 10x10; cell (1, 9) = low aggression, high diversification
target_cell = (1, 9)
if target_cell in archive:
    entry = archive[target_cell]
    print(f"Conservative niche: score={entry['score']:.4f}")
    print(entry["individual"])
else:
    print("No solution found in that niche — try more generations or a larger initial_population.")
```

### Scan a behavioral axis

```python
# All strategies along the aggression axis, at medium diversification (column 5)
print("Aggression spectrum at medium diversification:")
for agg_cell in range(10):
    cell = (agg_cell, 5)
    if cell in archive:
        s = archive[cell]["score"]
        print(f"  aggression={agg_cell}/10 → score={s:.4f}")
```

---

## Visualizing Coverage

The `history` list contains one entry per generation with a `coverage` field — the fraction of grid cells that have been filled (0.0 to 1.0).

```python
# Print coverage growth over time
for entry in history[::100]:
    gen      = entry["gen"]
    coverage = entry["coverage"]
    filled   = entry["archive_size"]
    bar      = "#" * int(coverage * 40)
    print(f"gen={gen:4d}  [{bar:<40}] {coverage:.1%}  ({filled} cells)")
```

You can also export history to a CSV and plot coverage vs generation to see how quickly the map fills up — a flat coverage curve means increasing generations will not help and you should raise `initial_population` or `mutation_rate` instead.

---

## Tips

- **`initial_population` seeds the grid.** If the grid has N cells, you need at least N evaluations to have a chance of filling every cell. A good rule: `initial_population >= 3 * total_cells` for moderate-resolution grids.
- **Behavior values outside `[0, 1]` are clamped** to the nearest boundary cell. This is silent — you will not get an error, but grid corners will be overrepresented. Always normalize explicitly.
- **High-resolution grids go sparse fast.** A `(20, 20)` grid has 400 cells. If your archive only fills 60% of them, the unfilled cells may represent genuinely unreachable behavioral combinations, not a lack of effort. Accept sparse archives as valid results.
- **`mode='minimize'` works as expected.** The archive keeps the lowest-scoring individual in each cell when minimizing.
- **Per-gene `mutation_rate=` on gene specs is honored.** You can lock certain genes to mutate rarely (e.g., a structural gene) while others explore freely.
- **Combine with manual inspection.** The archive is a dictionary — iterate it, sort it, slice it. The goal is to give you a tool for discovery, not a single number.

---

## See Also

- `GeneticAlgorithm` — single-objective optimization, one best solution.
- `MultiObjectiveGA` — Pareto front for problems with multiple explicit objectives.
- `IslandModel` — parallel population diversity for single-objective problems.
- `CMAESOptimizer` — gradient-free continuous optimization for `FloatRange`-only problems.
- `features.md` — full parameter reference for all optimizers.
