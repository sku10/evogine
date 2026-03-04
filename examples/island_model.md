# Island Model with evogine

The `IslandModel` runs multiple independent populations ("islands") in parallel and
periodically shares high-performing individuals between them. It is one of the most
effective tools for avoiding premature convergence on multimodal problems.

---

## When to Use IslandModel

Use `IslandModel` instead of `GeneticAlgorithm` when:

- **Your fitness landscape has multiple peaks.** A single population tends to crowd
  around one peak and ignore others. Islands maintain diversity by evolving separately.
- **GeneticAlgorithm keeps returning the same local optimum** regardless of seed.
  If you see the same score repeatedly, the population has collapsed.
- **The search space is large and you want broader initial coverage.** Four islands of
  50 each explore more of the space than one population of 200 with the same compute.
- **You have a mixed-type problem** (float + int + categorical) where gradient methods
  cannot help and pure randomness is too slow.

`GeneticAlgorithm` is fine for unimodal or mostly-smooth problems. Use `IslandModel`
when you suspect — or know — the landscape is rugged.

---

## Minimal Working Example

```python
from evogine import IslandModel, GeneBuilder, FloatRange

gb = GeneBuilder()
gb.add('x', FloatRange(-5.0, 5.0))
gb.add('y', FloatRange(-5.0, 5.0))

def fitness(params):
    x, y = params['x'], params['y']
    # Rastrigin function — 10+ local minima, hard for a single population
    return -(20 + x**2 - 10*math.cos(2*math.pi*x)
                + y**2 - 10*math.cos(2*math.pi*y))

import math
model = IslandModel(gb, fitness, n_islands=4, generations=150, seed=42)
best, score, history = model.run()

print(best)   # {'x': ..., 'y': ...}
print(score)  # fitness value
```

---

## Topology

The topology controls which islands can share migrants with each other.

```
ring                    fully_connected          star

  [0]                     [0]                    [0]
  / \                   / | \ \                  / | \
[3] [1]              [1] [2] [3]              [1] [2] [3]
  \ /                   \ | / /                  \ | /
  [2]                     ...                    [hub]
                                              (island 0)
```

| Topology           | How it works                                          | When to use                                           |
|--------------------|-------------------------------------------------------|-------------------------------------------------------|
| `ring`             | Each island sends migrants only to its two neighbours | Default. Slows homogenization; good diversity control |
| `fully_connected`  | Every island shares with every other island           | Faster convergence; use when islands plateau quickly  |
| `star`             | All islands send and receive via island 0             | Centralized control; island 0 acts as an elite hub    |

**Ring** is the right default for most problems. Use `fully_connected` only if
migration with ring topology is not helping after tuning `migration_interval`.

---

## Key Parameters

| Parameter            | Type    | Default                   | Description                                                   |
|----------------------|---------|---------------------------|---------------------------------------------------------------|
| `n_islands`          | int     | `4`                       | Number of independent sub-populations                         |
| `island_population`  | int     | `50`                      | Individuals per island                                        |
| `generations`        | int     | `100`                     | Total generations to run                                      |
| `migration_interval` | int     | `10`                      | Migrate every N generations                                   |
| `migration_size`     | int     | `2`                       | Individuals moved per migration event                         |
| `topology`           | str     | `'ring'`                  | `'ring'`, `'fully_connected'`, or `'star'`                    |
| `elitism`            | int     | `1`                       | Best N individuals preserved per island per generation        |
| `mutation_rate`      | float   | `0.1`                     | Probability of mutating each gene                             |
| `selection`          | object  | `TournamentSelection(3)`  | Selection strategy shared across all islands                  |
| `crossover`          | object  | `UniformCrossover()`      | Crossover strategy shared across all islands                  |
| `mode`               | str     | `'maximize'`              | `'maximize'` or `'minimize'`                                  |
| `patience`           | int     | `None`                    | Stop early if best score does not improve for N generations   |
| `seed`               | int     | `None`                    | Random seed for reproducibility                               |
| `log_path`           | str     | `None`                    | Write generation log to this file path (JSONL format)         |
| `on_generation`      | callable | `None`                   | Called each generation with `(gen, best_score, avg_score, best_individual)` |

---

## Full Example

This example optimizes a multimodal problem with topology choice, migration tuning,
and a live callback that prints per-generation progress.

```python
import math
from evogine import (
    IslandModel, GeneBuilder,
    FloatRange, IntRange, ChoiceList,
    TournamentSelection, UniformCrossover,
)

# --- Define the search space ---
gb = GeneBuilder()
gb.add('learning_rate', FloatRange(1e-4, 1e-1))
gb.add('momentum',      FloatRange(0.5, 0.99))
gb.add('batch_size',    IntRange(16, 256))
gb.add('optimizer',     ChoiceList(['sgd', 'adam', 'rmsprop']))

# --- Fitness: simulate a noisy, multimodal validation accuracy ---
def fitness(params):
    # Replace with your actual model training + evaluation
    return simulate_val_accuracy(params)

# --- Callback: print progress each generation ---
def on_gen(gen, best_score, avg_score, best_individual):
    if gen % 10 == 0:
        print(f"Gen {gen:4d} | best={best_score:.4f} | avg={avg_score:.4f}")

# --- Run ---
model = IslandModel(
    gb,
    fitness,
    n_islands=6,
    island_population=40,
    generations=200,
    migration_interval=15,   # migrate every 15 generations
    migration_size=3,         # move top 3 individuals per migration
    topology='ring',
    elitism=2,
    mutation_rate=0.12,
    selection=TournamentSelection(3),
    crossover=UniformCrossover(),
    mode='maximize',
    patience=40,              # stop if no improvement for 40 generations
    seed=42,
    on_generation=on_gen,
)

best_params, best_score, history = model.run()

print(f"\nBest score: {best_score:.4f}")
print(f"Best params: {best_params}")
print(f"Stopped at generation: {history[-1]['gen']}")
print(f"Stop reason: {history[-1]['stop_reason']}")
```

---

## Interpreting the Output

`model.run()` returns `(best_individual, best_score, history)`.

`history` is a list of dicts, one per generation:

```python
{
    'gen':         42,               # generation number (1-indexed)
    'best_score':  0.8731,           # all-time best score up to this generation
    'avg_score':   0.7412,           # average score across all islands this generation
    'island_bests': [0.87, 0.84, 0.81, 0.79],  # best score per island this generation
    'stop_reason': None,             # None until final entry; then 'patience' or 'generations'
}
```

`island_bests` is the key diagnostic. Use it to detect divergence and convergence:

```python
import statistics

for entry in history[::10]:
    bests = entry['island_bests']
    spread = max(bests) - min(bests)
    print(f"Gen {entry['gen']:4d} | spread={spread:.4f} | bests={[f'{b:.3f}' for b in bests]}")
```

- **High spread** — islands are exploring different regions. Good.
- **Spread collapses to ~0** — populations have homogenized. Reduce `migration_size`
  or increase `migration_interval`.

To plot best score over time:

```python
import matplotlib.pyplot as plt

gens        = [h['gen'] for h in history]
best_scores = [h['best_score'] for h in history]

plt.plot(gens, best_scores)
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.title('IslandModel Convergence')
plt.show()
```

---

## Migration Tuning Tips

Getting migration right is the main lever for controlling diversity vs. convergence speed.

**migration_interval too small (e.g., every 1-2 generations)**
- Islands share before they have time to diverge.
- Populations homogenize quickly — effectively one population with extra overhead.
- Symptom: `island_bests` spread collapses within the first 20 generations.

**migration_interval too large (e.g., every 80+ generations out of 100)**
- Islands evolve almost fully independently; migration happens too late to help.
- You get no benefit from the topology — might as well run separate GAs.
- Symptom: islands converge to very different scores and the best is never shared.

**migration_size too large**
- Sending too many individuals per event has the same effect as a small interval:
  rapid homogenization.

**Recommended starting points:**

| Scenario                    | migration_interval | migration_size |
|-----------------------------|-------------------|----------------|
| Short run (50-100 gen)      | 8-12              | 1-2            |
| Medium run (100-300 gen)    | 15-25             | 2-3            |
| Long run (300+ gen)         | 30-50             | 2-4            |
| Highly multimodal landscape | larger interval   | smaller size   |
| Faster convergence needed   | smaller interval  | larger size    |

A good diagnostic: if all `island_bests` are identical by generation 30, increase
`migration_interval` or decrease `migration_size`.

---

## See Also

- `GeneticAlgorithm` — single-population baseline; start here for simple problems
- `MultiObjectiveGA` — when you need a Pareto front instead of a single best
- `CMAESOptimizer` — fastest convergence on smooth, float-only problems
- `landscape_analysis()` — samples the fitness surface and recommends an optimizer
- `features.md` — full parameter reference for all classes
