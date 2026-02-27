# evogine — Complete Guide

---

## How a Genetic Algorithm Works

If you're new to GAs, here's the core idea in plain terms.

A genetic algorithm is a search technique inspired by biological evolution. Instead of
analytically finding the best parameters, it *evolves* a population of candidate solutions
over many generations.

**Each generation:**
1. **Evaluate** every individual in the population — run your fitness function on each
2. **Select** better individuals as parents (they get more chances to reproduce)
3. **Crossover** — combine two parents to produce a child that inherits traits from both
4. **Mutate** — randomly perturb some genes to maintain diversity and explore new regions
5. **Keep the best** (elitism) — the top N individuals survive unchanged

Repeat until the score stops improving or you run out of generations.

**When to use a GA:**
- Your search space is large and you can't exhaustively test every combination
- The fitness landscape is complex (many local optima, non-differentiable)
- You don't need the theoretical global optimum — a very good solution is enough
- Common use cases: trading strategy parameters, hyperparameter tuning, engineering design

**When NOT to use a GA:**
- The search space is small enough to grid search
- Your problem is convex and differentiable — gradient descent will be faster
- You need a guaranteed optimal solution — GAs are heuristic, not exact

---

## Gene Types

Genes define your search space. Each gene describes one parameter you want to optimize.

### `FloatRange(low, high, sigma=0.1, mutation_rate=None)`

Continuous float parameter. Mutation adds Gaussian noise: the jump size is
`sigma × (high - low)`. With the default `sigma=0.1`, each mutation step is
roughly 10% of the range width.

```python
from evogine import FloatRange

genes.add("threshold",   FloatRange(0.0, 1.0))              # ±0.1 per step
genes.add("weight",      FloatRange(-5.0, 5.0, sigma=0.05)) # finer steps: ±0.5
genes.add("factor",      FloatRange(0.0, 100.0, sigma=0.2)) # wider steps: ±20
```

**Choosing sigma:**
- `sigma=0.1` (default) — good for most problems
- Smaller (`0.02–0.05`) — fine-tuning; use when you're already close to the optimum
- Larger (`0.2–0.4`) — wide exploration; use when the optimum could be anywhere

### `IntRange(low, high, sigma=0.05, mutation_rate=None)`

Integer parameter. Jump size scales with range width:
`jump = max(1, round(sigma × (high - low)))`.

```python
from evogine import IntRange

genes.add("lookback",  IntRange(3, 50))              # jump = max(1, round(0.05×47)) = 2
genes.add("ma_period", IntRange(5, 200))             # jump = max(1, round(0.05×195)) = 10
genes.add("window",    IntRange(0, 500, sigma=0.02)) # jump = max(1, round(0.02×500)) = 10
```

**Why sigma matters for IntRange:** The old approach of ±1 steps on a wide range like
`IntRange(3, 300)` required ~150 mutations on average to cross the range — exploration
was glacially slow. Sigma-based jumps make wide integer ranges practical.

**Choosing sigma:**
- `sigma=0.05` (default) — 5% of range width per step; good balance
- Smaller — for fine-grained control (e.g. exact bar count)
- Larger — for wide ranges where you want fast initial exploration

### `ChoiceList(options, mutation_rate=None)`

Categorical parameter. Mutation always picks a *different* item — never stays the same
when mutating (unless only one option exists, which is safe).

```python
from evogine import ChoiceList

genes.add("ma_type",    ChoiceList(["sma", "ema", "wma"]))
genes.add("fib_level",  ChoiceList([3, 5, 8, 13, 21, 34]))
genes.add("enabled",    ChoiceList([True, False]))
```

**Note:** ChoiceList with a single option never crashes and always returns that option — this
was a real bug in DEAP that this library explicitly guards against.

---

## GeneBuilder

Composes named genes into a genome. Individuals are plain Python `dict`s — no special
classes, no indices, just names.

```python
from evogine import GeneBuilder, FloatRange, IntRange, ChoiceList

genes = GeneBuilder()
genes.add("entry_threshold", FloatRange(0.0, 1.0))
genes.add("exit_window",     IntRange(3, 30))
genes.add("signal_type",     ChoiceList(["rsi", "macd", "bb"]))

ind = genes.sample()
# → {'entry_threshold': 0.423, 'exit_window': 17, 'signal_type': 'rsi'}
```

**Custom gene types:** Subclass `GeneSpec` and implement `sample()`, `mutate()`, `describe()`.
See the Custom Gene Types section below.

---

## Per-Gene Mutation Rate

Every gene type accepts an optional `mutation_rate=` argument that overrides the global rate
for that gene specifically.

```python
genes = GeneBuilder()
genes.add("coarse_param", FloatRange(-10.0, 10.0, mutation_rate=0.5))  # changes often
genes.add("fine_param",   FloatRange(-1.0, 1.0,  mutation_rate=0.02))  # rarely changes
genes.add("mode_switch",  ChoiceList(["a", "b"], mutation_rate=0.01))   # almost never
genes.add("normal",       FloatRange(0.0, 5.0))                         # uses global rate
```

**When to use:** When your genes have very different sensitivities. If you have one gene
that is a broad architectural choice (fast vs slow strategy) and another that is a fine
numeric threshold, a single global rate will be wrong for both. Per-gene rates let each
dimension evolve at its natural speed.

When set, the per-gene rate is always used regardless of what the global `mutation_rate`
is passed as. Per-gene rates appear in `describe()` output and in the JSON log.

---

## GeneticAlgorithm — Full Parameter Reference

```python
from evogine import GeneticAlgorithm

ga = GeneticAlgorithm(
    gene_builder        = genes,         # GeneBuilder (required)
    fitness_function    = my_fitness,    # fn(dict) -> float (required)
    population_size     = 100,           # individuals per generation
    generations         = 50,            # maximum generations to run
    mutation_rate       = 0.1,           # probability per gene of mutating
    crossover_rate      = 0.5,           # probability of crossover vs. direct clone
    elitism             = 2,             # top N copied unchanged to next generation
    use_multiprocessing = False,         # parallel fitness via multiprocessing.Pool
    seed                = 42,            # random seed for reproducibility
    patience            = 20,            # early stop after N gens with no improvement
    min_delta           = 1e-6,          # minimum score change to count as improvement
    log_path            = "run.json",    # structured JSON log (optional)
    selection           = None,          # SelectionStrategy (default: RouletteSelection)
    crossover           = None,          # CrossoverStrategy (default: UniformCrossover)
    on_generation       = None,          # callback fn(gen, best_score, avg_score, best_ind)
    mode                = 'maximize',    # 'maximize' or 'minimize'
    adaptive_mutation   = False,         # auto-adjust mutation_rate each generation
    adaptive_mutation_min = 0.01,        # lower bound for adaptive rate
    adaptive_mutation_max = 0.5,         # upper bound for adaptive rate
    checkpoint_path     = None,          # file path for save/resume checkpoints
    checkpoint_every    = 1,             # save checkpoint every N generations
    restart_after       = None,          # inject fresh individuals after N stagnant gens
    restart_fraction    = 0.3,           # fraction of population replaced on restart
)
```

### Parameter intuition

**`population_size`** — How many candidate solutions to evaluate each generation.
Larger = more diverse exploration, higher quality results, but slower per generation.
- Start: 50–100 for most problems
- Increase if the algorithm keeps finding the same poor local optimum
- Decrease if each fitness call is very slow (e.g. long backtests)

**`generations`** — Upper limit on how long to run. With `patience`, the actual number
of generations run is often much lower. Set this generously; early stopping will handle
the rest.

**`mutation_rate`** — Probability that any given gene is mutated.
- `0.05–0.1` — conservative; good for fine-tuning
- `0.1–0.2` — balanced; good default
- `0.3–0.5` — aggressive; good for wide initial exploration
- Too high → random walk, never converges
- Too low → gets stuck in local optima, no diversity

**`crossover_rate`** — Probability that two parents are crossed over (vs. one parent
cloned directly). `0.5–0.7` is typical. Higher means more gene mixing.

**`elitism`** — Top N individuals survive unchanged. This guarantees the best solution
never gets lost. `2–5` is typical. Setting to 0 disables it.

**`patience`** — Stop after this many consecutive generations without improvement.
Pair with generous `generations`. Example: `generations=500, patience=30` — it will
stop early if nothing improves for 30 gens, but can go up to 500 if needed.

**`min_delta`** — Smallest improvement that counts. Prevents micro-improvements from
resetting the patience counter forever. Default `1e-6` works for most problems.

### Running

```python
best_individual, best_score, history = ga.run()

# Resume an interrupted run:
best_individual, best_score, history = ga.run(resume_from="checkpoint.json")
```

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

**With `mode='maximize'` (default):** return higher values for better solutions.

**With `mode='minimize'`:** return lower values for better solutions. No negation needed.

**Performance tip:** The fitness function is called `population_size × generations` times.
Make it fast. Profile it before optimizing the GA parameters — a slow fitness function
dominates everything else.

**Multiprocessing:** If your fitness function is slow and CPU-bound (not I/O-bound),
use `use_multiprocessing=True`. The function must be defined at module level (not a
lambda or nested function) to be picklable.

---

## Minimize Mode

For loss functions, error rates, drawdown, or anything where lower is better:

```python
def fitness(ind):
    return mean_squared_error(ind["window"], ind["threshold"])  # just return the error

ga = GeneticAlgorithm(..., mode='minimize')
best, score, history = ga.run()
# score is 0.003, not -0.003
# history shows real values decreasing over generations
```

All reported values — returned score, history entries, log, callback — are real
un-negated values. The negation is internal and invisible. Invalid mode raises `ValueError`.

---

## Choosing Selection and Crossover Strategies

### Selection: how parents are chosen

| Strategy | Best for | Watch out for |
|---|---|---|
| `RouletteSelection` | Simple problems, smooth fitness landscape | One dominant individual can crowd out diversity |
| `TournamentSelection(k)` | Most problems — robust and controllable | k too high → aggressive selection, converges fast but may miss global optimum |
| `RankSelection` | When fitness values vary wildly in magnitude | Slower than tournament on large populations |

**Recommendation:** Use `TournamentSelection(k=3)` or `k=4` as your default. It's the most
predictable strategy across different problem types.

**Tuning k:**
- `k=2` — gentle pressure; maintains diversity; slower convergence
- `k=3–4` — balanced; recommended starting point
- `k=7+` — aggressive; converges fast but risks premature convergence

### Crossover: how parents produce children

| Strategy | Best for | Watch out for |
|---|---|---|
| `UniformCrossover` | Mixed gene types, general problems | Can break gene co-dependencies |
| `ArithmeticCrossover` | Float-heavy continuous spaces | Non-float genes fall back to uniform |
| `SinglePointCrossover` | When gene order is meaningful | Less useful for unordered genes |

**Recommendation:** Use `ArithmeticCrossover` for problems with many `FloatRange` genes
(e.g. trading thresholds). Use `UniformCrossover` for mixed types.

```python
from evogine import TournamentSelection, ArithmeticCrossover

ga = GeneticAlgorithm(
    ...,
    selection = TournamentSelection(k=4),
    crossover = ArithmeticCrossover(),
)
```

---

## Selection Strategies — Full Reference

### `RouletteSelection()` — default

Fitness-proportionate selection. Each individual's chance of being selected is
proportional to its fitness. Scores are shifted so even the worst individual has
a nonzero chance.

**The problem:** If one individual has score 1000 and everyone else has 1, that individual
will be picked almost every time. Diversity collapses quickly. Works fine on simple
smooth landscapes but struggles on complex ones.

```python
from evogine import RouletteSelection
ga = GeneticAlgorithm(..., selection=RouletteSelection())
```

### `TournamentSelection(k=3)`

Randomly sample `k` individuals, return the best. No score normalization needed —
works natively with negative fitness values.

```python
from evogine import TournamentSelection
ga = GeneticAlgorithm(..., selection=TournamentSelection(k=4))
```

**k controls selection pressure:** At `k=2`, each tournament picks randomly from 2,
so even weak individuals frequently win. At `k=10`, the winner of a 10-person tournament
is nearly always the best in the population — aggressive convergence.

**Recommended for most problems.** It's robust, requires no score normalization, and
the pressure is tunable via a single intuitive parameter.

### `RankSelection()`

Individuals are ranked 1 to N (best = N, worst = 1) and selected by rank weight,
ignoring actual score values. A score of 1000 vs. 999 gives the same relative advantage
as 0.001 vs. 0.0009.

```python
from evogine import RankSelection
ga = GeneticAlgorithm(..., selection=RankSelection())
```

**When to use:** When fitness values change dramatically in scale across generations
(e.g. score goes from 0.1 to 1000 during the run). Rank selection maintains steady,
predictable selection pressure regardless of scale.

---

## Crossover Strategies — Full Reference

### `UniformCrossover()` — default

Each gene is independently taken from either parent with 50/50 probability.
Child genes are a random mix across the entire genome.

```python
from evogine import UniformCrossover
ga = GeneticAlgorithm(..., crossover=UniformCrossover())
```

**Pros:** Simple, fast, general. **Cons:** Ignores any co-dependencies between genes.
If genes "go together" (e.g. fast_ma should always be smaller than slow_ma), uniform
crossover can break these relationships.

### `ArithmeticCrossover()`

For `FloatRange` genes: `child = t × p1 + (1-t) × p2` where `t` is random in [0,1].
The child gene is a weighted blend of both parents — it falls somewhere between them.
Non-float genes fall back to uniform.

```python
from evogine import ArithmeticCrossover
ga = GeneticAlgorithm(..., crossover=ArithmeticCrossover())
```

**Why this is better for continuous problems:** Uniform crossover on floats picks
one parent's exact value — a hard jump. Arithmetic blending explores the space *between*
the parents, which is often more fruitful on smooth fitness landscapes.

### `SinglePointCrossover()`

A random split index is chosen. Genes before the split come from parent 1, genes
after from parent 2. Preserves co-dependencies among adjacent genes.

```python
from evogine import SinglePointCrossover
ga = GeneticAlgorithm(..., crossover=SinglePointCrossover())
```

**When to use:** When gene order is meaningful — e.g. a sequence of thresholds that
build on each other, or when earlier genes define a regime and later genes tune within it.

---

## Early Stopping

Stop when the score stops improving, rather than always running to `generations`:

```python
ga = GeneticAlgorithm(
    ...,
    generations = 500,   # max, but early stopping will likely fire sooner
    patience    = 30,    # stop after 30 gens without improvement
    min_delta   = 1e-6,  # improvement must exceed this to count
)
```

`patience=None` (default) disables early stopping.

**`min_delta` tip:** Set this to match the natural precision of your fitness function.
If Sharpe ratios are typically 0.1–3.0, `min_delta=0.001` is sensible. If you're
optimizing percentage returns, `min_delta=0.01` (1 basis point) might be right.

---

## Stagnation Restart (Population Injection)

When the population gets stuck in a local optimum, inject fresh random individuals
to escape:

```python
ga = GeneticAlgorithm(
    ...,
    restart_after    = 20,   # after 20 stagnant gens, inject fresh blood
    restart_fraction = 0.3,  # replace 30% of the population
)
```

- The top `elitism` individuals always survive untouched
- `restart_fraction` of the remaining slots get new random individuals
- The rest are built normally via crossover and mutation
- Restarts fire every `restart_after` stagnant generations (at 20, 40, 60, ...)
- `history[gen]['restarted']` records when it fired

**When to use:** Problems with many local optima (multi-modal fitness landscapes).
Common signs you need this: diversity collapses to near-zero early, score stops improving
long before `patience` fires, best score is clearly suboptimal.

**Interaction with `patience`:** These work independently. You might set
`restart_after=15, patience=50` — restart every 15 stagnant gens, but give up after
50 total stagnant gens. The restarts extend the run past the patience threshold only
if they produce an improvement.

---

## Adaptive Mutation Rate

Automatically increase `mutation_rate` when stagnating, decrease when converging:

```python
ga = GeneticAlgorithm(
    ...,
    adaptive_mutation     = True,
    mutation_rate         = 0.1,   # starting rate
    adaptive_mutation_min = 0.01,  # floor
    adaptive_mutation_max = 0.5,   # ceiling
)
```

- **On improvement:** `rate × 0.95` — converging toward something, fine-tune more precisely
- **On stagnation:** `rate × 1.10` — stuck, explore harder
- Rate is clamped to `[adaptive_mutation_min, adaptive_mutation_max]`
- Rate is recorded in `history[gen]['mutation_rate']` so you can plot it

**When to use:** When you don't know the right mutation rate in advance, or when
you want aggressive early exploration that naturally transitions into fine-tuning.

---

## Generation History

Always returned as the third value from `run()`:

```python
best_ind, best_score, history = ga.run()
```

Each entry is a `dict`:

```python
{
    'gen':                      int,    # generation number (1-based)
    'best_score':               float,  # best score this generation (real value)
    'avg_score':                float,  # average score across the population (real value)
    'improved':                 bool,   # did best_score set a new overall best?
    'gens_without_improvement': int,    # consecutive gens since last improvement
    'mutation_rate':            float,  # current mutation rate (adaptive or fixed)
    'diversity':                float,  # population diversity in [0.0, 1.0]
    'restarted':                bool,   # was a stagnation restart injected this gen?
}
```

Scores are always real values — un-negated even in `mode='minimize'`.

### Understanding `diversity`

Diversity measures how spread out the population is:
- **1.0** — individuals are spread across the full gene range; maximum exploration
- **0.5** — moderate convergence; mix of exploration and exploitation
- **0.0** — all individuals are nearly identical; fully converged (or stuck)

**How it's computed:**
- `FloatRange` / `IntRange`: `(max_value - min_value) / (high - low)` — fraction of range covered
- `ChoiceList`: fraction of options that appear in the current population

**Using diversity to diagnose problems:**
- Diversity collapses to near-zero by gen 10 → mutation rate too low, or population too small
- Diversity stays high and score doesn't improve → mutation rate too high, random walk
- Diversity falls gradually and score improves → healthy convergence
- Diversity stays high even late → might benefit from stronger selection pressure (higher k)

```python
import matplotlib.pyplot as plt
gens  = [h['gen']       for h in history]
bests = [h['best_score'] for h in history]
divs  = [h['diversity']  for h in history]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
ax1.plot(gens, bests); ax1.set_title('Best score'); ax1.set_ylabel('Score')
ax2.plot(gens, divs, color='green'); ax2.set_title('Population diversity')
ax2.set_ylabel('Diversity (0–1)'); ax2.set_ylim(0, 1)
plt.tight_layout(); plt.show()
```

---

## Callback Hook

Called after every generation. Use for live plots, progress bars, custom logging,
or triggering external actions.

```python
def on_gen(gen: int, best_score: float, avg_score: float, best_individual: dict):
    print(f"Gen {gen:04d} | best={best_score:.4f} | avg={avg_score:.4f}")

ga = GeneticAlgorithm(..., on_generation=on_gen)
```

- Scores are always real values (un-negated)
- Fires on every generation, including the last one before early stopping
- `best_individual` is the best seen so far across all generations, not just this gen
- Runs synchronously (blocking) — keep it fast or use a background thread for heavy work

**Live plot example:** See `examples/live_plot.py` for a two-panel real-time visualization
using matplotlib.

---

## Checkpoint / Resume

Save the run state to disk so an interrupted run can be continued.

```python
ga = GeneticAlgorithm(
    ...,
    checkpoint_path  = "checkpoint.json",  # path to save state
    checkpoint_every = 5,                  # save every 5 generations
)
best, score, history = ga.run()
```

A checkpoint is saved:
- Every `checkpoint_every` generations
- When early stopping fires

**Resuming after a crash or interruption:**

```python
ga = GeneticAlgorithm(
    gene_builder     = genes,      # same genes as the original run
    fitness_function = fitness,    # same fitness function
    generations      = 200,        # original target — not the remaining count
    patience         = 30,
    checkpoint_path  = "checkpoint.json",
    ...
)
best, score, history = ga.run(resume_from="checkpoint.json")
```

The returned `history` contains both the generations from the checkpoint **and** the
new generations — giving you the full picture of the run from start to finish.

**Important:** `generations=200` means "run until gen 200 total", not "run 200 more gens".
If the checkpoint saved gen 80, the resumed run will execute gens 81–200.

**What's saved in the checkpoint:**
```json
{
  "gen": 80,
  "population": [...],
  "best_individual": {...},
  "best_score_internal": 1.23,
  "gens_without_improvement": 5,
  "history": [...],
  "mutation_rate": 0.08
}
```

**When to use:** Any run where a single generation takes more than a second. For stock
backtests across many tickers, a crash can mean hours of lost work. Always use checkpoints.

---

## Structured JSON Logging

```python
ga = GeneticAlgorithm(..., log_path="logs/run_001.json")
```

**Log structure:**

```json
{
  "run": {
    "timestamp": "2025-05-12T14:32:01Z",
    "elapsed_seconds": 47.3,
    "type": "single_objective"
  },
  "config": {
    "population_size": 100, "generations_max": 200, "generations_run": 63,
    "mutation_rate": 0.1, "mode": "maximize", "selection": {"strategy": "tournament", "k": 4},
    "restart_after": 25, "adaptive_mutation": true, ...
  },
  "genes": {
    "ma_period": {"type": "IntRange", "low": 5, "high": 200, "sigma": 0.05},
    "threshold": {"type": "FloatRange", "low": 0.0, "high": 1.0, "sigma": 0.1}
  },
  "result": {
    "best_score": 1.84, "best_individual": {"ma_period": 22, "threshold": 0.41},
    "early_stopped": true, "convergence_gen": 43
  },
  "analysis": {
    "convergence_pattern": "converged_early",
    "notes": ["Algorithm converged well before the generation limit..."]
  },
  "history": [...]
}
```

### Reading convergence patterns

| Pattern | Meaning | What to try |
|---|---|---|
| `converged_early` | Stopped well before generation limit | If result is good: reduce `patience` to save time. If result is bad: increase `mutation_rate` |
| `converged_midway` | Stopped around halfway | Good. If result is bad, try `restart_after` to escape the local optimum |
| `converged_at_end` | Score still improving at the end | Increase `generations` — may not be done yet |
| `still_improving` | Improved most generations | Definitely increase `generations` |
| `no_progress_after_gen1` | Stuck immediately | Increase `mutation_rate` or `population_size`, or check if fitness function works correctly |
| `too_short_to_assess` | Fewer than 3 generations | Run longer |

### AI-readable logs

The log is structured specifically so AI agents (Claude, GPT, etc.) can read it and
provide useful analysis:

```
"Here's my run log: [paste log content]
What went wrong and what should I change?"
```

The `analysis.notes` field already contains pre-computed plain-English observations.

---

## Reproducibility

```python
ga = GeneticAlgorithm(..., seed=42)
```

Seed is applied at the **start of `run()`**, not at construction. This means:
- Calling `ga.run()` twice on the same instance produces identical results
- Two instances created with the same seed produce identical results regardless of
  construction order (because each applies the seed at run-time)
- `numpy.random` is also seeded when numpy is available — important if your fitness
  function uses numpy internally

---

## Multiprocessing

```python
ga = GeneticAlgorithm(..., use_multiprocessing=True)
```

Evaluates the fitness function across all available CPU cores using `multiprocessing.Pool`.

**Requirements:**
- Fitness function must be defined at **module level** — not a lambda, not a nested function,
  not a method. Python's `multiprocessing` serializes functions with `pickle`, which can't
  handle closures.
- Works on Linux/Mac. On Windows, use `if __name__ == '__main__':` guard.

**When it helps:** CPU-bound fitness functions with a non-trivial runtime (>10ms per call).
For fast fitness functions, the process spawning overhead outweighs the parallelism benefit.

---

## Island Model

Multiple independent populations (islands) that evolve in isolation, with occasional
migration of top individuals between them.

```python
from evogine import IslandModel, TournamentSelection, ArithmeticCrossover

im = IslandModel(
    gene_builder       = genes,
    fitness_function   = fitness,
    n_islands          = 4,          # number of independent sub-populations
    island_population  = 50,         # individuals per island (total pop = 4 × 50 = 200)
    generations        = 100,
    migration_interval = 10,         # exchange individuals every 10 gens
    migration_size     = 2,          # top 2 from each island migrate
    mutation_rate      = 0.1,
    seed               = 42,
    patience           = 30,
    mode               = 'maximize',
    log_path           = "island_run.json",
    selection          = TournamentSelection(k=4),
    crossover          = ArithmeticCrossover(),
)

best, score, history = im.run()
```

**How migration works (ring topology):**
Each island sends its top `migration_size` individuals to the next island in a ring:
island 0 → island 1 → island 2 → ... → island N-1 → island 0.
The migrants **replace** the worst individuals in the target island.

**History entries** include `island_bests: list[float]` — the best score on each island
that generation. Useful for seeing whether islands are exploring different regions.

```python
for h in history:
    print(f"Gen {h['gen']} | global best: {h['best_score']:.4f} | islands: {h['island_bests']}")
```

**Why island models work:** A single large population tends to converge on one region of
the search space quickly. Multiple smaller populations explore different regions independently,
and migration lets good solutions spread — combining the benefits of diversity and convergence.

**When to use:**
- Multi-modal problems with many local optima
- When a single GA consistently gets stuck at the same suboptimal score
- When you have a moderate number of genes (>5) and a complex fitness landscape
- When you have time to run more total fitness evaluations

**Tuning:**
- `n_islands=4` with `island_population=50` is often better than one population of 200
- Migration too frequent → islands converge to the same solution (defeats the purpose)
- Migration too rare → good solutions don't spread between islands
- `migration_size=2–3` is typically enough

---

## Multi-Objective Optimization (Pareto / NSGA-II)

When you have multiple competing objectives — like Sharpe ratio vs. max drawdown —
a single-objective GA forces you to combine them into one scalar. Multi-objective GA
instead finds the **Pareto front**: the set of all solutions where you can't improve
one objective without worsening another.

```python
from evogine import MultiObjectiveGA

def fitness(ind: dict) -> list[float]:
    bt = run_backtest(**ind)
    return [bt.sharpe_ratio, bt.max_drawdown]  # two objectives

ga = MultiObjectiveGA(
    gene_builder     = genes,
    fitness_function = fitness,
    n_objectives     = 2,
    objectives       = ['maximize', 'minimize'],  # per objective direction
    population_size  = 100,
    generations      = 50,
    seed             = 42,
    patience         = 20,
    log_path         = "pareto_run.json",
)

pareto_front, history = ga.run()
```

**Return value:**

```python
# pareto_front: list of non-dominated solutions
[
    {'individual': {'fast_ma': 5, 'slow_ma': 50, ...}, 'scores': [1.8, 0.12]},
    {'individual': {'fast_ma': 8, 'slow_ma': 80, ...}, 'scores': [1.5, 0.08]},
    {'individual': {'fast_ma': 3, 'slow_ma': 30, ...}, 'scores': [2.1, 0.19]},
    ...
]
# scores are real values (un-negated), in the order you defined objectives
```

You then choose your preferred trade-off from the front:

```python
# Pick highest Sharpe with max_drawdown below 0.15
candidates = [p for p in pareto_front if p['scores'][1] <= 0.15]
best = max(candidates, key=lambda p: p['scores'][0])
```

### Understanding Pareto dominance

Solution A **dominates** B if:
- A is at least as good as B on *all* objectives, AND
- A is strictly better than B on at least one objective

The Pareto front contains every solution that is not dominated by any other — the complete
set of "best" trade-offs. There's no single winner; the caller decides which trade-off
to accept.

### History entries

```python
{
    'gen':                      int,
    'pareto_size':              int,    # how many non-dominated solutions exist
    'hypervolume_proxy':        float,  # convergence quality indicator (higher = better)
    'improved':                 bool,
    'gens_without_improvement': int,
}
```

### Callback for MultiObjectiveGA

```python
def on_gen(gen: int, pareto_size: int, hv_proxy: float, pareto_front: list):
    print(f"Gen {gen}: {pareto_size} solutions on the Pareto front")
```

### When to use
- Your optimization has two or more naturally competing goals
- You don't want to arbitrary weight-combine objectives (which hides the trade-off)
- You want to present multiple "best" solutions to a decision maker
- Classic use case: maximize return AND minimize drawdown simultaneously

---

## CMA-ES Optimizer

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a different kind of
optimizer — not a genetic algorithm. Where a GA treats each generation independently,
CMA-ES **learns the shape of the fitness landscape** and uses that knowledge to
sample smarter each generation.

The result: for problems with only float genes, CMA-ES often converges **10–100×
faster** than a standard GA.

### The idea in plain terms

A GA with Gaussian mutation searches in a sphere: every direction is equally likely.
CMA-ES starts the same way, but after a few generations it notices that moving in
some directions tends to improve the score more than others. It tilts and stretches
its sampling distribution to match — essentially learning that "northeast" is more
promising than "northwest."

The internal state is a **covariance matrix** that describes this learned ellipsoid.
Each generation, CMA-ES:

1. Samples candidate points from the current ellipsoid
2. Evaluates their fitness
3. Moves the mean toward the better half
4. Shrinks or stretches the ellipsoid based on what worked

Over time, the ellipsoid tracks the curvature of the fitness landscape and concentrates
samples where improvements are most likely.

### When to use CMA-ES

| Situation | Recommendation |
|---|---|
| All genes are `FloatRange` | **Use CMA-ES** — designed exactly for this |
| Genes are correlated (e.g. two MA periods that must move together) | **Use CMA-ES** — adapts to correlations automatically |
| Fast convergence matters (large search space, expensive fitness) | **Use CMA-ES** — typically needs far fewer evaluations |
| Fitness landscape is smooth and unimodal | **Use CMA-ES** |
| You have `IntRange` or `ChoiceList` genes | **Use GeneticAlgorithm** — CMA-ES cannot handle discrete genes |
| Fitness landscape is highly multimodal (many local optima) | **Use IslandModel** — CMA-ES can get trapped |
| Mixed gene types and multiple objectives | **Use MultiObjectiveGA** |

### Limitations (important)

**FloatRange genes only.** CMA-ES works on continuous vectors. It has no concept of
categories or integer steps. If you pass any `IntRange` or `ChoiceList` gene,
`CMAESOptimizer` raises a `ValueError` immediately at construction — not at run time.

**Minimum 2 genes.** The covariance matrix is meaningless for 1-dimensional problems.
Use `GeneticAlgorithm` for single-gene problems.

**Requires numpy.** The covariance matrix math uses linear algebra. Install with
`pip install numpy`. The core library remains dependency-free; numpy is only needed
when you call `.run()`.

**Can get stuck in local optima.** CMA-ES follows gradients in the fitness landscape.
If the landscape has multiple basins, it converges to whichever basin it enters first.
Mitigation: run multiple times with different seeds and take the best result.

### API

```python
from evogine import CMAESOptimizer, GeneBuilder, FloatRange

genes = GeneBuilder()
genes.add("threshold", FloatRange(0.0, 1.0))
genes.add("alpha",     FloatRange(0.01, 0.5))
genes.add("beta",      FloatRange(0.01, 0.5))

def fitness(ind):
    return run_my_strategy(**ind)

opt = CMAESOptimizer(
    gene_builder     = genes,
    fitness_function = fitness,
    sigma0           = 0.3,      # initial step size (fraction of gene range)
    generations      = 200,      # maximum generations
    patience         = 30,       # stop after 30 gens without improvement
    mode             = 'maximize',
    seed             = 42,
    log_path         = "cmaes_run.json",
)

best, score, history = opt.run()
# Same return shape as GeneticAlgorithm.run()
```

### Full parameter reference

| Parameter | Default | Description |
|---|---|---|
| `gene_builder` | required | `GeneBuilder` with FloatRange genes only |
| `fitness_function` | required | `dict → float`, same as GeneticAlgorithm |
| `sigma0` | `0.3` | Initial step size as fraction of gene range. 0.1 = fine search; 0.5 = coarse search |
| `generations` | `200` | Maximum number of generations |
| `popsize` | auto | Population size λ. Default: `4 + floor(3·ln(n))`. Increase for multimodal problems |
| `patience` | `None` | Stop after N gens without improvement. None = run all generations |
| `min_delta` | `1e-9` | Minimum score improvement to reset patience counter |
| `mode` | `'maximize'` | `'maximize'` or `'minimize'` |
| `seed` | `None` | Random seed for reproducibility |
| `log_path` | `None` | Path to write JSON log |
| `tolx` | `1e-8` | Stop when step size × max eigenvalue < tolx (step too small to matter) |
| `tolfun` | `1e-10` | Stop when score range over recent history is below this (flat landscape) |

### Choosing sigma0

`sigma0` is the initial step size, expressed as a fraction of the normalized gene
range (each gene is internally mapped to [0, 1]).

- `sigma0 = 0.1` — start with small steps, good when you already know roughly where
  the optimum is (e.g. warm-starting near a previous solution)
- `sigma0 = 0.3` — default; good for most problems with no prior knowledge
- `sigma0 = 0.5` — start with large steps, good for wide landscapes where the optimum
  could be anywhere; CMA-ES will shrink sigma as it converges

If CMA-ES converges too quickly to a poor result: increase `sigma0`.
If CMA-ES is slow to improve initially: decrease `sigma0`.

### Understanding the history

Each history entry contains:

```python
{
    'gen':                      14,         # generation number (1-indexed)
    'best_score':               1.742,      # best score seen so far (non-decreasing)
    'avg_score':                1.508,      # mean score of this generation's samples
    'sigma':                    0.024,      # current step size
    'improved':                 True,       # did this generation set a new best?
    'gens_without_improvement': 0,          # how many gens since last improvement
    'stop_reason':              None,       # set on the final entry if stopping early
}
```

**Reading the sigma curve:**
- Sigma starts at `sigma0` and generally decreases as the algorithm converges
- A sudden drop in sigma means the algorithm has found a promising region and is zooming in
- If sigma stays large for many generations: the landscape may be flat or multimodal
- Very small sigma (< 1e-6) + no improvement = truly converged (tolx will trigger)

**Stopping reasons:**
- `'patience'` — `patience` generations elapsed without improvement
- `'tolx'` — step size became too small; the algorithm is fully converged
- `'tolfun'` — score has been effectively flat for many generations
- `None` — ran to the full generation limit

### CMA-ES in the JSON log

The log type is `"cmaes"`. The `config` section records all strategy parameters:

```json
{
  "run":    { "type": "cmaes", "timestamp": "...", "elapsed_seconds": 1.23 },
  "config": {
    "sigma0": 0.3, "generations_max": 200, "generations_run": 67,
    "mode": "maximize", "patience": 30,
    "lambda": 10, "mu": 5, "mueff": 3.44,
    "cc": 0.44, "cs": 0.35, "c1": 0.18, "cmu": 0.04, "damps": 1.36
  },
  "genes":  { "threshold": {"type": "FloatRange", "low": 0, "high": 1}, ... },
  "result": {
    "best_score": 1.84, "best_individual": { "threshold": 0.42, ... },
    "sigma_final": 2.3e-07, "convergence_gen": 55, "stop_reason": "tolx"
  },
  "history": [...]
}
```

The parameters `lambda`, `mu`, `mueff`, `cc`, `cs`, `c1`, `cmu`, `damps` are the
internal CMA-ES strategy parameters computed automatically from the problem dimension.
They are logged so that any run is fully reproducible from the log alone.

### Comparing CMA-ES to GeneticAlgorithm

Both return `(best_individual, best_score, history)`. The gene dict format is identical.
The key differences:

| | `GeneticAlgorithm` | `CMAESOptimizer` |
|---|---|---|
| Gene types | FloatRange, IntRange, ChoiceList | FloatRange only |
| Convergence speed | Moderate | Fast (10–100× on float problems) |
| Multimodal landscapes | Reasonable (IslandModel helps) | Vulnerable to local optima |
| Dependencies | None (pure Python) | numpy required |
| Population architecture | Explicit individuals | Implicit (sampled from ellipsoid) |
| History `diversity` field | Yes | No (sigma serves the same role) |
| Early stopping | patience + min_delta | patience + tolx + tolfun |

---

## Parameter Tuning Guide

If your run isn't working well, use this checklist.

### "The algorithm converges immediately and score is bad"
1. Check `diversity` in history — if it collapses by gen 5, selection pressure is too high
2. Increase `mutation_rate` (try 0.3–0.5)
3. Increase `population_size` (try 200+)
4. Use `TournamentSelection(k=2)` for gentler selection
5. Enable `restart_after=10` to keep injecting fresh material

### "The algorithm never converges, score bounces around"
1. `mutation_rate` is too high — try 0.05–0.1
2. Check if fitness function is deterministic (same input always gives same output)
3. Reduce `crossover_rate` slightly
4. Try `RankSelection` for more controlled pressure

### "The algorithm gets a good score then stops improving"
1. Increase `patience` — it may be stopping too early
2. Enable `restart_after` — probably stuck in a local optimum
3. Try `adaptive_mutation=True` — will increase mutation rate automatically when stuck
4. Try `IslandModel` — multiple populations explore different regions
5. Increase `mutation_rate` slightly

### "Results aren't reproducible"
1. Make sure you're using `seed=` parameter
2. If your fitness function uses numpy, seed is applied to numpy too (handled automatically)
3. Don't use `use_multiprocessing=True` with a seed if process scheduling varies

### "Runs take too long"
1. Profile the fitness function first — it's usually the bottleneck
2. Enable `use_multiprocessing=True` if fitness is CPU-bound
3. Reduce `population_size` and compensate with more `generations` + `patience`
4. Use `checkpoint_every=N` so you don't lose progress if you stop

---

## Property-Based Tests

Beyond normal unit tests, this library includes property-based tests using the
**hypothesis** library. These verify fundamental invariants across hundreds of
randomly generated inputs.

### What is property-based testing?

Normal (example-based) tests check specific hand-crafted inputs:
```python
def test_floatrange_stays_in_bounds():
    spec = FloatRange(0.0, 1.0)
    assert 0.0 <= spec.mutate(0.5, mutation_rate=1.0) <= 1.0
```
This tests one specific case (`start=0.5, low=0.0, high=1.0`). You chose those values.

Property-based tests instead describe the *shape* of valid inputs and let the library
generate hundreds of random examples automatically:

```python
@given(
    low=st.floats(-1000, 999),
    high=st.floats(-999, 1000),
    start=st.floats(-1000, 1000),
    sigma=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
)
def test_floatrange_mutate_stays_in_bounds(low, high, sigma, start):
    assume(low < high)
    spec = FloatRange(low, high)
    value = max(low, min(high, start))
    assert low <= spec.mutate(value, mutation_rate=1.0) <= high
```

Hypothesis generates 300 random combinations and runs your assertion on all of them.
If any combination fails, it *shrinks* the inputs to the smallest failing example —
so instead of `low=-847.3, high=0.002, start=999.9`, you get `low=0.0, high=0.1` — much
easier to debug.

**The key advantage:** Property-based tests find edge cases you wouldn't think to test.
Extreme values, near-zero ranges, negative numbers, very large numbers — hypothesis
will try all of these systematically.

### What invariants we test

- Gene values always stay within `[low, high]` bounds after any number of mutations
- `mutation_rate=0` never changes any gene value
- `GeneBuilder.sample()` always returns exactly the gene keys that were added
- `GeneBuilder.mutate()` preserves all keys
- GA history always contains all required keys on every entry
- Best score in history is monotonically non-decreasing
- Generation counter is always sequential (1, 2, 3, ...)
- Diversity metric always in [0.0, 1.0]

### Running

```bash
pip install hypothesis
pytest tests/test_property.py
```

The tests are automatically skipped if hypothesis is not installed.

---

## Custom Gene Types

Subclass `GeneSpec` and implement three methods:

```python
from evogine import GeneSpec

class LogRange(GeneSpec):
    """Float gene sampled on a log scale — ideal for learning rates, regularization, etc."""
    def __init__(self, low: float, high: float):
        import math
        self.low  = math.log10(low)
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
        import math
        return {'type': 'LogRange', 'low': 10**self.low, 'high': 10**self.high}

genes.add("learning_rate", LogRange(1e-5, 1e-1))
```

**The three methods:**
- `sample()` — return a random valid value (used to initialize the population)
- `mutate(value, mutation_rate)` — return a possibly-mutated value; respect mutation_rate
- `describe()` — return a JSON-serializable dict for logging

---

## Complete Example

```python
from evogine import (
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

def on_gen(gen, best_score, avg_score, best_ind):
    print(f"Gen {gen:04d} | best={best_score:.4f} | avg={avg_score:.4f}")

ga = GeneticAlgorithm(
    gene_builder          = genes,
    fitness_function      = fitness,
    population_size       = 100,
    generations           = 200,
    mutation_rate         = 0.15,
    crossover_rate        = 0.7,
    elitism               = 3,
    seed                  = 42,
    patience              = 30,
    mode                  = 'maximize',
    selection             = TournamentSelection(k=4),
    crossover             = ArithmeticCrossover(),
    use_multiprocessing   = True,
    log_path              = "backtest_run.json",
    checkpoint_path       = "checkpoint.json",
    checkpoint_every      = 10,
    restart_after         = 25,
    adaptive_mutation     = True,
    on_generation         = on_gen,
)

best, score, history = ga.run()

print(f"Best parameters: {best}")
print(f"Sharpe ratio: {score:.4f}")
print(f"Generations run: {len(history)}")
```
