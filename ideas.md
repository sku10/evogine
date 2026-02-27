# Genetic Engine — Ideas & Roadmap

## Vision

Make this a **publishable, production-quality genetic algorithm library** — simpler and more controllable than DEAP, with:
- Interchangeable strategy modules (selection, crossover, mutation)
- Rich documentation that makes it easy to pick the right mode
- A test suite with real benchmark problems
- Personal use: stock algo parameter optimization (the original use case)

The core philosophy: **full control, no magic, no drift**.
DEAP is ~8000 lines and hard to debug. This stays lean but capable.

---

## Bug Fixes (do first)

### Fix seed bug
**File:** `genetic_engine.py:111`
**Problem:** `random.seed(time.time())` ignores the user-provided seed entirely.
**Fix:** `random.seed(seed)`
**Why it matters:** Reproducibility is essential for debugging and benchmarking. Without it you can't compare runs or reproduce a good result.

### ChoiceList crash on single-option list
**Problem:** If `ChoiceList` has only 1 item, mutation tries to pick a *different* index from an empty list → crash.
**Fix:** Guard with `if len(self.options) <= 1: return value`
**Why it matters:** Silent crash in the middle of a long optimization run is painful.

---

## Selection Methods (interchangeable)

### Current: Fitness-proportionate (roulette wheel)
Uses shifted scores as weights. Works but has known weaknesses: one superstar individual can dominate early, stalling diversity.

### Add: Tournament selection
Pick k random individuals, return the best. Repeat for each parent.
**Why better:** No score normalization needed, works natively with negative fitness, selection pressure is tunable via k.
**When to use:** Default recommendation — robust, simple, no edge cases.
**Parameter:** `tournament_size` (default 3; higher = more elitism)

### Add: Rank-based selection
Assign weights by rank position (1st place gets weight N, 2nd gets N-1, etc.) rather than raw fitness value.
**Why better:** Prevents any one individual from dominating based on a huge fitness gap. Maintains steady selection pressure throughout.
**When to use:** When fitness values vary wildly in magnitude across generations.

### Design: pluggable `SelectionStrategy` base class
```python
class SelectionStrategy:
    def select_parents(self, scored: list[tuple[dict, float]]) -> tuple[dict, dict]:
        raise NotImplementedError
```
User can pass `selection=TournamentSelection(k=3)` or write their own.

---

## Crossover Methods (interchangeable)

### Current: Uniform crossover
Each gene independently 50/50 from either parent. Fast and simple.

### Add: Arithmetic / blend crossover (for FloatRange)
`child_gene = t * p1_gene + (1-t) * p2_gene` where t is random in [0,1].
**Why better:** Produces offspring that interpolate between parents smoothly. Better for continuous optimization — avoids the "jump to one extreme" behavior of binary selection.
**When to use:** When genes are floats and the fitness landscape is relatively smooth.

### Add: Single-point crossover
Pick a random split index; take all genes from p1 before it, p2 after.
**Why better:** Preserves gene co-dependencies (genes that work well together stay together). Better when genes are ordered/correlated.
**When to use:** When gene order is meaningful, e.g. a sequence of trading signal thresholds.

### Design: pluggable `CrossoverStrategy`
```python
class CrossoverStrategy:
    def crossover(self, p1: dict, p2: dict, gene_builder: GeneBuilder) -> dict:
        raise NotImplementedError
```

---

## Mutation Improvements

### IntRange: scale jump size to range width
**Current problem:** ±1 step on an `IntRange(3, 300)` needs ~150 mutations on average to cross the range. Exploration is glacially slow.
**Fix:** Jump = `max(1, int(range_width * mutation_sigma))`, clipped to bounds.
**Why it matters:** Makes `IntRange` practically usable for wide integer ranges.

### Adaptive mutation rate
**Idea:** Automatically increase `mutation_rate` when the population stagnates (no improvement for N generations), decrease when converging well.
**Why:** In real optimizations you want aggressive exploration early and fine-tuning late. Fixed rate is a compromise that's rarely ideal.
**Implementation:** Track `generations_without_improvement`, scale mutation_rate up/down by a factor.

### Per-gene mutation rate override
**Idea:** Each `GeneSpec` can carry its own `mutation_rate` that overrides the global one.
**Why:** Some genes are sensitive (small range, fine control) and some are coarse (mode switches). One-size mutation rate hurts both.

---

## Termination & Control

### Early stopping
Stop when best score hasn't improved by more than `min_delta` for `patience` generations.
**Why:** Saves compute time on problems that converge quickly. Essential for production use.
**Parameters:** `patience=20`, `min_delta=1e-6`

### Stagnation restart (population injection)
When stuck for N generations, replace a fraction of the population with fresh random individuals.
**Why:** Escapes local optima without throwing away the good individuals found so far.
**Alternative to:** Restarting the whole run from scratch.

---

## Observability & Debugging

### Return generation history
**Current:** Returns only `(best_individual, best_score)`.
**Proposed:** Also return `history: list[dict]` with `gen`, `best_score`, `avg_score`, `diversity` per generation.
**Why:** Without history you can't diagnose premature convergence, stagnation, or verify the algorithm is actually learning. Critical for tuning.

### Population diversity metric
Track average pairwise distance between individuals each generation.
**Why:** When diversity collapses to near-zero you're stuck. Visible metric lets you decide to inject noise or restart.

### Callback hook
```python
ga = GeneticAlgorithm(..., on_generation=my_callback)
# my_callback(gen, best_ind, best_score, population)
```
**Why:** Lets users log to files, update a progress bar, checkpoint best individual, or implement custom early stopping — without modifying the engine.

---

## Architecture & API

### Seed numpy.random too
If any fitness function uses numpy internally (very common in stock/ML use cases), `random.seed()` won't cover it.
**Fix:** Also call `numpy.random.seed(seed)` if numpy is available.

### Checkpoint / resume
Save best individual and population to disk at each generation (or every N).
**Why:** Long runs (stock backtests over many tickers) can crash. Resume from last checkpoint instead of starting over.

### Minimize mode
Add `mode='minimize'` parameter so users don't have to negate their fitness function.
**Why:** Cognitive friction — most loss functions (MSE, drawdown, error) are minimized, not maximized. The negation trick is a paper cut for new users.

---

## Testing

### Unit tests per component
- `FloatRange`: sample always in [low, high], mutate respects bounds, sigma=0 means no change
- `IntRange`: sample always integer, mutate never goes out of bounds, ±jump stays valid
- `ChoiceList`: sample always in options, mutate always picks different value (or same if only 1)
- `GeneBuilder`: sample and mutate produce all named keys
- `GeneticAlgorithm`: known-optimum problems converge (sphere function, Rastrigin, etc.)

### Benchmark problems (standard GA test suite)
- **Sphere function** — `f(x) = -sum(xi^2)`, optimum at origin. Simplest possible.
- **Rastrigin function** — many local optima, tests ability to escape them
- **Rosenbrock (banana) function** — narrow curved valley, tests fine-tuning
- **OneMax** — binary gene version, classic combinatorial benchmark
- **Travelling Salesman (small)** — tests discrete/permutation handling

These are well-known, results are comparable to published algorithms including DEAP benchmarks.

### Property-based tests
Use `hypothesis` library to generate random gene configs and verify invariants always hold (individuals never escape bounds, population size never drifts, etc.)

---

## Bigger Ideas (future)

### Island model (parallel populations)
Run multiple independent sub-populations in parallel (processes), occasionally migrate top individuals between islands.
**Why:** Better exploration of the search space. Naturally parallel on multi-core. Standard technique for hard problems.

### Multi-objective support (Pareto / NSGA-II style)
Fitness function returns a tuple `(profit, -drawdown)`. Use Pareto dominance ranking.
**Why:** Stock optimization almost always has two competing objectives. Single scalar forces an arbitrary trade-off. Pareto gives you a frontier of solutions to choose from.

### CMA-ES hybrid
For FloatRange-only problems, offer CMA-ES (Covariance Matrix Adaptation) as an alternative engine — it's theoretically optimal for continuous spaces.
**Why:** For continuous parameter search (like trading signal thresholds) it often converges 10–100x faster than standard GA.

---

## Stock Algo Use Case (personal)

This engine was built to optimize trading strategy parameters (MA periods, signal thresholds, etc.) across stocks.

Key needs for that use case:
- **`IntRange` must work for window sizes** (currently too slow on wide ranges)
- **`ChoiceList` for MA types** (SMA, EMA, WMA)
- **Multiprocessing** is essential — each fitness call runs a full backtest
- **History logging** — need to see convergence to know if enough generations were run
- **Pareto** eventually — optimize Sharpe ratio vs. max drawdown simultaneously
- **Checkpoint/resume** — backtests over 500+ tickers can take hours

---

## Priority Order

1. Fix seed bug
2. Fix ChoiceList crash
3. Early stopping + patience
4. Return generation history
5. Tournament selection
6. IntRange large jumps
7. Arithmetic crossover for floats
8. Unit tests with benchmark problems
9. Callback hook
10. Adaptive mutation rate
11. Checkpoint/resume
12. Island model
13. Multi-objective / Pareto
