# evogine — Ideas & Roadmap

## Vision

Make this a **publishable, production-quality genetic algorithm library** — simpler and more controllable than DEAP, with:
- Interchangeable strategy modules (selection, crossover, mutation)
- Rich documentation that makes it easy to pick the right mode
- A test suite with real benchmark problems
- Personal use: stock algo parameter optimization (the original use case)
- Long-term: AI-readable logs + documentation that together enable an AI agent to
  diagnose runs and auto-tune parameters (see PRINCIPLES.md)

The core philosophy: **full control, no magic, no drift**.

---

## Completed

All items below have been implemented, tested, and documented.

**Bug fixes**
- [x] Fix seed bug (`random.seed(time.time())` → `random.seed(seed)`)
- [x] ChoiceList crash on single-option list

**Gene types**
- [x] FloatRange with sigma-based Gaussian mutation
- [x] IntRange with sigma-based scaled jumps (no longer ±1 only)
- [x] ChoiceList categorical gene
- [x] Per-gene mutation rate override on all gene types

**Selection strategies**
- [x] RouletteSelection (fitness-proportionate)
- [x] TournamentSelection(k) (recommended default)
- [x] RankSelection (rank-based weights)
- [x] Pluggable SelectionStrategy base class

**Crossover strategies**
- [x] UniformCrossover (50/50 per gene)
- [x] ArithmeticCrossover (blend for FloatRange)
- [x] SinglePointCrossover (split index)
- [x] Pluggable CrossoverStrategy base class

**Termination & control**
- [x] Early stopping with patience + min_delta
- [x] Stagnation restart (inject fresh individuals after N stagnant gens)
- [x] Adaptive mutation rate (auto-adjust on improvement/stagnation)
- [x] Checkpoint / resume (save state to JSON, continue after crash)

**Observability**
- [x] Return generation history from run()
- [x] Population diversity metric per generation (in history)
- [x] Callback hook (on_generation)
- [x] Structured JSON logging with AI-readable analysis section
- [x] Convergence pattern classification in logs

**Architecture**
- [x] Minimize mode (mode='minimize', no negation needed)
- [x] Reproducible seeding (seed applied at run() time, not __init__)
- [x] Numpy seed (_seed_all covers both random and numpy.random)
- [x] Multiprocessing (use_multiprocessing=True)

**Population architectures**
- [x] Island model (IslandModel: N populations + ring-topology migration)
- [x] Multi-objective GA (MultiObjectiveGA: NSGA-II Pareto ranking)
- [x] CMA-ES hybrid (CMAESOptimizer: covariance matrix adaptation for FloatRange-only problems)

**Testing**
- [x] 248 unit tests across 5 test files
- [x] Property-based tests with hypothesis library
- [x] Benchmark problems (sphere, Rastrigin) in tests

**Documentation**
- [x] README — entry point with decision guides and when-to-use advice
- [x] features.md — full reference with tuning guide and troubleshooting checklist
- [x] deap_comparison.md — detailed DEAP comparison with GitHub issue references
- [x] PRINCIPLES.md — strategic vision and design principles

---

## Remaining

### GitHub Actions CI
Auto-run `pytest` on every push to main. Show a green badge in the README.

**Why:** Makes the repo credible to potential users. Signals that the tests are
actually run and passing, not just claimed to be.

**Implementation:**
- `.github/workflows/test.yml` — run tests on Python 3.10, 3.11, 3.12
- Add badge to README

**Effort:** ~30 minutes. No code changes needed, just workflow config.

---

### Run Comparison / Analysis Tool

A utility to compare multiple JSON logs side by side — useful when tuning parameters
across several runs.

**Why:** After running with different mutation rates or selection strategies, you want to
see at a glance which run converged faster, reached a higher score, or maintained better
diversity.

**API sketch:**
```python
from evogine import compare_runs

compare_runs(["run_001.json", "run_002.json", "run_003.json"])
# Prints a table: config differences, best scores, convergence gen, patterns
```

Or returns a dict for use in AI agents:
```python
analysis = compare_runs(["run_001.json", "run_002.json"])
# analysis['winner'], analysis['key_differences'], analysis['recommendations']
```

**Why this aligns with PRINCIPLES.md:** Directly serves the AI co-pilot goal. An agent
running multiple tuning iterations would use this to understand which direction is working.

**Effort:** Small-to-medium. Mostly JSON parsing and comparison logic.

---

### Formal Benchmark Suite

A set of standard benchmark problems with known optima — serves two purposes:
(1) verify the library works correctly on your system before running a real problem,
(2) compare optimizers side by side on representative problem types.

**Why it matters for library selection:** A user evaluating this library vs DEAP vs
something else can run the benchmark and see concrete numbers — not claims.
"CMA-ES solved Rosenbrock in 180 evaluations, GA needed 4,200" is a decision.

**Benchmark problems:**

| Problem | Type | What it tests | Expected winner |
|---|---|---|---|
| Sphere (2D, 5D, 10D) | Unimodal float | Basic convergence, scaling | CMA-ES |
| Rosenbrock | Correlated float | Curved narrow valley, gene correlation | CMA-ES dramatically |
| Rastrigin | Multimodal float | Local optima escape | IslandModel |
| OneMax | Categorical/binary | Discrete gene handling | GA |
| Mixed (float + int + choice) | Mixed types | Real-world gene mix | GA (only supported) |

**What the output looks like:**
```
Sphere 5D
  GeneticAlgorithm : found optimum in  89 gens / 8,900 evals  ✓
  CMAESOptimizer   : found optimum in  31 gens /   310 evals  ✓  (28x fewer evals)

Rosenbrock 5D
  GeneticAlgorithm : did not converge in 500 gens              ✗
  CMAESOptimizer   : found optimum in  67 gens /   670 evals  ✓

Rastrigin 5D (multimodal)
  GeneticAlgorithm : best score -3.2 (local optimum)           ~
  IslandModel      : found optimum in 120 gens                 ✓
  CMAESOptimizer   : best score -2.1 (trapped in local opt)    ~
```

**Small prerequisite:** Add `total_evaluations` to result logs (`popsize × generations_run`
for GA, `lambda × generations_run` for CMA-ES). Makes the comparison concrete and dramatic.

**Format:** `examples/benchmarks.py` — runnable script, human-readable table output,
optionally saves comparison JSON for further analysis.

**Effort:** Small-to-medium. Benchmark functions are simple; the comparison table logic
is the main work.

---

### Optimizer Auto-Selection Tool

**The idea:** Given a user's gene definitions and a small sample fitness function,
automatically run all relevant optimizers for a short trial (e.g. 20 generations)
and recommend which one to use for the full run — with reasoning.

**Why this is powerful:** A new user doesn't know whether their problem is unimodal or
multimodal, whether genes are correlated, or whether CMA-ES or an IslandModel will win.
The auto-selector runs the experiment for them.

```python
from evogine import select_optimizer

recommendation = select_optimizer(
    gene_builder     = genes,
    fitness_function = fitness,
    trial_generations = 20,    # short trial per optimizer
    seed             = 42,
)

# recommendation:
# {
#   'winner':      'CMAESOptimizer',
#   'reason':      'All genes are FloatRange and CMA-ES converged 8x faster in trial',
#   'scores':      {'GeneticAlgorithm': 1.42, 'CMAESOptimizer': 1.89},
#   'evals':       {'GeneticAlgorithm': 2000, 'CMAESOptimizer': 200},
#   'suggestion':  'Run CMAESOptimizer with sigma0=0.3, patience=30',
# }
```

**What it actually tests in the trial:**
- If any non-FloatRange genes → skip CMA-ES immediately (it can't handle them)
- Run GA, CMA-ES (if eligible), IslandModel for `trial_generations` each
- Compare: best score achieved, evaluations used, diversity curve shape
- Detect early signals: diversity collapse (too aggressive), flat score (too slow)

**Output levels:**
- `recommend_only=True` → just prints a recommendation and config suggestion
- `recommend_only=False` → returns full dict for programmatic use or AI agent input

**Why this aligns with PRINCIPLES.md:** This is Phase 2.5 of the AI co-pilot vision.
The library itself can reason about which tool fits the problem — before involving an
external AI agent. The AI agent then gets a pre-filtered, well-reasoned starting point.

**Effort:** Medium. Mostly orchestration — run optimizers, compare results, apply
decision rules. Decision rules can start simple (gene type check + score comparison)
and get smarter over time.

---

### AI-Driven Model Selection

**The bigger idea (Phase 3 of PRINCIPLES.md):** The user provides their genes, fitness
function, and a goal (e.g. "converge within 100 generations, maximize Sharpe > 1.5").
An AI agent:

1. Runs `select_optimizer()` to get trial data
2. Reads the trial logs + features.md
3. Selects the best optimizer and starting parameters
4. Runs the full optimization
5. Reads the result log
6. If goal not met: adjusts parameters, reruns
7. Returns the best configuration found

```python
from evogine import ai_optimize

best, score, report = ai_optimize(
    gene_builder     = genes,
    fitness_function = fitness,
    goal             = "maximize, target > 1.5, converge within 100 gens",
    max_attempts     = 5,
    anthropic_key    = "...",
)
# report contains: which optimizers were tried, what was adjusted, why it worked
```

**Why this is the endgame:** The library was designed from day one — structured logs,
shared vocabulary with docs, pre-computed analysis — specifically so this is possible.
The benchmark and auto-selector are the stepping stones that make this reliable.

**Effort:** Large, but built on top of everything already here. The library is already
ready for this. The remaining work is the agent orchestration layer.

---

### AI Agent Integration Example

A worked example demonstrating Phase 2 of the AI co-pilot vision: an agent reads a
run log and the docs, and produces a concrete tuning recommendation.

**Why:** Makes the PRINCIPLES.md vision tangible and shows users how to use the library
with an AI assistant. Also validates that the log + docs combination actually works
for this purpose.

**Format:** `examples/ai_tuning.py` or a Jupyter notebook — runs a GA, saves the log,
calls an AI API with the log + relevant docs section, prints the recommendation.

**Effort:** Medium. Requires the Anthropic API but demonstrates the core vision.

---

## Stock Algo Use Case (personal)

This engine was built to optimize trading strategy parameters (MA periods, signal thresholds, etc.) across stocks.

Key needs for that use case:
- ✅ `IntRange` works for window sizes (sigma-based jumps)
- ✅ `ChoiceList` for MA types (SMA, EMA, WMA)
- ✅ Multiprocessing (each fitness call runs a full backtest)
- ✅ History logging (convergence visible)
- ✅ Pareto (optimize Sharpe ratio vs. max drawdown)
- ✅ Checkpoint/resume (backtests over 500+ tickers can take hours)
- ✅ CMA-ES (implemented — speeds up threshold/continuous parameter search significantly)
- ⬜ AI agent integration (auto-tune across stocks)
