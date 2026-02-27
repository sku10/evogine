# Genetic Engine — Ideas & Roadmap

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
from genetic_engine import compare_runs

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

A small set of standard benchmark functions with known optima — so a user can verify
the library is working correctly on their system before applying it to their real problem.

**Why:** Before trusting a GA on a 6-hour stock backtest, you want to know it actually
works. Standard benchmarks give a ground truth to check against.

**Functions:**
- Sphere: `f(x) = Σxi²`, optimum at origin = 0
- Rastrigin: multi-modal, optimum at origin = 0
- Rosenbrock: narrow curved valley, optimum at (1,1,...) = 0
- OneMax: binary genes, optimum = all ones

**Format:** `examples/benchmarks.py` — runnable script that tests each benchmark and
reports whether the library found the optimum within expected generations.

**Effort:** Small. Most of the benchmark logic already exists in the tests.

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
