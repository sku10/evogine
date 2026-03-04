# evogine — Ideas & Roadmap

## Vision

Make this a **publishable, production-quality evolutionary optimization library** with:
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

**Why it matters for library selection:** A user evaluating optimizers can run the benchmark
and see concrete numbers — not claims.
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

### Optimization Visualization

Visualize what the optimizer is doing — especially useful for understanding convergence
behavior, comparing optimizers, and building intuition about fitness landscapes.

**The core challenge:** Real problems have N genes = N dimensions. Can't plot a 20D
surface. But there are well-known techniques to project into 2D/3D.

**Visualization modes:**

| Mode | Input | Output | Best for |
|---|---|---|---|
| 2-gene surface | 2 genes, fitness | 3D surface plot (X, Y = genes, Z = fitness) | Problems with exactly 2 float genes |
| Gene-pair slice | N genes, pick 2, freeze rest at best values | 3D surface or contour map of the slice | Exploring one pair at a time in high-dim problems |
| Convergence path | Population history + 2-gene slice | Contour map with generation-by-generation path overlay | Seeing how the optimizer navigates the landscape |
| CMA-ES ellipse animation | CMA-ES history + 2-gene slice | Contour map with covariance ellipse shrinking/rotating per gen | Understanding why CMA-ES adapts faster than GA |
| PCA projection | All individuals from history | 2D scatter, color = fitness | Seeing main search directions in high-dim problems |
| t-SNE / UMAP | All individuals from history | 2D scatter, color = fitness | Visualizing population clusters in very high dimensions |

**Convergence path + CMA-ES ellipse is the money shot:** On a contour map you literally
see GA searching in circles while CMA-ES tilts its ellipse to follow the valley. One
image explains the 10-100x convergence difference better than any text.

**3D point cloud with transparency cutoff (the interactive mode):**
For 3 genes (or N genes projected to 3 via PCA), render the fitness landscape as a
rotatable 3D point cloud. Each point's color intensity = fitness value. Points below
a cutoff threshold (e.g. 70th percentile) are fully transparent — so you can see
*through* the low-fitness void and the high-fitness peaks glow like hot spots floating
in space. Rotate freely to inspect ridges, valleys, and clusters from any angle.

Two-layer rendering: the fitness landscape is semi-transparent colored dots (intensity
= fitness, low fitness = fully transparent). The actual candidate individuals are
small solid black dots — always visible, even deep inside a dense peak region. You
see the "actors" moving through the "world."

Animation layer: render one generation at a time and watch the black candidate dots
swarm through the transparent space toward the peak. Like a time-lapse X-ray of the
optimization process.

Zoom is essential: as candidates converge near the optimum, they cluster into a tiny
region. Without zoom the endgame is invisible. Interactive scroll-zoom (or auto-zoom
to the bounding box of current candidates) lets you follow the final approach.

Best tool: plotly (runs in browser, smooth rotation, no GUI toolkit needed) or
pyvista for heavier 3D. Matplotlib's 3D is clunky for interactive use.

**API sketch:**
```python
from evogine.viz import plot_landscape, plot_convergence, plot_cloud

# 2-gene surface
plot_landscape(gene_builder, fitness_function, gene_x="threshold", gene_y="alpha")

# Convergence path overlaid on contour map
plot_convergence(history, gene_builder, fitness_function,
                 gene_x="threshold", gene_y="alpha")

# 3D interactive point cloud with transparency cutoff
plot_cloud(history, gene_builder, fitness_function,
           gene_x="threshold", gene_y="alpha", gene_z="beta",
           cutoff_percentile=70,   # below this = transparent
           animate=True)           # step through generations
```

**Dependencies:** matplotlib (optional, like numpy for CMA-ES). Core library stays
dependency-free.

**Effort:** Medium. The math is straightforward; matplotlib is the main work.
Good candidate for a separate `evogine.viz` module or even a separate package.

---

## Scaling & Compute

### Distributed Worker Pools

**What:** Scale fitness evaluation horizontally across tens, hundreds, or thousands of CPU
workers — from a local cluster to cloud spot instances. The `fn(dict) -> float` interface
serializes trivially over the wire, so no API redesign is needed.

**Why:** For expensive fitness functions (backtests, simulations, ML training loops), a single
machine hits a ceiling. Distributed evaluation offers near-linear speedup: 100 workers,
pop_size=10000, 1s/eval → ~100s/gen instead of ~10000s. The sweet spot is fitness cost >>
network roundtrip.

**Design — external pool injection:**
```python
# Local (current)
ga = GeneticAlgorithm(..., workers=8)

# Distributed — user brings their own pool
from ray.util.multiprocessing import Pool as RayPool
pool = RayPool(ray_address="auto")
ga = GeneticAlgorithm(..., pool=pool)

# Or Dask
from dask.distributed import Client
client = Client("scheduler:8786")
ga = GeneticAlgorithm(..., pool=client)
```

A `pool` parameter accepting anything with `.map(fn, iterable)`. `multiprocessing.Pool`,
Ray, Dask, `concurrent.futures.ProcessPoolExecutor` — all have this interface. evogine
doesn't need to know which backend it is.

**Lifecycle:** External pool = user-managed. evogine doesn't create or close it.

**Resolution priority:**
1. `pool` (external, user-managed) — highest
2. `workers` (local multiprocessing) — current
3. `use_multiprocessing` (legacy compat)

**Natural fits at scale:**
- **Island model** — each island on a separate node, migration over network
- **MAP-Elites** with batch_size=1000 on 1000 workers — archive fills fast
- **Ensemble runs** — 100 independent GA runs with different seeds, best-of-100

**Effort:** Small. The internal `.map()` call is already compatible. Main work is the
pool parameter plumbing and lifecycle docs.

---

### GPU-Accelerated Optimizers

**What:** Optional GPU backend for optimizers with vectorizable inner loops. Not a
replacement for CPU mode — a parallel path for specific use cases where population sizes
are large (10k+) and the fitness function is itself GPU-native.

**Why:** CMA-ES eigendecomposition, DE trial generation, and large-batch MAP-Elites map
cleanly to GPU. Corporate users with heavy compute budgets will expect this option.

**Design — two-tier API:**
```python
# CPU (default, current)
ga = GeneticAlgorithm(..., fitness_function=my_fn, workers=8)

# GPU — requires batched tensor fitness function
ga = GeneticAlgorithm(..., fitness_function=my_batched_fn, device='cuda')
# fn(tensor[N, D]) -> tensor[N] instead of fn(dict) -> float
```

Dict↔tensor conversion only at boundaries (init, final result). Internally stays on-device.

**Implementation order by ROI:**
1. **CMA-ES** — eigendecomposition + matrix ops via cuBLAS, cleanest win
2. **DE** — vectorized trial generation, trivial in torch
3. **MAP-Elites** — batch mutations + evals, archive stays CPU-side
4. **GA/Island** — branchy operators need custom kernels, lowest ROI

**Architecture:**
- New module `evogine/backends/` with a backend protocol: `evaluate_batch`, `mutate_batch`
- Auto-detect torch/jax, graceful fallback
- Optional dependency — no torch import unless `device='cuda'`

**Effort:** Large. Requires new fitness function interface and backend abstraction. Worth
doing after distributed pools prove the demand.

---

## Research-Backed Future Ideas

State-of-the-art findings from evolutionary computation research (2022–2026).
Prioritized by practical impact and fit with evogine's architecture.

---

### Differential Evolution (DE) Optimizer

**What:** Population-based optimizer for float-only problems that generates trial vectors
via `v = x_r1 + F * (x_r2 - x_r3)` — no covariance matrix, lower memory than CMA-ES.

**Why add it:** CMA-ES and DE have complementary strengths.
- CMA-ES wins on **non-separable** functions (correlated variables)
- DE wins on **separable** functions (independent variables)
Together they cover continuous optimization completely.

**Best variant to implement: SHADE** (Success-History based Adaptive DE, CEC competition
winner). Maintains a memory of successful scale factor (F) and crossover rate (CR) values,
samples new values from distributions centered on historical successes. Zero tuning required.
L-SHADE adds linear population size reduction (large early, tiny late) — explore then exploit.

**API:** Same shape as `CMAESOptimizer`: `DEOptimizer(gene_builder, fitness_function, ...)`.
FloatRange-only like CMA-ES.

**Effort:** Medium. The core loop is simpler than CMA-ES (no eigendecomposition).

---

### NSGA-III (Many-Objective Optimization)

**What:** Extends NSGA-II for 4+ objectives. Replaces crowding distance with
reference-point-based niching — uniformly distributed reference points guide diversity
along the Pareto front. NSGA-II's crowding distance degrades badly above 3 objectives.

**Why add it:** NSGA-II is the standard for 2–3 objectives. NSGA-III is the expected
successor for engineering problems with many objectives. `pymoo` has a reference
implementation to study.

**API:** Same as `MultiObjectiveGA` — just a new `algorithm='nsga3'` parameter, or a
separate `ManyObjectiveGA` class.

**Effort:** Medium. The reference point generation (Das-Dennis procedure) is the main
new piece; the rest reuses existing NSGA-II scaffolding.

---

### Constraint Handling

**What:** Support for user-defined constraints (equality and inequality). Currently
evogine has no explicit constraint support — users must encode constraints as fitness
penalties manually.

**Best approach: Deb's Feasibility Rules** — zero tuning required:
1. Feasible solution always beats infeasible
2. Between two infeasible: prefer lower total constraint violation
3. Between two feasible: use fitness normally

**API:**
```python
ga = GeneticAlgorithm(
    ...,
    constraints=[
        lambda ind: ind['x'] + ind['y'] <= 10,   # must be True to be feasible
        lambda ind: ind['z'] >= 0,
    ],
)
```

**Why this matters:** No published GA library can be seriously recommended for
real-world engineering without constraint support. This is table-stakes for credibility.

**Effort:** Small-medium. Deb's rules slot into selection logic with no architecture changes.

---

### MAP-Elites (Quality-Diversity)

**What:** Instead of finding one optimal solution, find the best solution *for each
region of a behavior space*. Output is a filled archive — a grid mapping behavioral
descriptors to their elite performers.

Example: optimizing a robot gait. One cell = (speed=fast, stability=high). Another cell
= (speed=slow, stability=very high). MAP-Elites fills all cells simultaneously. Useful
any time you want a diverse set of high-quality solutions rather than one winner.

**Algorithm:**
1. User defines a behavior descriptor function (maps individual → 2D or ND coordinates)
2. Grid of cells, each stores the best individual with that behavior signature
3. Select a random occupied cell, mutate its individual, insert into appropriate cell if better
4. Repeat

**API:**
```python
from evogine import MAPElites

opt = MAPElites(
    gene_builder     = genes,
    fitness_function = fitness,
    behavior_fn      = lambda ind: (ind['speed'], ind['energy']),  # user-supplied
    grid_shape       = (20, 20),
    generations      = 500,
)
archive = opt.run()  # returns filled grid: {(i,j): {'individual': ..., 'score': ...}}
```

**Why add it:** Genuinely novel feature — no clean equivalent in any major Python GA
library. Increasingly used in robotics, game AI, materials design, drug discovery.
Differentiator that sets evogine apart from the crowd.

**Effort:** Medium. The algorithm itself is simpler than NSGA-II; the main work is the
archive data structure and behavior space discretization.

---

### Fitness Landscape Analysis

**What:** Analyze the structure of a fitness function before optimization — detect
ruggedness, neutrality, modality — and recommend which optimizer fits the problem.

**How:** Random walk sampling + nearest-neighbor analysis. No need to know the optimum.

**API:**
```python
from evogine import landscape_analysis

report = landscape_analysis(gene_builder, fitness_function, n_samples=1000)
# report: {
#   'ruggedness': 0.72,       # 0=smooth, 1=jagged
#   'neutrality': 0.15,       # fraction of equal-fitness neighbors
#   'estimated_modes': 3,     # number of distinct basins found
#   'recommendation': 'IslandModel',
#   'reason': 'High ruggedness and multimodal structure detected',
# }
```

**Why add it:** Directly supports the Optimizer Auto-Selection Tool idea. Standalone it
helps users understand their problem before running a long optimization. No major
dependencies — just random sampling and distance calculations.

**Effort:** Small-medium. The sampling loop is simple; the analysis metrics take some
research to implement correctly.

---

### LLM as Evolutionary Operator

**What:** Use an LLM as a crossover/mutation operator. The LLM receives descriptions of
parent individuals (as text or JSON) and generates a new offspring by "reasoning" about
which combination makes sense. Published as *EvoPrompt* (ICLR 2024) and *LLaMEA*
(IEEE TEVC Jan 2025).

**Why it's interesting:** LLM operators understand semantic structure that binary
operators cannot. For discrete/symbolic problems (prompts, configurations, code
parameters) where recombining at the bit level is meaningless, an LLM that understands
"what this gene represents" can generate smarter offspring.

**API:**
```python
ga = GeneticAlgorithm(
    ...,
    crossover = LLMCrossover(
        llm_fn=lambda parent1, parent2: my_llm_api(parent1, parent2),
    ),
)
```

**Practical scope:** Only useful for expensive fitness functions (the LLM call overhead
must be small relative to evaluation cost). Forward-looking feature for the AI-native use case.

**Effort:** Small (the integration is a callback wrapper). The LLM API is the user's problem.

---

### SHADE Parameter Adaptation (for DE and GA)

**What:** Success-History based Adaptation — maintains a memory of historically
successful parameter values (mutation rate, crossover rate), samples new values from
distributions centered on past successes. Dominant technique in CEC competition winners
2013–2024.

**For evogine:** Could replace or augment the current stagnation-based adaptive mutation
in `GeneticAlgorithm`. The "success history" approach is more principled — it adapts
based on what *actually worked*, not just whether improvement happened.

**Effort:** Small-medium. Can be implemented as an optional `AdaptationStrategy` plug-in.

---

### Levy Flight Mutation

**What:** Heavy-tailed probability distribution for mutation jumps. Instead of Gaussian
(bounded steps), Levy flight allows occasional very large jumps (global exploration) with
frequent small steps (local exploitation). Used in many recent nature-inspired algorithms.

**Why add it:** Simple addition to the existing mutation system — just an alternative
distribution alongside Gaussian. Useful for escaping local optima on multimodal landscapes.

**API:** `FloatRange(0, 1, mutation='levy')` — one extra parameter.

**Effort:** Very small. `scipy.stats.levy` or a simple pure-Python approximation.

---

### Island Model Topology Options

**What:** Currently ring topology (each island talks to its two neighbors). Research shows
topology matters significantly:
- **Ring:** Good default, maintains diversity
- **Fully connected:** Fast convergence, less diversity
- **Star:** One hub island receives and redistributes migrants
- **Dynamic / spectral:** Islands with similar gene distributions communicate;
  divergent islands stay isolated. Maintains diversity best of all.

**API:** `IslandModel(..., topology='ring' | 'fully_connected' | 'star')`

**Effort:** Small. Topology is just a mapping of which islands send to which.

---

### Linear Population Size Reduction

**What:** Start with a large population (diverse exploration), shrink it linearly to a
small population (focused exploitation) over the run. Used in L-SHADE. Simple but
effective — matches population size to the phase of search.

**API:** `GeneticAlgorithm(..., linear_pop_reduction=True)` — population shrinks from
`population_size` to a small minimum (e.g., 4) over the course of `generations`.

**Effort:** Very small. One line of math per generation.

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
