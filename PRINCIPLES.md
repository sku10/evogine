# evogine — Design Principles

This document captures the strategic vision and design principles that guide
every decision in this library. It exists so that new features, API changes,
and documentation updates all pull in the same direction.

Read this before adding a feature, changing a parameter name, or restructuring a log.

---

## Optimizer Roster

evogine implements six optimizer classes and one diagnostic function, all following
the same design contract:

| Class | Algorithm | Gene constraint | Return shape |
|---|---|---|---|
| `GeneticAlgorithm` | Standard GA | Any | `(individual, score, history)` |
| `IslandModel` | Multi-population GA | Any | `(individual, score, history)` |
| `MultiObjectiveGA` | NSGA-II / NSGA-III | Any | `(pareto_front, history)` |
| `CMAESOptimizer` | CMA-ES | FloatRange only | `(individual, score, history)` |
| `DEOptimizer` | SHADE / L-SHADE | FloatRange only | `(individual, score, history)` |
| `MAPElites` | Quality-diversity | Any | `(archive, history)` |
| `landscape_analysis()` | Landscape sampling | Any | `dict` report |

Most classes share common conventions: `seed=`, `log_path=`, `on_generation=`. Single-objective
optimizers share `mode=` and `patience=`; MultiObjectiveGA uses per-objective `objectives=`
instead of `mode=`; MAPElites has no `patience=`. Scores in history and callbacks are always
real user-facing values (un-negated even in minimize mode).

---

## Vision

**Make evolutionary optimization runs fully interpretable — by humans and by AI agents.**

A run should produce enough structured information that an intelligent agent (human or AI)
can read the output, understand what happened, diagnose what went wrong, and propose
concrete parameter changes — without needing to read the source code.

---

## Strategic Goals

### 1. The AI Co-Pilot Goal

This is the most forward-looking goal and the one that most shapes the library's design.

The aim: a user runs a genetic algorithm, gets poor results, pastes the JSON log to an
AI agent (Claude, GPT, or a future automated system), and the agent can:

- Understand the problem domain from the gene definitions and config
- Identify what went wrong (premature convergence, too slow, local optimum, wrong mode)
- Explain the issue in plain terms
- Suggest specific parameter changes with reasoning
- Ultimately: **auto-tune parameters across multiple runs without human involvement**

Neither logs nor documentation alone is sufficient for this. The power comes from their
combination:

- **Logs** provide the data — what actually happened in this specific run
- **Documentation** provides the vocabulary and knowledge — what the data means, what causes
  it, and what to do about it

When logs use the same terminology as the documentation, and documentation explains exactly
what each log value implies, an AI agent can bridge between them fluently. This is not an
accident — it requires deliberate design.

**Practical implication:** Before any feature is considered "done", ask:
*"If an AI agent read the log of a run using this feature, would it understand what
happened and know what to recommend?"*

### 2. Full Observability

You cannot improve what you cannot see. Every internal state that affects the outcome
must be visible in the history and the log.

This means: not just best score, but average score, diversity, mutation rate, restart
events, convergence generation, per-island performance, Pareto front size, archive
coverage (MAPElites), adaptive F/CR means (DEOptimizer), population size when shrinking
(L-SHADE), sigma trajectory (CMA-ES). The log is a complete reconstruction of the run,
not a summary.

Each optimizer adds its own diagnostic fields to history:
- **GA:** `diversity`, `restarted`, `mutation_rate` (adaptive), `diagnosis`, `recommendation`
- **IslandModel:** `island_bests` per island, `diagnosis`, `recommendation`
- **MultiObjectiveGA:** `pareto_size`, `hypervolume_proxy`, `diagnosis`, `recommendation`
- **CMAESOptimizer:** `sigma`, `diagnosis`, `recommendation`
- **DEOptimizer:** `F_mean`, `CR_mean`, `pop_size`, `diagnosis`, `recommendation`
- **MAPElites:** `archive_size`, `coverage`, `diagnosis`, `recommendation`

### 3. No Magic, No Drift

The core promise is: **a gene defined as FloatRange(0, 1) will never produce a value
outside [0, 1], under any circumstances.** This holds across all mutation distributions
(Gaussian, Levy), all optimizers, all modes.

More broadly: nothing should happen that the user didn't explicitly allow. No implicit
normalization, no undocumented transformations, no silent fallbacks that change behavior.

**LLMCrossover exception:** when the LLM function returns invalid output, the fallback
to `UniformCrossover` is explicit, documented, printed to stderr, and tracked via
`fallback_count`. The user opted into the possibility of fallback by using the class.

### 4. Interchangeable Components

Selection strategies, crossover strategies, and gene types are all pluggable. A user
should be able to swap `TournamentSelection` for `RankSelection` with a one-line change
and get results that are directly comparable.

This also means: strategies must be self-describing. Their `describe()` method must
produce a log entry that fully captures their configuration — so two runs with different
strategies produce logs that a reader can directly compare.

### 5. Zero Magic Dependencies

The core library has no required dependencies. Pure Python, zero imports beyond the
standard library. Fitness functions often run in constrained environments; the library
should never be the bottleneck. Optional integrations (numpy seeding, multiprocessing,
hypothesis tests) are explicitly opt-in.

This applies to new additions too: Levy flight uses a pure-Python Chambers-Mallows-Stuck
approximation rather than scipy. CMA-ES requires numpy but raises a clear error if it
is missing.

### 6. Modular Architecture

Each optimizer lives in its own file. The library is a Python package (`evogine/`)
with one file per component:

```
evogine/
  __init__.py       — public API, re-exports everything
  genes.py          — FloatRange, IntRange, ChoiceList, GeneBuilder
  operators.py      — selection + crossover strategies
  ga.py             — GeneticAlgorithm
  island.py         — IslandModel
  multi_objective.py — MultiObjectiveGA
  cmaes.py          — CMAESOptimizer
  de.py             — DEOptimizer
  mapelites.py      — MAPElites
  analysis.py       — landscape_analysis()
  _utils.py         — shared utilities (_seed_all)
```

**Rule:** No file should exceed ~400 lines. If it does, it's a sign the abstraction
needs splitting. Monolithic files make edits error-prone, slow to navigate, and hard
to isolate for debugging.

**Backward compatibility:** `__init__.py` re-exports everything. `from evogine import X`
always works regardless of which submodule `X` lives in. Users never import from submodules
directly.

---

## Design Principles

These are the concrete rules derived from the goals above. Apply them to every decision.

### On logging

**Every parameter must appear in the log.**
If a parameter affects behavior, it must be recorded in `config`. A log should be a
complete specification of the run — someone should be able to reproduce it from the
log alone (given the same fitness function).

**Log vocabulary must match documentation vocabulary.**
If the log says `convergence_pattern: "converged_early"`, the documentation must have
a section explaining exactly what that means, why it happens, and what to try. The
log is useless if the reader has to guess what the words mean.

**Pre-compute analysis; don't make the reader do inference.**
The `analysis` section in the log contains plain-English observations. This reduces
the cognitive load on the reader (human or AI) and ensures consistent interpretation.
An AI agent reading two logs should reach the same conclusions as a human expert.

**Real values only in logs.**
In minimize mode, scores are negated internally. The log always shows real values.
An AI agent reading the log should never need to know about internal sign conventions.

**Include enough context to understand the domain.**
Gene definitions in the log (`type`, `low`, `high`, `sigma`, `options`) let a reader
understand the search space without seeing the code. A log of a stock strategy run
should be self-contained enough to understand the problem.

### On the history

**Record every meaningful event.**
Improvement events, stagnation streaks, restarts, diversity — all in the history.
A convergence curve without diversity data is half the picture. A run that restarted
should say so in the history, not just in a print statement.

**History entries must be stable.**
Once a key is in a history dict, it stays there forever. Removing or renaming history
keys is a breaking change. New keys can be added; old ones cannot be removed.

**History values are always in user units.**
A user who defined `mode='minimize'` with a loss function sees decreasing loss values
in the history — not negated internal values. The history is for the user, not the engine.

### On parameters and APIs

**Parameter names should be self-explanatory in a log context.**
A log showing `patience: 20` should be immediately clear. A log showing `p: 20` is not.
Optimize for readability of the log, not for terseness of code.

**Validation at construction, not at run time.**
Invalid parameters (bad mode, wrong objectives length) raise `ValueError` at
`__init__` time. An error that surfaces in generation 47 of a long run is far more
costly than one that surfaces immediately.

**Return values should not require the user to understand internals.**
The three-tuple `(best_individual, best_score, history)` is designed to be immediately
usable. The user does not need to understand how selection works, what internal scoring
looks like, or how minimize mode is implemented. They get what they asked for.

### On documentation

**Documentation is part of the feature.**
A feature is not complete until its documentation explains not just *what* it does
but *when* to use it, *why* it works, and *what goes wrong* if you use it incorrectly.
The goal: a user reading the docs can make good decisions without needing to experiment.

**Docs and logs share a vocabulary deliberately.**
Every convergence pattern name, every parameter name, every strategy name appears in
both the log and the documentation with the same spelling and the same meaning.
This shared vocabulary is what allows an AI agent to reason from log observations
to documented recommendations.

**Explain the intuition, not just the interface.**
"sigma controls mutation aggressiveness" is useful. "sigma: float = 0.1" is not.
Every parameter should have an explanation of what higher vs. lower values feel like
in practice, not just what the type is.

---

## The AI Co-Pilot Vision — In Detail

This is worth expanding because it is the most novel goal and the hardest to get right.

The vision has three phases:

**Phase 1 (current): Human-readable, AI-usable logs**

Logs are structured JSON with a shared vocabulary with the docs. A user can paste a
log to any AI agent and get useful analysis. The `analysis.notes` field pre-interprets
the run. This phase is implemented across all six optimizers.

**Phase 1.5 (current): Automated optimizer selection**

`landscape_analysis()` samples the fitness landscape and recommends the most suitable
optimizer class. This is a first step toward autonomous decision-making: the library
itself tells you which of its own tools to use, and explains why in plain language.

**Phase 2 (implemented): AI-driven diagnosis and steering**

Every history entry now includes `diagnosis` and `recommendation` strings — machine-readable
labels that an AI agent can act on without interpreting raw numbers. The `on_generation`
callback can return a dict of parameter overrides (the steering interface), allowing an agent
to adjust parameters mid-run based on diagnosis. `GeneticAlgorithm.from_checkpoint()` lets
an agent resume a run with different parameters. `AGENT_GUIDE.md` provides decision trees
formatted for LLM consumption.

An AI agent can now:
- Read `diagnosis` / `recommendation` from history entries directly
- Steer parameters mid-run via the callback return value
- Resume from checkpoints with overrides via `from_checkpoint()`
- Follow the decision trees in AGENT_GUIDE.md for optimizer selection and tuning
- Identify convergence patterns, diversity issues, F/CR collapse, sigma divergence
- Compare island bests and identify islands that are stuck vs. progressing

**Phase 3 (future): Autonomous parameter tuning**

An automated agent runs an optimizer, reads the log, adjusts parameters, runs again,
and iterates toward user-defined goals. For example:

```
Goal: Sharpe ratio > 1.5, max drawdown < 0.15, converge within 100 generations
```

The agent runs `landscape_analysis()`, selects `DEOptimizer`, runs it, reads the log,
sees that `F_mean` collapsed to near-zero (step size too small), increases `sigma0` or
switches strategy to `'rand1'`, runs again. The library's structured logs and shared
vocabulary make each run legible to the agent. The docs provide the knowledge base for
reasoning about changes.

**What makes this possible:**
- Logs are structured (not free text)
- Vocabulary is consistent across logs and docs
- Every observable state is logged
- The analysis section provides pre-reasoned observations
- Parameter names in logs match parameter names in code match parameter names in docs

**What would break this:**
- Logging internal values that differ from user-facing values (e.g. negated scores)
- Convergence patterns that aren't explained in the docs
- Parameters that appear in behavior but not in the log
- Documentation that uses different terminology from the logs

---

## When Adding a New Feature

Use this checklist:

- [ ] Does the feature appear fully in the JSON log config section?
- [ ] Is every new parameter name self-explanatory in a log context?
- [ ] Does the history record any new events this feature produces?
- [ ] Are the history values in user units (not internal representation)?
- [ ] Does the documentation explain *when* to use it and *when not to*?
- [ ] Does the documentation match the log vocabulary exactly?
- [ ] If an AI agent read a log of a run using this feature, would it understand what happened?
- [ ] Are the parameters validated at `__init__` time, not at run time?
- [ ] Are bounds enforced strictly — can this feature produce out-of-range values?
- [ ] Are tests written, including edge cases?
- [ ] Does the feature live in the correct module (not bolted onto a file that is already large)?
- [ ] Is the new module file under ~400 lines?
- [ ] Is the new public name re-exported from `__init__.py`?
- [ ] Is the example in `examples/` updated or a new example file written?
- [ ] Are `README.md`, `features.md`, and `PRINCIPLES.md` updated?

A feature that passes this checklist is done. A feature that skips any item is not.

---

## What This Library Is Not

**Not a neural architecture search library.** Genes are scalars and categories,
not computational graphs. The library is intentionally narrow.

**Not an automatic ML library.** It does not pick the right hyperparameters for you.
It gives you the tools to define the search space, run the search, and read the results.
`landscape_analysis()` recommends a starting point but the user remains in control.

**Not a general-purpose optimization framework.** There is no gradient-based optimization,
no Bayesian optimization, no simulated annealing. It is an evolutionary optimization
library — one family of algorithms, done well and fully documented.

**Not a research library.** It does not implement every variant of every operator from
every paper. It implements what is useful, well-documented, and testable. SHADE was chosen
over basic DE because it removes manual F/CR tuning. NSGA-III was added for 3+ objectives.
Both are well-established, not experimental.

**Not a black box.** Every parameter is logged. Every decision is documented. The user
should always be able to understand why the algorithm did what it did.
