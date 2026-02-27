# Genetic Engine — Design Principles

This document captures the strategic vision and design principles that guide
every decision in this library. It exists so that new features, API changes,
and documentation updates all pull in the same direction.

Read this before adding a feature, changing a parameter name, or restructuring a log.

---

## Vision

**Make genetic algorithm runs fully interpretable — by humans and by AI agents.**

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
events, convergence generation, per-island performance, Pareto front size. The log is
a complete reconstruction of the run, not a summary.

### 3. No Magic, No Drift

The original motivation for building this library was that DEAP silently produced values
outside defined gene ranges. The core promise is: **a gene defined as FloatRange(0, 1)
will never produce a value outside [0, 1], under any circumstances.**

More broadly: nothing should happen that the user didn't explicitly allow. No implicit
normalization, no undocumented transformations, no silent fallbacks that change behavior.

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
the run. This phase is implemented.

**Phase 2 (near-term): AI-driven diagnosis**

An AI agent is given a log and the docs as context and can:
- Identify the convergence pattern and its cause
- Read the diversity curve and diagnose premature convergence or random walk
- Compare island bests and identify islands that are stuck vs. progressing
- Read Pareto history and see if the front is growing or stagnant
- Suggest specific parameter changes with reasoning pulled from the docs

This works today if the user provides both the log and features.md as context.
The goal is that it works reliably — the docs explain every pattern the AI will encounter.

**Phase 3 (future): Autonomous parameter tuning**

An automated agent runs the GA, reads the log, adjusts parameters, runs again,
and iterates toward user-defined goals. For example:

```
Goal: Sharpe ratio > 1.5, max drawdown < 0.15, converge within 100 generations
```

The agent runs, reads the log, sees convergence pattern, adjusts mutation rate or
switches selection strategy, runs again. The library's structured logs make each run
legible to the agent. The docs provide the knowledge base for reasoning about changes.

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

A feature that passes this checklist is done. A feature that skips any item is not.

---

## What This Library Is Not

**Not a neural architecture search library.** Genes are scalars and categories,
not computational graphs. The library is intentionally narrow.

**Not an automatic ML library.** It does not pick the right hyperparameters for you.
It gives you the tools to define the search space, run the search, and read the results.
The AI co-pilot goal is about interpretability, not automation of the user's problem.

**Not a general-purpose optimization framework.** There is no gradient-based optimization,
no Bayesian optimization, no simulated annealing. It is a genetic algorithm library.
Do one thing well.

**Not a research library.** It does not implement every variant of every operator from
every paper. It implements what is useful, well-documented, and testable.
