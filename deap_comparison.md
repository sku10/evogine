# Why Not DEAP? — A Comparison

This document summarizes known problems with DEAP (Distributed Evolutionary Algorithms in Python)
gathered from GitHub issues, academic papers, and community discussions. It explains why this
library takes a different approach.

---

## Maintenance Status

DEAP is not abandoned, but effectively stalled:
- **237 open issues**, many unresolved for years
- **47 open pull requests**, oldest from December 2020 (a crossover function waiting 4+ years for review)
- **DEAP 2.0** has been a GitHub milestone since ~2014, currently at 27% completion with no due date
- Longest recent gap: **10 months without commits** (July 2023 – May 2024)
- Latest release: 1.4.3, May 2025 — primarily compatibility patches

---

## Core Design Problems

### 1. Global mutable state via `creator`

```python
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
```

DEAP dynamically injects classes into the `deap.creator` module namespace at runtime.

**Consequences:**
- Calling `creator.create()` twice with the same name silently overwrites the previous class
- Workers in Spark, `ipyparallel`, or any distributed system fail with:
  `"Can't get attribute 'Individual' on <module 'deap.creator'>"`
  because dynamically-created classes don't exist in worker process memory
- All setup code must live at module scope — impossible to encapsulate in functions or classes
- Fundamentally incompatible with distributed computing (architectural, not fixable with a patch)

**This library:** individuals are plain Python `dict`s. No global state. No dynamic class injection.

---

### 2. Fitness weights: misleading API

DEAP's `weights=(1.0, -1.0)` does not perform a weighted sum. The values only control sign of
optimization. A user summarized it well:

> *"DEAP asking for and using 'weights' despite them having no real impact aside from their sign —
> essentially requesting boolean arguments in float form."*

Additional problems:
- Reassigning weights recalculates fitness values in a counterintuitive direction
- Mismatched weight/value counts are silently truncated
- Under continuous fitness (the common case), secondary objectives are **never used** because two
  primary scores are practically never exactly equal under lexicographic comparison

**This library:** single scalar fitness, maximization by default. Simple and unambiguous.

---

### 3. Reproducibility broken in Genetic Programming

DEAP's `gp.cxOnePoint` uses a Python `set()` for intersection, which orders elements by memory
address hash — different every run. Result: even with `random.seed(1)`, GP runs produce different
results across invocations. The fix (sort before choosing) is trivial but has remained unpatched
for years (issues #190, #765).

Separately, DEAP uses the global `random` module singleton, making it impossible to run multiple
independent instances in parallel with reproducible results (issue #75 — open since early project
history, never resolved).

**This library:** `seed` parameter calls `random.seed(seed)` at init. Same seed = identical run.

---

### 4. Multiprocessing: pickling landmines

- Lambda functions can't be pickled → breaks multiprocessing entirely if used anywhere in setup
- On Windows, `creator.create()` must go *outside* `if __name__ == '__main__':` — opposite of
  standard Python multiprocessing practice
- Workers often fail to deserialize DEAP individuals due to the dynamic class issue above

**This library:** fitness function receives a plain `dict`. Any picklable module-level function works.

---

### 5. No built-in early stopping

Issue #271 shows a user re-implementing a full algorithm variant just to add stagnation-based
stopping. Maintainer response: *"DEAP 2.0 will handle this."* (2.0 is still unreleased.)

**This library:** `patience` and `min_delta` parameters built in.

---

### 6. No callback API

Issue #107 requested per-generation callbacks for monitoring, visualization, or passing
generation-specific data into the fitness function. Maintainers closed it, saying callbacks
weren't necessary. A user responded:

> *"Advising people to copy, paste and change is basically against what software development
> was meant for. Keras supports this, and I don't see why DEAP shouldn't."*

**This library:** `on_generation` callback is on the roadmap (see ideas.md).

---

### 7. No mixed-type genes

Issue #755 (2025): there is no mutation operator for individuals with heterogeneous gene types
(mix of float, int, categorical). Users must implement all of this manually.

**This library:** `FloatRange`, `IntRange`, and `ChoiceList` all coexist in the same individual
via `GeneBuilder`. Adding new gene types requires only subclassing `GeneSpec`.

---

### 8. No structured logging or observability

DEAP's `verbose=True` prints to stdout only. No `logging` integration, no JSON output, no
history object, no machine-readable output for later analysis (issue #750).

**This library:** `log_path` produces a structured JSON log with config, per-generation history,
and an `analysis` block with plain-English notes interpretable by AI agents or humans.

---

### 9. Values drift outside defined bounds

The original motivation for this library. DEAP's mutation operators can produce values outside
user-specified gene ranges, and controlling this is not straightforward.

**This library:** every `GeneSpec` strictly clips mutations to its defined range. Bounds are
enforced by design, not by the user remembering to add clipping.

---

### 10. Stale fitness values: silent error

DEAP requires users to manually `del individual.fitness.values` after mutation/crossover.
Forgetting this causes individuals to be compared against stale scores — silent logical error,
no warning, no exception.

**This library:** fitness is computed fresh every generation. No stale state possible.

---

### 11. Excessive boilerplate

Minimal DEAP setup before any EA logic:

```python
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
```

Equivalent setup in this library:
```python
genes = GeneBuilder()
genes.add("x", FloatRange(0, 10))
ga = GeneticAlgorithm(gene_builder=genes, fitness_function=my_fn)
```

---

### 12. Python compatibility: slow to adapt

DEAP was originally Python 2.7 code converted with `2to3`. When Python 3.10 removed `lib_2to3`,
DEAP's build broke entirely for many users. The only workarounds were downgrading `setuptools`
to ≤57 or using a community fork (`deap-er`). Now fixed in 1.4.x, but illustrated the project's
slow response to ecosystem changes.

---

## What DEAP Does Well (That This Library Doesn't Yet)

To be fair:
- **Genetic Programming (GP)** — tree-based program evolution. Not in scope here.
- **NSGA-II multi-objective** — mature implementation (with bugs, but present). Multi-objective
  is on this library's roadmap.
- **Wide operator library** — many crossover and selection operators available. This library
  is growing its selection strategy options.
- **Academic adoption** — widely cited in papers, many examples and tutorials exist.

---

## Summary

| Feature | DEAP | This library |
|---|---|---|
| Named genes (not anonymous lists) | No | Yes |
| Bounds-safe mutation | No (user must clip) | Yes (enforced) |
| Reproducible seed | Broken in GP | Works |
| Multiprocessing safety | Fragile | Plain dict, picklable |
| Early stopping | No (DIY) | Built in |
| Generation history | No | Built in |
| Structured JSON logging | No | Built in |
| Callback hooks | No (rejected) | Roadmap |
| Mixed gene types | No | Yes |
| Setup boilerplate | ~30–50 lines | ~5 lines |
| Active maintenance | Minimal | Active |
| Global state | Yes (creator) | No |
| DEAP 2.0 | ~27% done, stalled | — |
