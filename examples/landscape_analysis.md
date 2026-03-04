# Landscape Analysis — Example Guide

`landscape_analysis()` samples your fitness landscape and recommends the optimizer most likely to solve your problem efficiently. Run it before committing to a long optimization run. It takes seconds and can save hours.

---

## When to use

Run `landscape_analysis()` any time you are unsure which optimizer to use. It is especially valuable when:

- You are starting with a new fitness function and have no prior intuition about its shape.
- A previous `GeneticAlgorithm` run stagnated and you suspect a multimodal landscape.
- You want to confirm that CMA-ES is appropriate before switching to it.
- You are building a pipeline that selects an optimizer programmatically.

The analysis evaluates a random sample of individuals and uses local neighborhood comparisons to characterize the landscape. It does not run a full optimization — the entire call typically completes in well under a second for `n_samples=500`.

---

## Minimal working example

```python
from evogine import GeneBuilder, FloatRange, landscape_analysis

genes = (
    GeneBuilder()
    .add("x", FloatRange(-5.0, 5.0))
    .add("y", FloatRange(-5.0, 5.0))
)

def fitness(ind):
    x, y = ind["x"], ind["y"]
    return -(x**2 + y**2)   # simple bowl — one global optimum

report = landscape_analysis(genes, fitness, seed=42)
print(report["recommendation"])  # CMAESOptimizer
print(report["reason"])
```

---

## What it measures

### ruggedness
A value from 0 to 1 that captures how jagged the landscape is. It is computed as the normalized variance of fitness differences between each sampled individual and its nearest neighbors in gene space. A value near 0 means nearby individuals tend to have similar fitness (smooth); a value near 1 means fitness jumps erratically across small distances.

### neutrality
The fraction of neighbor pairs whose fitness values are nearly identical, within `epsilon` of each other (where epsilon is expressed as a fraction of the observed fitness range). High neutrality indicates large flat regions — the optimizer can wander without improving, which slows convergence regardless of optimizer choice.

### estimated_modes
An estimate of how many distinct local optima exist in the landscape. It is derived from the number of sampled individuals that are local bests within their neighborhood — essentially a count of distinct fitness peaks visible in the sample. The true number of modes may be higher if `n_samples` is too small relative to the search space.

### float_only
`True` when every gene in the `GeneBuilder` is a `FloatRange`. This matters because gradient-aware optimizers like `CMAESOptimizer` and `DEOptimizer` only support continuous genes. If any gene is `IntRange` or `ChoiceList`, this flag is `False` and those optimizers are off the table.

---

## Recommendation logic

| Condition | Recommendation |
|-----------|----------------|
| `float_only` and `ruggedness < 0.15` and `estimated_modes <= 2` | `CMAESOptimizer` |
| `float_only` and `ruggedness < 0.4` and `estimated_modes <= 3` | `DEOptimizer` |
| `float_only` and `estimated_modes >= 4` | `IslandModel` |
| anything else (mixed genes or moderate-to-high ruggedness) | `GeneticAlgorithm` |

The logic prioritizes the most powerful specialized optimizer when the landscape is sufficiently well-behaved, and falls back to more general strategies as ruggedness or complexity increases.

---

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | `500` | Number of random individuals evaluated. Higher values give more reliable estimates at proportionally higher cost. |
| `seed` | `None` | Random seed for reproducible sampling. Pass an integer to get the same report on repeated calls. |
| `epsilon` | `0.01` | Neutrality threshold as a fraction of the observed fitness range. Pairs with fitness difference below `epsilon * (max_fitness - min_fitness)` count as neutral. |
| `n_neighbors` | `5` | Number of nearest neighbors used per sample when computing local ruggedness and mode estimates. Increase for denser neighborhood comparisons; decrease for speed on large `n_samples`. |

---

## Full example: analyze then instantiate

This pattern analyzes the landscape and directly instantiates the recommended optimizer without any manual branching.

```python
from evogine import (
    GeneBuilder, FloatRange, IntRange,
    landscape_analysis,
    CMAESOptimizer, IslandModel, GeneticAlgorithm,
)

# --- define your problem ---
genes = (
    GeneBuilder()
    .add("lr",      FloatRange(1e-5, 1e-1))
    .add("dropout", FloatRange(0.0, 0.5))
    .add("layers",  IntRange(1, 6))
)

def fitness(ind):
    # Replace with your actual model evaluation
    return train_and_score(
        lr=ind["lr"],
        dropout=ind["dropout"],
        layers=ind["layers"],
    )

# --- analyze ---
report = landscape_analysis(genes, fitness, n_samples=500, seed=0)

print(f"Ruggedness:      {report['ruggedness']:.3f}")
print(f"Neutrality:      {report['neutrality']:.3f}")
print(f"Estimated modes: {report['estimated_modes']}")
print(f"Float only:      {report['float_only']}")
print(f"Recommendation:  {report['recommendation']}")
print(f"Reason:          {report['reason']}")
print(f"Sample best:     {report['sample_best']:.4f}")

# --- instantiate the recommended optimizer ---
OPTIMIZERS = {
    "CMAESOptimizer": lambda: CMAESOptimizer(
        gene_builder=genes,
        fitness_function=fitness,
        population_size=50,
        generations=200,
        seed=1,
    ),
    "IslandModel": lambda: IslandModel(
        gene_builder=genes,
        fitness_function=fitness,
        n_islands=4,
        population_size=50,
        generations=200,
        seed=1,
    ),
    "GeneticAlgorithm": lambda: GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=fitness,
        population_size=100,
        generations=200,
        seed=1,
    ),
}

optimizer = OPTIMIZERS.get(
    report["recommendation"],
    OPTIMIZERS["GeneticAlgorithm"],   # safe fallback
)()

result = optimizer.run()
print(f"\nBest score: {result['best_score']:.4f}")
print(f"Best params: {result['best_individual']}")
```

Note: `DEOptimizer` is included in the recommendation logic but omitted from the dispatch table above for brevity — add it alongside the others if your project uses it.

---

## Interpreting the report

**High ruggedness (> 0.4).** The fitness surface changes sharply over short distances. Gradient-following strategies will chase local noise rather than global structure. Prefer `IslandModel` (diverse parallel search) or `GeneticAlgorithm`. If you control the fitness function, consider smoothing it — for example, averaging over multiple data samples rather than evaluating on a single one.

**High neutrality (> 0.5).** Large portions of the space return nearly identical fitness scores. The optimizer has no gradient signal in these regions and will move randomly until it stumbles onto a slope. This is common with thresholded or discretized fitness functions. Consider reformulating the fitness function to return more informative values across the neutral region.

**estimated_modes interpretation.**

| Value | Landscape character |
|-------|---------------------|
| 1–2   | Effectively unimodal — gradient methods excel |
| 3–4   | Mildly multimodal — a few competing peaks |
| 5+    | Richly multimodal — population diversity is critical |

High `estimated_modes` is the primary signal for choosing `IslandModel` over a single-population optimizer. A single population will converge on whichever peak it reaches first; islands independently explore multiple peaks before sharing their best finds.

---

## Limitations

- **n_samples=500 is cheap, not exhaustive.** A landscape with many sharp narrow peaks may show low ruggedness in a sparse random sample simply because the sample rarely lands on or near a peak. The estimate improves with larger `n_samples` but cost scales linearly.

- **A landscape can look smooth in random samples but have sharp ridges.** If your fitness function has constraints, steep cliffs, or discontinuities only near the optimum, the random sample may miss them entirely. The analysis characterizes the bulk of the space, not the region near the solution.

- **estimated_modes is a lower bound.** The sample-based mode detection can only find peaks that happen to be represented in the sample. In high-dimensional spaces the true number of modes is frequently larger than reported.

- **Use it as a guide, not a guarantee.** The recommendation is a starting point. If the recommended optimizer underperforms, try the next tier — for example, if `CMAESOptimizer` is recommended but stagnates, step up to `IslandModel`.

---

## See also

- [`GeneticAlgorithm` example guide](genetic_algorithm.md) — general-purpose optimizer for mixed gene types
- [`IslandModel` example guide](island_model.md) — parallel islands for multimodal landscapes
- `features.md` — full parameter reference for all optimizers
- `PRINCIPLES.md` — design rationale and library philosophy
