# evogine Agent Guide

Machine-readable decision trees for LLM agents using evogine.

## 1. Which Optimizer?

```
IF n_objectives > 1:
    USE MultiObjectiveGA
    IF n_objectives == 2: algorithm='nsga2'
    IF n_objectives >= 3: algorithm='nsga3'

ELSE IF goal == "quality_diversity":
    USE MAPElites
    REQUIRES: behavior_fn, grid_shape

ELSE IF all genes are FloatRange AND n_genes >= 2:
    IF landscape is unimodal or unknown:
        USE CMAESOptimizer  (fastest convergence for continuous problems)
    IF landscape is multimodal:
        USE IslandModel  (parallel exploration of multiple basins)
    IF landscape is noisy or high-dimensional (>50 genes):
        USE DEOptimizer  (robust, parameter-free adaptation via SHADE)

ELSE (mixed gene types: IntRange, ChoiceList, FloatRange):
    USE GeneticAlgorithm
    IF multimodal: USE IslandModel
```

Use `landscape_analysis()` to auto-detect landscape characteristics:
```python
from evogine import landscape_analysis
result = landscape_analysis(gene_builder, fitness_fn, n_samples=200)
print(result['recommendation'])  # e.g. "CMAESOptimizer"
```

## 2. Reading History Logs

Every optimizer returns `history` — a list of per-generation dicts.

### Common fields (all optimizers)

| Field | Type | Meaning |
|-------|------|---------|
| `gen` | int | Generation number (1-indexed) |
| `best_score` | float | All-time best score (non-decreasing) |
| `diagnosis` | str | Machine-readable status label |
| `recommendation` | str | Suggested action for the agent |

### Fields shared by most optimizers (not MAPElites or MultiObjectiveGA)

| Field | Type | Present in |
|-------|------|------------|
| `avg_score` | float | GA, Island, DE, CMA-ES |
| `improved` | bool | GA, Island, DE, CMA-ES, MultiObjectiveGA |
| `gens_without_improvement` | int | GA, Island, DE, CMA-ES, MultiObjectiveGA |

### Optimizer-specific fields

**GeneticAlgorithm**: `mutation_rate`, `diversity`, `restarted`
**IslandModel**: `island_bests` (list of per-island best scores)
**DEOptimizer**: `F_mean`, `CR_mean`, `pop_size`, `stop_reason`
**CMAESOptimizer**: `sigma`, `stop_reason`
**MultiObjectiveGA**: `pareto_size`, `hypervolume_proxy`
**MAPElites**: `archive_size`, `coverage`

### Concerning values

```
IF diversity < 0.05:        population has converged, risk of local optimum
IF diversity > 0.8 AND not improving:  search is random, not converging
IF F_mean < 0.1:            DE step size collapsed
IF CR_mean > 0.95:          DE crossover saturated
IF sigma < tolx * 10:       CMA-ES step size near zero
IF sigma > 1.0:             CMA-ES diverging
IF pareto_size == pop_size: all solutions non-dominated (front saturated)
IF coverage < 0.05 at >50% gens: MAP-Elites not finding diverse solutions
```

## 3. Diagnosis Labels

### GeneticAlgorithm
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `low_diversity` | Population converged (<5% spread) | Increase mutation_rate or enable restart_after |
| `random_walk` | High diversity but no improvement for 5+ gens | Decrease mutation_rate or increase population_size |
| `population_restarted` | Fresh individuals injected | Restart injected fresh individuals |
| `improving` | Score improved this generation | No changes needed |
| `approaching_stagnation` | Over 50% of patience used | Consider increasing mutation_rate |
| `stable` | Normal operation, no concerns | No changes needed |

### IslandModel
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `islands_converged` | All island bests within 1% of each other | Increase migration_interval or try star topology |
| `improving` | Score improved | No changes needed |
| `stagnating` | No improvement for 10+ gens | Increase migration_size or mutation_rate |
| `stable` | Normal operation | No changes needed |

### DEOptimizer
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `F_collapsed` | Mutation factor < 0.1 | Increase population_size or try rand1 strategy |
| `CR_saturated` | Crossover rate > 0.95 | Problem may be separable; try rand1 |
| `improving` | Score improved | No changes needed |
| `stagnating` | No improvement for 10+ gens | Increase population_size |
| `stable` | Normal operation | No changes needed |

### CMAESOptimizer
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `sigma_collapsed` | Step size near zero | Optimum found or increase sigma0 |
| `sigma_diverging` | Step size > 1.0 | Landscape may be deceptive |
| `improving` | Score improved | No changes needed |
| `stable` | Normal operation | No changes needed |

### MultiObjectiveGA
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `front_saturated` | All solutions non-dominated | Increase population_size |
| `single_optimum` | Only 1 Pareto solution | Check objective independence |
| `improving` | Hypervolume improved | No changes needed |
| `stable` | Normal operation | No changes needed |

### MAPElites
| Diagnosis | Meaning | Recommendation |
|-----------|---------|----------------|
| `well_covered` | >80% archive coverage | Consider finer grid_shape |
| `poor_coverage` | <5% coverage past halfway | Increase mutation_rate or initial_population |
| `archive_stagnant` | No new cells for 50+ gens | Increase mutation_rate |
| `exploring` | Normal operation | No changes needed |

## 4. Parameter Tuning Rules

### Steering (runtime adjustment)

`on_generation` callbacks can return a dict to adjust parameters mid-run:

```python
def my_callback(gen, best_score, avg_score, best_individual):
    # Read the latest history entry or use callback args
    if some_condition:
        return {'mutation_rate': 0.3}  # apply override
    return None  # no change

ga = GeneticAlgorithm(..., on_generation=my_callback)
```

Steerable parameters per optimizer:
- **GeneticAlgorithm**: `mutation_rate`, `crossover_rate`, `elitism`, `patience`
- **IslandModel**: `mutation_rate`, `crossover_rate`, `elitism`, `migration_interval`, `migration_size`
- **DEOptimizer**: `strategy` (`'current_to_best'` or `'rand1'`), `patience`, `min_delta`
- **CMAESOptimizer**: `patience`, `min_delta`, `tolx`, `tolfun`
- **MultiObjectiveGA**: `mutation_rate`, `crossover_rate`, `patience`
- **MAPElites**: `mutation_rate`

### IF/THEN tuning patterns

```
IF diagnosis == "low_diversity":
    SET mutation_rate = min(current * 2, 0.5)
    OR SET restart_after = 10  (requires new run)

IF diagnosis == "random_walk":
    SET mutation_rate = max(current * 0.5, 0.01)

IF diagnosis == "approaching_stagnation":
    SET mutation_rate = min(current * 1.5, 0.4)

IF diagnosis == "islands_converged":
    SET migration_interval = current * 2

IF diagnosis == "F_collapsed":
    NEXT RUN: increase population_size by 50%
    OR steer strategy = "rand1"

IF diagnosis == "sigma_collapsed" AND score is unsatisfactory:
    NEXT RUN: increase sigma0 (e.g. 0.3 -> 0.5)

IF diagnosis == "front_saturated":
    NEXT RUN: increase population_size by 50%

IF diagnosis == "poor_coverage":
    SET mutation_rate = min(current * 2, 0.5)
```

## 5. Multi-Run Workflow

Recommended agent loop:

```
1. SELECT optimizer using decision tree (Section 1)
2. RUN with default parameters
3. READ final history entries
4. CHECK diagnosis field of last 5 entries
5. IF diagnosis indicates a problem:
     a. APPLY parameter adjustment (Section 4)
     b. RE-RUN (or steer mid-run via callback)
6. IF diagnosis == "improving" in final entries:
     INCREASE generations and re-run
7. IF diagnosis == "stable" and score is satisfactory:
     STOP — solution found
8. IF 3+ runs show no progress:
     SWITCH optimizer (e.g. GA -> IslandModel, or CMA-ES -> DE)
```

### Checkpoint resume workflow

```python
from evogine import GeneticAlgorithm, GeneBuilder, FloatRange

# First run
ga = GeneticAlgorithm(..., checkpoint_path='run.json', generations=50)
best, score, history = ga.run()

# Agent reads history, decides to continue with different params
ga2 = GeneticAlgorithm.from_checkpoint(
    'run.json',
    gene_builder=gb,
    fitness_function=fn,
    generations=100,       # override: run longer
    mutation_rate=0.2,     # override: explore more
)
best2, score2, history2 = ga2.run()
```

## 6. Quick Reference: Default Parameters

| Parameter | GA | Island | DE | CMA-ES | MO-GA | MAP-Elites |
|-----------|-----|--------|-----|--------|-------|------------|
| population_size | 100 | 50/island | 50 | auto | 100 | 200 (init) |
| generations | 50 | 100 | 200 | 200 | 50 | 1000 |
| mutation_rate | 0.1 | 0.1 | adaptive | N/A | 0.1 | 0.1 |
| crossover_rate | 0.5 | 0.5 | adaptive | N/A | 0.5 | N/A |
| patience | None | None | None | None | None | N/A |
