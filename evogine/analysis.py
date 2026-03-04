from typing import Callable, Optional

from ._utils import _seed_all
from .genes import FloatRange, IntRange, ChoiceList, GeneBuilder


def landscape_analysis(
    gene_builder: GeneBuilder,
    fitness_function: Callable[[dict], float],
    n_samples: int = 500,
    seed: Optional[int] = None,
    epsilon: float = 0.01,
    n_neighbors: int = 5,
) -> dict:
    """
    Sample the fitness landscape and return a diagnostic report with an
    optimizer recommendation.

    Samples n_samples random individuals, evaluates them, then measures:
    - Ruggedness: fitness variation between nearby individuals (0=smooth, 1=jagged)
    - Neutrality: fraction of neighbor pairs with nearly identical fitness
    - Estimated modes: approximate number of local optima

    Args:
        gene_builder:     GeneBuilder defining the search space.
        fitness_function: Callable (dict -> float) to evaluate individuals.
        n_samples:        Number of random individuals to evaluate (default 500).
        seed:             Random seed for reproducibility.
        epsilon:          Threshold for neutrality. Relative to fitness range
                          (default 0.01 = 1% of range).
        n_neighbors:      Nearest neighbors used per sample for local analysis
                          (default 5).

    Returns dict with keys:
        ruggedness:              float [0,1]. 0 = smooth, 1 = maximally jagged.
        neutrality:              float [0,1]. Fraction of flat neighbor pairs.
        estimated_modes:         int. Estimated number of local optima.
        float_only:              bool. True if all genes are FloatRange.
        recommendation:          str. Suggested optimizer class name.
        reason:                  str. Plain English explanation.
        sample_best:             float. Best fitness found during sampling.
        sample_best_individual:  dict. Individual with best fitness.
    """
    _seed_all(seed)

    individuals = [gene_builder.sample() for _ in range(n_samples)]
    fitnesses   = [fitness_function(ind) for ind in individuals]

    float_only = all(isinstance(spec, FloatRange) for spec in gene_builder.specs.values())

    names = gene_builder.order

    def _to_vec(ind: dict) -> list[float]:
        vec = []
        for name in names:
            spec = gene_builder.specs[name]
            val = ind[name]
            if isinstance(spec, FloatRange):
                rng = spec.high - spec.low
                vec.append((val - spec.low) / rng if rng > 0 else 0.0)
            elif isinstance(spec, IntRange):
                rng = spec.high - spec.low
                vec.append((val - spec.low) / rng if rng > 0 else 0.0)
            else:  # ChoiceList
                opts = spec.options
                try:
                    idx = opts.index(val)
                except ValueError:
                    idx = 0
                vec.append(idx / max(1, len(opts) - 1))
        return vec

    vecs = [_to_vec(ind) for ind in individuals]

    def _dist(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    f_min = min(fitnesses)
    f_max = max(fitnesses)
    f_range = max(f_max - f_min, 1e-12)
    neutral_threshold = epsilon * f_range

    ruggedness_values: list[float] = []
    neutral_pairs = 0
    total_pairs   = 0

    for i in range(n_samples):
        dists = [(j, _dist(vecs[i], vecs[j])) for j in range(n_samples) if j != i]
        dists.sort(key=lambda x: x[1])
        neighbors = [dists[k][0] for k in range(min(n_neighbors, len(dists)))]
        for j in neighbors:
            diff = abs(fitnesses[i] - fitnesses[j])
            ruggedness_values.append(diff / f_range)
            if diff < neutral_threshold:
                neutral_pairs += 1
            total_pairs += 1

    ruggedness = min(1.0, sum(ruggedness_values) / len(ruggedness_values)) if ruggedness_values else 0.0
    neutrality  = neutral_pairs / total_pairs if total_pairs > 0 else 0.0

    # Estimate modes: count local maxima in a 20-bin fitness histogram
    n_bins = 20
    bin_size = f_range / n_bins
    bins = [0] * n_bins
    for f in fitnesses:
        idx = min(n_bins - 1, int((f - f_min) / bin_size))
        bins[idx] += 1
    estimated_modes = 1
    for k in range(1, n_bins - 1):
        if bins[k] > bins[k - 1] and bins[k] > bins[k + 1]:
            estimated_modes += 1
    estimated_modes = max(1, estimated_modes)

    best_idx = fitnesses.index(max(fitnesses))
    sample_best = fitnesses[best_idx]
    sample_best_individual = individuals[best_idx]

    if float_only and ruggedness < 0.15 and estimated_modes <= 2:
        recommendation = 'CMAESOptimizer'
        reason = (
            "Smooth landscape with a single dominant basin — CMA-ES will converge "
            "quickly by adapting its covariance to the local gradient."
        )
    elif float_only and ruggedness < 0.4 and estimated_modes <= 3:
        recommendation = 'DEOptimizer'
        reason = (
            "Moderately smooth float-only landscape — Differential Evolution (SHADE) "
            "handles mild multimodality well with adaptive F/CR parameters."
        )
    elif float_only and estimated_modes >= 4:
        recommendation = 'IslandModel'
        reason = (
            f"Estimated {estimated_modes} modes suggest a highly multimodal landscape. "
            "Island model maintains multiple sub-populations that explore different "
            "basins simultaneously."
        )
    else:
        recommendation = 'GeneticAlgorithm'
        reason = (
            "Mixed gene types or moderate ruggedness — the standard GeneticAlgorithm "
            "with tournament selection and uniform crossover is a robust default choice."
        )

    return {
        'ruggedness':             round(ruggedness, 6),
        'neutrality':             round(neutrality, 6),
        'estimated_modes':        estimated_modes,
        'float_only':             float_only,
        'recommendation':         recommendation,
        'reason':                 reason,
        'sample_best':            sample_best,
        'sample_best_individual': sample_best_individual,
    }
