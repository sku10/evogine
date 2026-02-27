"""
Property-based tests using the hypothesis library.

These verify invariants that must hold for ANY valid inputs:
- Gene values always stay within defined bounds
- Mutation never produces invalid types
- GeneBuilder always produces all keys
- GA history always has correct structure

Install hypothesis: pip install hypothesis
"""

import random
import pytest

# pytest.importorskip skips the entire module if hypothesis is not installed.
hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
assume = hypothesis.assume
st = hypothesis.strategies

from evogine import (
    FloatRange,
    IntRange,
    ChoiceList,
    GeneBuilder,
    GeneticAlgorithm,
)


# ---------------------------------------------------------------------------
# FloatRange invariants
# ---------------------------------------------------------------------------

@given(
    low=st.floats(-1000, 999, allow_nan=False, allow_infinity=False),
    high=st.floats(-999, 1000, allow_nan=False, allow_infinity=False),
    sigma=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_floatrange_sample_within_bounds(low, high, sigma):
    assume(low < high)
    spec = FloatRange(low, high, sigma)
    for _ in range(10):
        v = spec.sample()
        assert low <= v <= high, f"sample {v} outside [{low}, {high}]"


@given(
    low=st.floats(-1000, 999, allow_nan=False, allow_infinity=False),
    high=st.floats(-999, 1000, allow_nan=False, allow_infinity=False),
    sigma=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
    start=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_floatrange_mutate_stays_in_bounds(low, high, sigma, start):
    assume(low < high)
    spec = FloatRange(low, high, sigma)
    value = max(low, min(high, start))  # clamp to valid range
    for _ in range(20):
        mutated = spec.mutate(value, mutation_rate=1.0)
        assert low <= mutated <= high, f"mutated {mutated} outside [{low}, {high}]"


@given(
    low=st.floats(-100, 99, allow_nan=False, allow_infinity=False),
    high=st.floats(-99, 100, allow_nan=False, allow_infinity=False),
    sigma=st.floats(0.01, 0.5, allow_nan=False, allow_infinity=False),
    start=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_floatrange_mutation_rate_zero_no_change(low, high, sigma, start):
    assume(low < high)
    spec = FloatRange(low, high, sigma)
    value = max(low, min(high, start))
    for _ in range(10):
        assert spec.mutate(value, mutation_rate=0.0) == value


# ---------------------------------------------------------------------------
# IntRange invariants
# ---------------------------------------------------------------------------

@given(
    low=st.integers(-1000, 999),
    high=st.integers(-999, 1000),
    sigma=st.floats(0.01, 0.3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_intrange_sample_within_bounds(low, high, sigma):
    assume(low < high)
    spec = IntRange(low, high, sigma)
    for _ in range(10):
        v = spec.sample()
        assert low <= v <= high
        assert isinstance(v, int)


@given(
    low=st.integers(-1000, 999),
    high=st.integers(-999, 1000),
    sigma=st.floats(0.01, 0.3, allow_nan=False, allow_infinity=False),
    start=st.integers(-1000, 1000),
)
@settings(max_examples=300)
def test_intrange_mutate_stays_in_bounds(low, high, sigma, start):
    assume(low < high)
    spec = IntRange(low, high, sigma)
    value = max(low, min(high, start))
    for _ in range(20):
        mutated = spec.mutate(value, mutation_rate=1.0)
        assert low <= mutated <= high
        assert isinstance(mutated, int)


@given(
    low=st.integers(-100, 99),
    high=st.integers(-99, 100),
    start=st.integers(-100, 100),
)
@settings(max_examples=200)
def test_intrange_mutation_rate_zero_no_change(low, high, start):
    assume(low < high)
    spec = IntRange(low, high)
    value = max(low, min(high, start))
    for _ in range(10):
        assert spec.mutate(value, mutation_rate=0.0) == value


# ---------------------------------------------------------------------------
# ChoiceList invariants
# ---------------------------------------------------------------------------

@given(
    options=st.lists(st.text(min_size=1), min_size=1, max_size=20, unique=True),
)
@settings(max_examples=300)
def test_choicelist_sample_in_options(options):
    spec = ChoiceList(options)
    for _ in range(10):
        v = spec.sample()
        assert v in options


@given(
    options=st.lists(st.text(min_size=1), min_size=2, max_size=20, unique=True),
)
@settings(max_examples=300)
def test_choicelist_mutate_in_options(options):
    spec = ChoiceList(options)
    value = options[0]
    for _ in range(10):
        mutated = spec.mutate(value, mutation_rate=1.0)
        assert mutated in options


@given(
    options=st.lists(st.text(min_size=1), min_size=1, max_size=20, unique=True),
)
@settings(max_examples=200)
def test_choicelist_single_option_never_crashes(options):
    """Single-option ChoiceList must not crash on mutate."""
    spec = ChoiceList(options[:1])
    result = spec.mutate(options[0], mutation_rate=1.0)
    assert result == options[0]


@given(
    options=st.lists(st.text(min_size=1), min_size=2, max_size=20, unique=True),
    value_idx=st.integers(0, 19),
)
@settings(max_examples=200)
def test_choicelist_mutate_picks_different(options, value_idx):
    """Mutation always picks a different value when mutation_rate=1."""
    assume(len(options) >= 2)
    spec = ChoiceList(options)
    value = options[value_idx % len(options)]
    for _ in range(10):
        mutated = spec.mutate(value, mutation_rate=1.0)
        assert mutated != value


# ---------------------------------------------------------------------------
# GeneBuilder invariants
# ---------------------------------------------------------------------------

@given(
    n_float=st.integers(0, 5),
    n_int=st.integers(0, 5),
    n_choice=st.integers(0, 5),
)
@settings(max_examples=200)
def test_genebuilder_sample_has_all_keys(n_float, n_int, n_choice):
    assume(n_float + n_int + n_choice > 0)
    genes = GeneBuilder()
    for i in range(n_float):
        genes.add(f"f{i}", FloatRange(0.0, 1.0))
    for i in range(n_int):
        genes.add(f"i{i}", IntRange(0, 10))
    for i in range(n_choice):
        genes.add(f"c{i}", ChoiceList(["a", "b", "c"]))

    ind = genes.sample()
    assert set(ind.keys()) == set(genes.keys())


@given(
    n_float=st.integers(1, 5),
    n_int=st.integers(1, 5),
)
@settings(max_examples=200)
def test_genebuilder_mutate_preserves_keys(n_float, n_int):
    genes = GeneBuilder()
    for i in range(n_float):
        genes.add(f"f{i}", FloatRange(0.0, 1.0))
    for i in range(n_int):
        genes.add(f"i{i}", IntRange(0, 10))

    ind = genes.sample()
    mutated = genes.mutate(ind, mutation_rate=1.0)
    assert set(mutated.keys()) == set(genes.keys())


@given(n_float=st.integers(1, 4))
@settings(max_examples=200)
def test_genebuilder_mutate_stays_in_bounds(n_float):
    genes = GeneBuilder()
    bounds = {}
    for i in range(n_float):
        lo = random.uniform(-100, 0)
        hi = random.uniform(0, 100)
        genes.add(f"f{i}", FloatRange(lo, hi))
        bounds[f"f{i}"] = (lo, hi)

    ind = genes.sample()
    for _ in range(20):
        ind = genes.mutate(ind, mutation_rate=1.0)
        for name, (lo, hi) in bounds.items():
            assert lo <= ind[name] <= hi


# ---------------------------------------------------------------------------
# GA history structural invariants
# ---------------------------------------------------------------------------

HISTORY_KEYS = {
    'gen', 'best_score', 'avg_score', 'improved',
    'gens_without_improvement', 'mutation_rate', 'diversity', 'restarted',
}


@given(
    pop_size=st.integers(10, 30),
    n_gens=st.integers(1, 10),
    seed=st.integers(0, 9999),
)
@settings(max_examples=100)
def test_ga_history_always_has_required_keys(pop_size, n_gens, seed):
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))

    ga = GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=lambda ind: -ind["x"] ** 2,
        population_size=pop_size,
        generations=n_gens,
        seed=seed,
    )
    _, _, history = ga.run()

    assert len(history) == n_gens
    for h in history:
        for key in HISTORY_KEYS:
            assert key in h, f"Missing key '{key}' in history entry"


@given(
    pop_size=st.integers(10, 30),
    n_gens=st.integers(2, 15),
    seed=st.integers(0, 9999),
)
@settings(max_examples=100)
def test_ga_best_score_never_decreases(pop_size, n_gens, seed):
    """Best score should be monotonically non-decreasing."""
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))

    ga = GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=lambda ind: -ind["x"] ** 2,
        population_size=pop_size,
        generations=n_gens,
        seed=seed,
    )
    _, _, history = ga.run()

    for i in range(1, len(history)):
        assert history[i]['best_score'] >= history[i - 1]['best_score'] - 1e-9


@given(
    pop_size=st.integers(5, 20),
    n_gens=st.integers(1, 8),
    seed=st.integers(0, 9999),
)
@settings(max_examples=100)
def test_ga_gen_counter_is_sequential(pop_size, n_gens, seed):
    genes = GeneBuilder()
    genes.add("x", FloatRange(0.0, 1.0))

    ga = GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=lambda ind: ind["x"],
        population_size=pop_size,
        generations=n_gens,
        seed=seed,
    )
    _, _, history = ga.run()

    assert [h['gen'] for h in history] == list(range(1, n_gens + 1))


@given(
    pop_size=st.integers(5, 20),
    n_gens=st.integers(1, 8),
    seed=st.integers(0, 9999),
)
@settings(max_examples=100)
def test_ga_diversity_always_in_unit_interval(pop_size, n_gens, seed):
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("c", ChoiceList(["a", "b", "c"]))

    ga = GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=lambda ind: -ind["x"] ** 2,
        population_size=pop_size,
        generations=n_gens,
        seed=seed,
    )
    _, _, history = ga.run()

    for h in history:
        assert 0.0 <= h['diversity'] <= 1.0


@given(
    pop_size=st.integers(5, 20),
    n_gens=st.integers(1, 5),
    seed=st.integers(0, 9999),
)
@settings(max_examples=100)
def test_ga_best_individual_has_all_genes(pop_size, n_gens, seed):
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("n", IntRange(1, 10))
    genes.add("m", ChoiceList(["a", "b"]))

    ga = GeneticAlgorithm(
        gene_builder=genes,
        fitness_function=lambda ind: ind["x"],
        population_size=pop_size,
        generations=n_gens,
        seed=seed,
    )
    best, _, _ = ga.run()

    assert set(best.keys()) == {"x", "n", "m"}
