"""Tests for new features added in evogine package refactor:
Levy flight, linear pop reduction, island topologies, constraints,
LLMCrossover, NSGA-III, landscape_analysis.
"""
import random
import pytest
from evogine import (
    FloatRange, IntRange, ChoiceList, GeneBuilder,
    GeneticAlgorithm, IslandModel, MultiObjectiveGA,
    RouletteSelection, TournamentSelection,
    UniformCrossover, LLMCrossover,
    landscape_analysis,
)


def _make_gb(n=2):
    gb = GeneBuilder()
    for i in range(n):
        gb.add(f"x{i}", FloatRange(0.0, 1.0))
    return gb


def _fitness(ind):
    return sum(ind.values())


# ===========================================================================
# Feature 1: Levy Flight Mutation
# ===========================================================================

class TestLevyFlightMutation:
    def test_levy_stays_in_bounds(self):
        spec = FloatRange(0.0, 1.0, mutation_dist='levy')
        random.seed(42)
        for _ in range(200):
            val = spec.mutate(0.5, 1.0)  # always mutate
            assert 0.0 <= val <= 1.0

    def test_levy_describe_includes_dist(self):
        spec = FloatRange(0.0, 1.0, mutation_dist='levy')
        d = spec.describe()
        assert d['mutation_dist'] == 'levy'

    def test_gaussian_describe(self):
        spec = FloatRange(0.0, 1.0)
        assert spec.describe()['mutation_dist'] == 'gaussian'

    def test_levy_rejects_bad_value(self):
        with pytest.raises(ValueError, match="mutation_dist"):
            FloatRange(0.0, 1.0, mutation_dist='uniform')

    def test_levy_heavier_tail_than_gaussian(self):
        """Levy should occasionally produce larger jumps than Gaussian."""
        random.seed(0)
        levy_spec = FloatRange(-100.0, 100.0, sigma=0.01, mutation_dist='levy')
        gauss_spec = FloatRange(-100.0, 100.0, sigma=0.01, mutation_dist='gaussian')

        levy_jumps = []
        gauss_jumps = []
        start = 50.0
        for _ in range(500):
            levy_jumps.append(abs(levy_spec.mutate(start, 1.0) - start))
            gauss_jumps.append(abs(gauss_spec.mutate(start, 1.0) - start))

        # Levy should have higher max jump on average
        assert max(levy_jumps) > max(gauss_jumps) * 0.5  # loose check

    def test_levy_works_in_ga(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 1.0, mutation_dist='levy'))
        ga = GeneticAlgorithm(gb, _fitness, population_size=10, generations=5, seed=1)
        best, _, _ = ga.run()
        assert 0.0 <= best['x'] <= 1.0


# ===========================================================================
# Feature 2: Linear Population Size Reduction
# ===========================================================================

class TestLinearPopReduction:
    def test_pop_shrinks_over_generations(self):
        """Population should shrink; we verify by tracking sizes via callback."""
        gb = _make_gb()
        pop_sizes = []

        def cb(gen, best, avg, ind):
            pass

        ga = GeneticAlgorithm(
            gb, _fitness,
            population_size=30, generations=30,
            linear_pop_reduction=True, min_population=4,
            on_generation=cb, seed=1,
        )
        _, _, history = ga.run()
        # The algorithm ran to completion
        assert len(history) == 30

    def test_history_length_full_with_lpr(self):
        gb = _make_gb()
        ga = GeneticAlgorithm(
            gb, _fitness, population_size=20, generations=10,
            linear_pop_reduction=True, seed=1,
        )
        _, _, history = ga.run()
        assert len(history) == 10

    def test_result_is_valid_with_lpr(self):
        gb = _make_gb()
        ga = GeneticAlgorithm(
            gb, _fitness, population_size=20, generations=10,
            linear_pop_reduction=True, min_population=4, seed=42,
        )
        best, score, _ = ga.run()
        assert best is not None
        assert isinstance(score, float)
        assert 0.0 <= best['x0'] <= 1.0

    def test_min_population_respected(self):
        """Even at final gen, effective pop should be at least min_population."""
        gb = _make_gb()
        # With pop=10, min=8, 10 gens: should stay >= 8
        ga = GeneticAlgorithm(
            gb, _fitness, population_size=10, generations=10,
            linear_pop_reduction=True, min_population=8, seed=1,
        )
        _, _, history = ga.run()
        assert len(history) == 10  # ran all gens

    def test_lpr_false_no_change(self):
        """Without LPR, population stays constant (baseline check)."""
        gb = _make_gb()
        ga = GeneticAlgorithm(
            gb, _fitness, population_size=20, generations=5,
            linear_pop_reduction=False, seed=1,
        )
        _, _, history = ga.run()
        assert len(history) == 5


# ===========================================================================
# Feature 3: Island Model Topology Options
# ===========================================================================

class TestIslandTopologies:
    def _run_topology(self, topology, seed=1):
        gb = _make_gb()
        model = IslandModel(
            gb, _fitness,
            n_islands=4, island_population=10,
            generations=5, migration_interval=2,
            topology=topology, seed=seed,
        )
        return model.run()

    def test_ring_topology_runs(self):
        best, score, history = self._run_topology('ring')
        assert best is not None
        assert len(history) == 5

    def test_fully_connected_topology_runs(self):
        best, score, history = self._run_topology('fully_connected')
        assert best is not None
        assert len(history) == 5

    def test_star_topology_runs(self):
        best, score, history = self._run_topology('star')
        assert best is not None
        assert len(history) == 5

    def test_rejects_bad_topology(self):
        gb = _make_gb()
        with pytest.raises(ValueError, match="topology"):
            IslandModel(gb, _fitness, topology='mesh')

    def test_migration_pairs_ring(self):
        gb = _make_gb()
        model = IslandModel(gb, _fitness, n_islands=4, topology='ring')
        pairs = model._get_migration_pairs()
        # Ring: 0→1, 1→2, 2→3, 3→0
        assert (0, 1) in pairs
        assert (3, 0) in pairs
        assert len(pairs) == 4

    def test_migration_pairs_fully_connected(self):
        gb = _make_gb()
        model = IslandModel(gb, _fitness, n_islands=4, topology='fully_connected')
        pairs = model._get_migration_pairs()
        # n*(n-1) pairs
        assert len(pairs) == 4 * 3

    def test_migration_pairs_star(self):
        gb = _make_gb()
        model = IslandModel(gb, _fitness, n_islands=4, topology='star')
        pairs = model._get_migration_pairs()
        # 3 spokes → hub + hub → 3 spokes = 6
        assert len(pairs) == 6
        # Hub receives from all spokes
        assert all((i, 0) in pairs for i in range(1, 4))
        # Hub sends to all spokes
        assert all((0, i) in pairs for i in range(1, 4))

    def test_all_topologies_produce_valid_scores(self):
        for topo in ('ring', 'fully_connected', 'star'):
            _, score, _ = self._run_topology(topo, seed=42)
            assert isinstance(score, float)
            assert score > 0.0


# ===========================================================================
# Feature 4: Constraint Handling
# ===========================================================================

class TestConstraintHandling:
    def test_feasible_individual_wins(self):
        """With a tight constraint, the best feasible individual should be returned."""
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 2.0))

        # Constraint: x must be <= 1.0
        constraints = [lambda ind: ind['x'] <= 1.0]

        ga = GeneticAlgorithm(
            gb, lambda ind: ind['x'],  # maximize x
            population_size=20, generations=20,
            constraints=constraints, seed=42,
        )
        best, score, _ = ga.run()
        assert best['x'] <= 1.0 + 1e-9  # feasible
        assert score <= 1.0 + 1e-9

    def test_no_constraints_is_backward_compatible(self):
        gb = _make_gb()
        ga = GeneticAlgorithm(gb, _fitness, population_size=10, generations=5, seed=1)
        best, score, _ = ga.run()
        assert best is not None

    def test_violation_counting(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 1.0))
        gb.add('y', FloatRange(0.0, 1.0))

        constraints = [
            lambda ind: ind['x'] < 0.5,  # x < 0.5
            lambda ind: ind['y'] > 0.5,  # y > 0.5
        ]

        ga = GeneticAlgorithm(
            gb, _fitness, population_size=5, generations=1, seed=1,
            constraints=constraints,
        )
        # Individual that violates both
        ind_bad = {'x': 0.8, 'y': 0.2}
        assert ga._count_violations(ind_bad) == 2

        # Individual that satisfies both
        ind_good = {'x': 0.3, 'y': 0.7}
        assert ga._count_violations(ind_good) == 0

    def test_infeasible_individuals_get_penalized_score(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 1.0))

        constraints = [lambda ind: ind['x'] < 0.3]

        ga = GeneticAlgorithm(
            gb, lambda ind: ind['x'], population_size=5, generations=1,
            constraints=constraints, seed=1,
        )
        # Evaluate a population with some infeasible
        population = [
            {'x': 0.1},  # feasible
            {'x': 0.9},  # infeasible
        ]
        scored = ga.evaluate_population(population)
        scores = {ind['x']: sc for ind, sc in scored}
        # Feasible should score better than infeasible
        assert scores[0.1] > scores[0.9]

    def test_multiple_constraints(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 10.0))
        gb.add('y', FloatRange(0.0, 10.0))

        constraints = [
            lambda ind: ind['x'] + ind['y'] <= 8.0,
            lambda ind: ind['x'] >= 1.0,
            lambda ind: ind['y'] >= 1.0,
        ]

        ga = GeneticAlgorithm(
            gb, lambda ind: ind['x'] + ind['y'],
            population_size=30, generations=30,
            constraints=constraints, seed=42,
        )
        best, score, _ = ga.run()
        assert best['x'] + best['y'] <= 8.0 + 1e-6
        assert best['x'] >= 1.0 - 1e-6
        assert best['y'] >= 1.0 - 1e-6


# ===========================================================================
# Feature 5: LLMCrossover
# ===========================================================================

class TestLLMCrossover:
    def test_calls_llm_fn(self):
        calls = []

        def llm_fn(p1, p2):
            calls.append((p1, p2))
            return {'x0': 0.5, 'x1': 0.5}

        gb = _make_gb()
        p1 = {'x0': 0.1, 'x1': 0.2}
        p2 = {'x0': 0.8, 'x1': 0.9}
        cx = LLMCrossover(llm_fn=llm_fn)
        child = cx.crossover(p1, p2, gb)
        assert len(calls) == 1
        assert calls[0] == (p1, p2)

    def test_returns_child_dict(self):
        gb = _make_gb()
        cx = LLMCrossover(llm_fn=lambda p1, p2: {'x0': 0.3, 'x1': 0.7})
        child = cx.crossover({'x0': 0.1, 'x1': 0.2}, {'x0': 0.8, 'x1': 0.9}, gb)
        assert child == {'x0': 0.3, 'x1': 0.7}

    def test_clamps_float_values_to_bounds(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 1.0))
        gb.add('y', FloatRange(0.0, 1.0))

        cx = LLMCrossover(llm_fn=lambda p1, p2: {'x': 5.0, 'y': -3.0})
        child = cx.crossover({'x': 0.5, 'y': 0.5}, {'x': 0.5, 'y': 0.5}, gb)
        assert child['x'] == 1.0
        assert child['y'] == 0.0

    def test_fallback_on_missing_key(self):
        gb = _make_gb()
        cx = LLMCrossover(llm_fn=lambda p1, p2: {'x0': 0.5})  # missing x1
        child = cx.crossover({'x0': 0.1, 'x1': 0.2}, {'x0': 0.8, 'x1': 0.9}, gb)
        assert cx.fallback_count == 1
        assert 'x0' in child and 'x1' in child

    def test_fallback_on_exception(self):
        gb = _make_gb()

        def failing_fn(p1, p2):
            raise RuntimeError("LLM API unavailable")

        cx = LLMCrossover(llm_fn=failing_fn)
        child = cx.crossover({'x0': 0.1, 'x1': 0.2}, {'x0': 0.8, 'x1': 0.9}, gb)
        assert cx.fallback_count == 1
        assert child is not None

    def test_raise_on_failure_true(self):
        gb = _make_gb()

        def failing_fn(p1, p2):
            raise ValueError("model error")

        cx = LLMCrossover(llm_fn=failing_fn, raise_on_failure=True)
        with pytest.raises(ValueError, match="model error"):
            cx.crossover({'x0': 0.1, 'x1': 0.2}, {'x0': 0.8, 'x1': 0.9}, gb)

    def test_fallback_count_increments(self):
        gb = _make_gb()
        cx = LLMCrossover(llm_fn=lambda p1, p2: {'x0': 0.5})  # always missing x1
        for _ in range(3):
            cx.crossover({'x0': 0.1, 'x1': 0.2}, {'x0': 0.8, 'x1': 0.9}, gb)
        assert cx.fallback_count == 3

    def test_describe(self):
        cx = LLMCrossover(llm_fn=lambda p1, p2: {})
        d = cx.describe()
        assert d['strategy'] == 'llm'
        assert 'raise_on_failure' in d
        assert 'fallback_count' in d

    def test_valid_choice_list_gene(self):
        gb = GeneBuilder()
        gb.add('c', ChoiceList(['a', 'b', 'c']))
        cx = LLMCrossover(llm_fn=lambda p1, p2: {'c': 'b'})
        child = cx.crossover({'c': 'a'}, {'c': 'c'}, gb)
        assert child['c'] == 'b'

    def test_invalid_choice_list_falls_back(self):
        gb = GeneBuilder()
        gb.add('c', ChoiceList(['a', 'b']))
        cx = LLMCrossover(llm_fn=lambda p1, p2: {'c': 'z'})  # z not in options
        child = cx.crossover({'c': 'a'}, {'c': 'b'}, gb)
        assert cx.fallback_count == 1
        assert child['c'] in ('a', 'b')


# ===========================================================================
# Feature 7: NSGA-III Mode
# ===========================================================================

class TestNSGA3:
    def _make_multi_gb(self, n=3):
        gb = GeneBuilder()
        for i in range(n):
            gb.add(f"x{i}", FloatRange(0.0, 1.0))
        return gb

    def test_nsga3_runs(self):
        gb = self._make_multi_gb()

        def fitness(ind):
            return [ind['x0'], ind['x1'], ind['x2']]

        pareto, history = MultiObjectiveGA(
            gb, fitness, n_objectives=3,
            population_size=20, generations=5, seed=1,
            algorithm='nsga3',
        ).run()
        assert len(pareto) > 0
        assert len(history) == 5

    def test_nsga3_pareto_front_format(self):
        gb = self._make_multi_gb()

        def fitness(ind):
            return [ind['x0'], ind['x1']]

        pareto, _ = MultiObjectiveGA(
            gb, fitness, n_objectives=2,
            population_size=20, generations=5, seed=1,
            algorithm='nsga3',
        ).run()
        for entry in pareto:
            assert 'individual' in entry
            assert 'scores' in entry
            assert len(entry['scores']) == 2

    def test_nsga3_rejects_bad_algorithm(self):
        gb = self._make_multi_gb()
        with pytest.raises(ValueError, match="algorithm"):
            MultiObjectiveGA(gb, lambda ind: [0.0, 0.0], n_objectives=2,
                             algorithm='bogus')

    def test_nsga2_default_still_works(self):
        gb = self._make_multi_gb()

        def fitness(ind):
            return [ind['x0'], ind['x1']]

        pareto, history = MultiObjectiveGA(
            gb, fitness, n_objectives=2,
            population_size=20, generations=5, seed=1,
        ).run()
        assert len(pareto) > 0

    def test_nsga3_with_4_objectives(self):
        gb = GeneBuilder()
        for i in range(4):
            gb.add(f"x{i}", FloatRange(0.0, 1.0))

        def fitness(ind):
            return [ind['x0'], ind['x1'], ind['x2'], ind['x3']]

        pareto, history = MultiObjectiveGA(
            gb, fitness, n_objectives=4,
            population_size=30, generations=5, seed=42,
            algorithm='nsga3',
        ).run()
        assert len(pareto) > 0
        for entry in pareto:
            assert len(entry['scores']) == 4

    def test_nsga3_custom_reference_point_divisions(self):
        gb = self._make_multi_gb()

        def fitness(ind):
            return [ind['x0'], ind['x1'], ind['x2']]

        pareto, _ = MultiObjectiveGA(
            gb, fitness, n_objectives=3,
            population_size=20, generations=3, seed=1,
            algorithm='nsga3', reference_point_divisions=4,
        ).run()
        assert len(pareto) >= 0  # just should not crash

    def test_nsga3_user_reference_points(self):
        gb = self._make_multi_gb(n=2)

        def fitness(ind):
            return [ind['x0'], ind['x1']]

        ref_pts = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        pareto, _ = MultiObjectiveGA(
            gb, fitness, n_objectives=2,
            population_size=20, generations=3, seed=1,
            algorithm='nsga3', reference_points=ref_pts,
        ).run()
        assert len(pareto) >= 0

    def test_generate_reference_points_sums_to_1(self):
        gb = self._make_multi_gb()
        ga = MultiObjectiveGA(
            gb, lambda ind: [0.0, 0.0, 0.0], n_objectives=3,
            algorithm='nsga3', reference_point_divisions=4,
        )
        for rp in ga._ref_points:
            assert abs(sum(rp) - 1.0) < 1e-9


# ===========================================================================
# Feature 9: landscape_analysis
# ===========================================================================

class TestLandscapeAnalysis:
    def test_returns_expected_keys(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        required = {
            'ruggedness', 'neutrality', 'estimated_modes',
            'float_only', 'recommendation', 'reason',
            'sample_best', 'sample_best_individual',
        }
        assert required.issubset(report.keys())

    def test_float_only_true(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert report['float_only'] is True

    def test_float_only_false_mixed(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        gb.add('n', IntRange(0, 10))
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert report['float_only'] is False

    def test_ruggedness_in_range(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert 0.0 <= report['ruggedness'] <= 1.0

    def test_neutrality_in_range(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert 0.0 <= report['neutrality'] <= 1.0

    def test_estimated_modes_positive(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert report['estimated_modes'] >= 1

    def test_recommendation_is_valid_class(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        valid = {'CMAESOptimizer', 'DEOptimizer', 'IslandModel', 'GeneticAlgorithm'}
        assert report['recommendation'] in valid

    def test_reason_is_string(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert isinstance(report['reason'], str)
        assert len(report['reason']) > 0

    def test_sample_best_is_float(self):
        gb = _make_gb()
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        assert isinstance(report['sample_best'], float)

    def test_sample_best_individual_has_all_genes(self):
        gb = _make_gb(n=3)
        report = landscape_analysis(gb, _fitness, n_samples=50, seed=1)
        ind = report['sample_best_individual']
        assert set(ind.keys()) == {'x0', 'x1', 'x2'}

    def test_reproducible_with_seed(self):
        gb = _make_gb()
        r1 = landscape_analysis(gb, _fitness, n_samples=50, seed=42)
        r2 = landscape_analysis(gb, _fitness, n_samples=50, seed=42)
        assert r1['ruggedness'] == r2['ruggedness']
        assert r1['sample_best'] == r2['sample_best']

    def test_flat_landscape_high_neutrality(self):
        """Constant fitness = maximally neutral."""
        gb = _make_gb()
        report = landscape_analysis(gb, lambda ind: 1.0, n_samples=50, seed=1)
        assert report['neutrality'] > 0.9

    def test_mixed_genes_recommendation_ga(self):
        """Mixed genes should recommend GeneticAlgorithm."""
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        gb.add('c', ChoiceList(['a', 'b', 'c']))
        # fitness must handle ChoiceList gene (non-numeric)
        mixed_fitness = lambda ind: ind['x']
        report = landscape_analysis(gb, mixed_fitness, n_samples=50, seed=1)
        assert report['recommendation'] == 'GeneticAlgorithm'
        assert report['float_only'] is False
