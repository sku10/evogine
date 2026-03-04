"""
Tests for MultiObjectiveGA (NSGA-II style Pareto optimization).
"""

import json
import os
import pytest

from evogine import (
    GeneBuilder,
    FloatRange,
    IntRange,
    MultiObjectiveGA,
)


def sphere_genes():
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("y", FloatRange(-5.0, 5.0))
    return genes


def two_obj_fitness(ind):
    """Two objectives: minimize x^2 and minimize y^2 (reported as maximized)."""
    return [-ind["x"] ** 2, -ind["y"] ** 2]


def make_moea(genes, fitness_fn, n_obj=2, **kwargs):
    defaults = dict(
        population_size=40,
        generations=10,
        mutation_rate=0.2,
        seed=42,
    )
    defaults.update(kwargs)
    return MultiObjectiveGA(
        gene_builder=genes,
        fitness_function=fitness_fn,
        n_objectives=n_obj,
        **defaults,
    )


class TestMultiObjectiveBasic:
    def test_returns_two_values(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        result = ga.run()
        assert len(result) == 2

    def test_pareto_front_is_list(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        pareto, history = ga.run()
        assert isinstance(pareto, list)

    def test_pareto_front_not_empty(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        pareto, _ = ga.run()
        assert len(pareto) > 0

    def test_each_front_entry_has_individual_and_scores(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        pareto, _ = ga.run()
        for entry in pareto:
            assert 'individual' in entry
            assert 'scores' in entry
            assert isinstance(entry['individual'], dict)
            assert len(entry['scores']) == 2

    def test_individual_keys_match_genes(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        pareto, _ = ga.run()
        for entry in pareto:
            assert 'x' in entry['individual']
            assert 'y' in entry['individual']

    def test_history_is_list(self):
        ga = make_moea(sphere_genes(), two_obj_fitness)
        _, history = ga.run()
        assert isinstance(history, list)

    def test_history_length(self):
        ga = make_moea(sphere_genes(), two_obj_fitness, generations=8)
        _, history = ga.run()
        assert len(history) == 8

    def test_history_keys(self):
        ga = make_moea(sphere_genes(), two_obj_fitness, generations=5)
        _, history = ga.run()
        for h in history:
            assert 'gen' in h
            assert 'pareto_size' in h
            assert 'hypervolume_proxy' in h
            assert 'improved' in h
            assert 'gens_without_improvement' in h

    def test_gen_counter(self):
        ga = make_moea(sphere_genes(), two_obj_fitness, generations=6)
        _, history = ga.run()
        assert [h['gen'] for h in history] == list(range(1, 7))

    def test_pareto_size_in_history(self):
        ga = make_moea(sphere_genes(), two_obj_fitness, generations=5)
        _, history = ga.run()
        for h in history:
            assert h['pareto_size'] >= 1


class TestMultiObjectiveObjectiveValidation:
    def test_wrong_objectives_length_raises(self):
        with pytest.raises(ValueError, match="n_objectives"):
            MultiObjectiveGA(
                gene_builder=sphere_genes(),
                fitness_function=two_obj_fitness,
                n_objectives=2,
                objectives=['maximize'],  # wrong length
            )

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="maximize.*minimize"):
            MultiObjectiveGA(
                gene_builder=sphere_genes(),
                fitness_function=two_obj_fitness,
                n_objectives=2,
                objectives=['maximize', 'minimize_bad'],
            )

    def test_all_maximize_default(self):
        """Default objectives are all maximize."""
        ga = make_moea(sphere_genes(), two_obj_fitness, n_obj=2)
        assert ga.objectives == ['maximize', 'maximize']

    def test_mixed_objectives(self):
        """Mix of maximize and minimize should not crash."""
        def fitness(ind):
            return [ind["x"], ind["y"]]  # maximize x, minimize y

        ga = MultiObjectiveGA(
            gene_builder=sphere_genes(),
            fitness_function=fitness,
            n_objectives=2,
            objectives=['maximize', 'minimize'],
            population_size=30,
            generations=5,
            seed=1,
        )
        pareto, _ = ga.run()
        assert len(pareto) > 0


class TestMultiObjectiveParetoProperties:
    def test_pareto_front_non_dominated(self):
        """No individual in the Pareto front should be dominated by another."""
        ga = make_moea(sphere_genes(), two_obj_fitness, generations=20, seed=7)
        pareto, _ = ga.run()

        # Check non-domination: no entry dominates any other
        for i, a in enumerate(pareto):
            for j, b in enumerate(pareto):
                if i == j:
                    continue
                a_dominates_b = (
                    all(ai >= bi for ai, bi in zip(a['scores'], b['scores']))
                    and any(ai > bi for ai, bi in zip(a['scores'], b['scores']))
                )
                assert not a_dominates_b, f"Entry {i} dominates entry {j}"

    def test_scores_length_matches_n_objectives(self):
        ga = make_moea(sphere_genes(), two_obj_fitness, n_obj=2)
        pareto, _ = ga.run()
        for entry in pareto:
            assert len(entry['scores']) == 2


class TestMultiObjectiveMinimize:
    def test_minimize_objectives_return_real_values(self):
        """Minimize mode should return positive real values, not negated."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def fitness(ind):
            return [ind["x"] ** 2, (ind["x"] - 5) ** 2]

        ga = MultiObjectiveGA(
            gene_builder=genes,
            fitness_function=fitness,
            n_objectives=2,
            objectives=['minimize', 'minimize'],
            population_size=30,
            generations=10,
            seed=42,
        )
        pareto, _ = ga.run()
        for entry in pareto:
            for s in entry['scores']:
                assert s >= 0, "Minimize mode should return non-negative real values"


class TestMultiObjectiveReproducibility:
    def test_same_seed_same_pareto_size(self):
        ga1 = make_moea(sphere_genes(), two_obj_fitness, seed=3)
        ga2 = make_moea(sphere_genes(), two_obj_fitness, seed=3)
        p1, _ = ga1.run()
        p2, _ = ga2.run()
        assert len(p1) == len(p2)


class TestMultiObjectiveEarlyStopping:
    def test_early_stop_fires(self):
        ga = make_moea(
            sphere_genes(), two_obj_fitness,
            generations=50,
            patience=5,
            seed=42,
        )
        _, history = ga.run()
        assert len(history) < 50

    def test_early_stop_at_patience_boundary(self):
        ga = make_moea(
            sphere_genes(), two_obj_fitness,
            generations=50,
            patience=5,
            seed=42,
        )
        _, history = ga.run()
        assert history[-1]['gens_without_improvement'] >= 5


class TestMultiObjectiveThreeObjectives:
    def test_three_objectives(self):
        """Three objectives should produce a valid Pareto front."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(-3.0, 3.0))
        genes.add("y", FloatRange(-3.0, 3.0))
        genes.add("z", FloatRange(-3.0, 3.0))

        def fitness(ind):
            return [-ind["x"] ** 2, -ind["y"] ** 2, -ind["z"] ** 2]

        ga = MultiObjectiveGA(
            gene_builder=genes,
            fitness_function=fitness,
            n_objectives=3,
            population_size=30,
            generations=8,
            seed=1,
        )
        pareto, history = ga.run()
        assert len(pareto) > 0
        for entry in pareto:
            assert len(entry['scores']) == 3


class TestMultiObjectiveLogging:
    def test_log_file_created(self, tmp_path):
        log = str(tmp_path / "mo.json")
        ga = make_moea(sphere_genes(), two_obj_fitness, log_path=log)
        ga.run()
        assert os.path.isfile(log)

    def test_log_valid_json(self, tmp_path):
        log = str(tmp_path / "mo.json")
        ga = make_moea(sphere_genes(), two_obj_fitness, log_path=log)
        ga.run()
        with open(log) as f:
            data = json.load(f)
        assert data['run']['type'] == 'multi_objective'
        assert 'config' in data
        assert 'result' in data
        assert 'history' in data
        assert data['config']['n_objectives'] == 2

    def test_log_contains_pareto_front(self, tmp_path):
        log = str(tmp_path / "mo.json")
        ga = make_moea(sphere_genes(), two_obj_fitness, log_path=log)
        pareto, _ = ga.run()
        with open(log) as f:
            data = json.load(f)
        assert data['result']['pareto_front_size'] == len(pareto)


class TestMultiObjectiveCallback:
    def test_callback_fires_every_gen(self):
        called = []

        def cb(gen, pareto_size, hv, pareto):
            called.append(gen)

        ga = make_moea(
            sphere_genes(), two_obj_fitness,
            generations=6,
            on_generation=cb,
        )
        ga.run()
        assert called == list(range(1, 7))

    def test_callback_receives_pareto_front(self):
        fronts = []

        def cb(gen, pareto_size, hv, pareto):
            fronts.append(pareto)

        ga = make_moea(sphere_genes(), two_obj_fitness, generations=5, on_generation=cb)
        ga.run()
        assert len(fronts) == 5
        for front in fronts:
            assert isinstance(front, list)


class TestDominatesHelper:
    def _ga(self):
        return make_moea(sphere_genes(), two_obj_fitness)

    def test_a_dominates_b(self):
        ga = self._ga()
        assert ga._dominates([1.0, 1.0], [0.5, 0.5])

    def test_a_does_not_dominate_equal(self):
        ga = self._ga()
        assert not ga._dominates([1.0, 1.0], [1.0, 1.0])

    def test_a_does_not_dominate_when_worse_in_one(self):
        ga = self._ga()
        assert not ga._dominates([1.0, 0.5], [0.5, 1.0])

    def test_non_dominated_sort_simple(self):
        """Two individuals: one dominates the other."""
        ga = self._ga()
        scored = [
            ({}, [1.0, 1.0]),   # index 0: dominates index 1
            ({}, [0.5, 0.5]),   # index 1: dominated
        ]
        fronts = ga._non_dominated_sort(scored)
        assert 0 in fronts[0]
        assert 1 in fronts[1]

    def test_non_dominated_sort_incomparable(self):
        """Two incomparable individuals should both be in front 0."""
        ga = self._ga()
        scored = [
            ({}, [1.0, 0.0]),
            ({}, [0.0, 1.0]),
        ]
        fronts = ga._non_dominated_sort(scored)
        assert len(fronts[0]) == 2


class TestNSGA3SmallPopulation:
    """NSGA-III must not crash when population_size < number of reference points."""

    def test_nsga3_pop_smaller_than_ref_points(self):
        """3 objectives, divisions=6 → 28 ref points. Pop=6 must work."""
        genes = sphere_genes()

        def fitness(ind):
            return [ind['x'], -ind['y'], ind['x'] + ind['y']]

        ga = MultiObjectiveGA(
            gene_builder=genes,
            fitness_function=fitness,
            n_objectives=3,
            objectives=['maximize', 'maximize', 'maximize'],
            algorithm='nsga3',
            reference_point_divisions=6,
            population_size=6,
            generations=5,
            seed=42,
        )
        pareto, history = ga.run()
        assert len(pareto) > 0
        assert len(history) == 5

    def test_nsga3_minimal_population(self):
        """Pop=4 with 3 objectives — extreme edge case."""
        genes = sphere_genes()

        def fitness(ind):
            return [ind['x'], ind['y'], -(ind['x'] ** 2 + ind['y'] ** 2)]

        ga = MultiObjectiveGA(
            gene_builder=genes,
            fitness_function=fitness,
            n_objectives=3,
            objectives=['maximize', 'maximize', 'maximize'],
            algorithm='nsga3',
            reference_point_divisions=3,
            population_size=4,
            generations=5,
            seed=42,
        )
        pareto, history = ga.run()
        assert len(pareto) > 0
