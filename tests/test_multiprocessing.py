"""
Tests for multiprocessing support across all optimizers.

Verifies that use_multiprocessing=True produces valid results for all 6 optimizers,
that MAPElites batch_size works correctly, and that the workers parameter resolves
correctly.
"""

import json
import multiprocessing as mp
from unittest.mock import patch

import pytest

from evogine import (
    GeneticAlgorithm,
    IslandModel,
    MultiObjectiveGA,
    DEOptimizer,
    CMAESOptimizer,
    MAPElites,
    GeneBuilder,
    FloatRange,
)
from evogine._utils import _resolve_workers, _SafeEncoder


def _make_genes():
    gb = GeneBuilder()
    gb.add("x", FloatRange(-5, 5))
    gb.add("y", FloatRange(-5, 5))
    return gb


def _fitness(ind):
    return -(ind["x"] ** 2 + ind["y"] ** 2)


def _multi_fitness(ind):
    return [-(ind["x"] ** 2), -(ind["y"] ** 2)]


def _behavior(ind):
    return ((ind["x"] + 5) / 10, (ind["y"] + 5) / 10)


class TestGAMultiprocessing:
    def test_ga_multiprocessing(self):
        ga = GeneticAlgorithm(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            use_multiprocessing=True,
            seed=42,
        )
        best, score, history = ga.run()
        assert best is not None
        assert isinstance(score, float)
        assert len(history) == 3


class TestIslandMultiprocessing:
    def test_island_multiprocessing(self):
        im = IslandModel(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            n_islands=2,
            island_population=10,
            generations=3,
            migration_interval=2,
            use_multiprocessing=True,
            seed=42,
        )
        best, score, history = im.run()
        assert best is not None
        assert isinstance(score, float)
        assert len(history) == 3


class TestMultiObjectiveMultiprocessing:
    def test_multi_objective_multiprocessing(self):
        mo = MultiObjectiveGA(
            gene_builder=_make_genes(),
            fitness_function=_multi_fitness,
            n_objectives=2,
            population_size=10,
            generations=3,
            use_multiprocessing=True,
            seed=42,
        )
        pareto, history = mo.run()
        assert len(pareto) > 0
        assert len(history) == 3


class TestDEMultiprocessing:
    def test_de_multiprocessing(self):
        de = DEOptimizer(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            use_multiprocessing=True,
            seed=42,
        )
        best, score, history = de.run()
        assert best is not None
        assert isinstance(score, float)
        assert len(history) == 3

    def test_de_default_no_multiprocessing(self):
        de = DEOptimizer(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert de.use_multiprocessing is False
        best, score, history = de.run()
        assert best is not None


class TestCMAESMultiprocessing:
    def test_cmaes_multiprocessing(self):
        cmaes = CMAESOptimizer(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            generations=3,
            use_multiprocessing=True,
            seed=42,
        )
        best, score, history = cmaes.run()
        assert best is not None
        assert isinstance(score, float)
        assert len(history) == 3

    def test_cmaes_default_no_multiprocessing(self):
        cmaes = CMAESOptimizer(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            generations=3,
            seed=42,
        )
        assert cmaes.use_multiprocessing is False


class TestMAPElitesMultiprocessing:
    def test_mapelites_multiprocessing(self):
        me = MAPElites(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            behavior_fn=_behavior,
            grid_shape=(5, 5),
            initial_population=20,
            generations=10,
            use_multiprocessing=True,
            seed=42,
        )
        archive, history = me.run()
        assert len(archive) > 0
        assert len(history) == 11  # gen 0 (seed) + 10 gens

    def test_mapelites_batch_size(self):
        me = MAPElites(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            behavior_fn=_behavior,
            grid_shape=(5, 5),
            initial_population=20,
            generations=5,
            batch_size=10,
            seed=42,
        )
        archive, history = me.run()
        assert len(archive) > 0
        assert len(history) == 6  # gen 0 + 5 gens

    def test_mapelites_default_no_multiprocessing(self):
        me = MAPElites(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            behavior_fn=_behavior,
            grid_shape=(5, 5),
            initial_population=10,
            generations=3,
            seed=42,
        )
        assert me.use_multiprocessing is False
        assert me.batch_size == 1


class TestSafeEncoder:
    """Verify _SafeEncoder handles numpy scalars in JSON."""

    def test_numpy_bool(self):
        np = pytest.importorskip("numpy")
        data = {"flag": np.bool_(True), "off": np.bool_(False)}
        result = json.loads(json.dumps(data, cls=_SafeEncoder))
        assert result == {"flag": True, "off": False}
        assert isinstance(result["flag"], bool)

    def test_numpy_int64(self):
        np = pytest.importorskip("numpy")
        data = {"val": np.int64(42)}
        result = json.loads(json.dumps(data, cls=_SafeEncoder))
        assert result == {"val": 42}
        assert isinstance(result["val"], int)

    def test_numpy_float64(self):
        np = pytest.importorskip("numpy")
        data = {"val": np.float64(3.14)}
        result = json.loads(json.dumps(data, cls=_SafeEncoder))
        assert abs(result["val"] - 3.14) < 1e-10

    def test_plain_types_unaffected(self):
        data = {"a": 1, "b": True, "c": "hello", "d": [1, 2]}
        result = json.loads(json.dumps(data, cls=_SafeEncoder))
        assert result == data


class TestResolveWorkers:
    """Unit tests for _resolve_workers resolution logic."""

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_none_false_returns_none(self, _mock):
        assert _resolve_workers(None, False) is None

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_none_true_returns_cpu_count(self, _mock):
        assert _resolve_workers(None, True) == 8

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_zero_returns_cpu_count(self, _mock):
        assert _resolve_workers(0, False) == 8
        assert _resolve_workers(0, True) == 8

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_positive_capped(self, _mock):
        assert _resolve_workers(4, False) == 4
        assert _resolve_workers(100, False) == 8

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_positive_one(self, _mock):
        assert _resolve_workers(1, False) == 1

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_negative_one(self, _mock):
        assert _resolve_workers(-1, False) == 7

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_negative_two(self, _mock):
        assert _resolve_workers(-2, False) == 6

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_negative_floors_at_one(self, _mock):
        assert _resolve_workers(-100, False) == 1

    @patch.object(mp, 'cpu_count', return_value=8)
    def test_workers_overrides_use_multiprocessing(self, _mock):
        # workers=4 implies multiprocessing, use_multiprocessing is ignored
        assert _resolve_workers(4, False) == 4
        assert _resolve_workers(4, True) == 4


class TestGAWorkers:
    def test_ga_workers_fixed(self):
        ga = GeneticAlgorithm(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            workers=2,
            seed=42,
        )
        best, score, history = ga.run()
        assert best is not None
        assert len(history) == 3

    def test_ga_workers_negative(self):
        ga = GeneticAlgorithm(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            workers=-1,
            seed=42,
        )
        best, score, history = ga.run()
        assert best is not None
        assert len(history) == 3

    def test_ga_workers_zero(self):
        ga = GeneticAlgorithm(
            gene_builder=_make_genes(),
            fitness_function=_fitness,
            population_size=10,
            generations=3,
            workers=0,
            seed=42,
        )
        best, score, history = ga.run()
        assert best is not None
        assert len(history) == 3
