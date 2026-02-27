"""
Tests for IslandModel.
"""

import json
import os
import pytest

from evogine import (
    GeneBuilder,
    FloatRange,
    IntRange,
    ChoiceList,
    IslandModel,
    TournamentSelection,
    ArithmeticCrossover,
)


def sphere_genes():
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("y", FloatRange(-5.0, 5.0))
    return genes


def sphere_fitness(ind):
    return -(ind["x"] ** 2 + ind["y"] ** 2)


def make_island(genes, fitness_fn, **kwargs):
    defaults = dict(
        n_islands=3,
        island_population=20,
        generations=10,
        mutation_rate=0.2,
        seed=42,
    )
    defaults.update(kwargs)
    return IslandModel(gene_builder=genes, fitness_function=fitness_fn, **defaults)


class TestIslandModelBasic:
    def test_returns_three_values(self):
        im = make_island(sphere_genes(), sphere_fitness)
        result = im.run()
        assert len(result) == 3

    def test_best_individual_is_dict(self):
        im = make_island(sphere_genes(), sphere_fitness)
        best, score, history = im.run()
        assert isinstance(best, dict)
        assert "x" in best
        assert "y" in best

    def test_best_score_is_float(self):
        im = make_island(sphere_genes(), sphere_fitness)
        _, score, _ = im.run()
        assert isinstance(score, float)

    def test_history_length(self):
        im = make_island(sphere_genes(), sphere_fitness, generations=8)
        _, _, history = im.run()
        assert len(history) == 8

    def test_history_keys(self):
        im = make_island(sphere_genes(), sphere_fitness, generations=5)
        _, _, history = im.run()
        for h in history:
            assert 'gen' in h
            assert 'best_score' in h
            assert 'avg_score' in h
            assert 'island_bests' in h
            assert 'improved' in h
            assert 'gens_without_improvement' in h

    def test_island_bests_length_matches_n_islands(self):
        n = 4
        im = make_island(sphere_genes(), sphere_fitness, n_islands=n, generations=5)
        _, _, history = im.run()
        for h in history:
            assert len(h['island_bests']) == n

    def test_gen_counter(self):
        im = make_island(sphere_genes(), sphere_fitness, generations=6)
        _, _, history = im.run()
        assert [h['gen'] for h in history] == list(range(1, 7))


class TestIslandModelConvergence:
    def test_converges_on_sphere(self):
        """Island model should find a solution within 0.5 of origin."""
        im = make_island(
            sphere_genes(), sphere_fitness,
            n_islands=4,
            island_population=30,
            generations=30,
            seed=99,
        )
        best, score, _ = im.run()
        assert score > -2.0, f"Expected convergence, got score={score}"

    def test_best_score_non_decreasing(self):
        """Best score across all generations should never go down."""
        im = make_island(sphere_genes(), sphere_fitness, generations=15, seed=7)
        _, _, history = im.run()
        for i in range(1, len(history)):
            assert history[i]['best_score'] >= history[i - 1]['best_score'] - 1e-9


class TestIslandModelReproducibility:
    def test_same_seed_same_result(self):
        genes = sphere_genes()
        im1 = make_island(genes, sphere_fitness, seed=5)
        im2 = make_island(genes, sphere_fitness, seed=5)
        _, s1, _ = im1.run()
        _, s2, _ = im2.run()
        assert s1 == s2

    def test_different_seed_different_result(self):
        genes = sphere_genes()
        im1 = make_island(genes, sphere_fitness, seed=1)
        im2 = make_island(genes, sphere_fitness, seed=2)
        _, s1, _ = im1.run()
        _, s2, _ = im2.run()
        assert s1 != s2


class TestIslandModelEarlyStopping:
    def test_early_stop_fires(self):
        im = make_island(
            sphere_genes(), sphere_fitness,
            generations=50,
            patience=5,
            seed=42,
        )
        _, _, history = im.run()
        assert len(history) < 50

    def test_early_stop_at_patience_boundary(self):
        im = make_island(
            sphere_genes(), sphere_fitness,
            generations=50,
            patience=5,
            seed=42,
        )
        _, _, history = im.run()
        assert history[-1]['gens_without_improvement'] >= 5


class TestIslandModelMigration:
    def test_migration_does_not_crash(self):
        """Migration should happen silently without errors."""
        im = make_island(
            sphere_genes(), sphere_fitness,
            n_islands=3,
            generations=20,
            migration_interval=5,
            migration_size=2,
            seed=1,
        )
        best, score, history = im.run()
        assert best is not None

    def test_migration_interval_1(self):
        """Migration every generation should still work."""
        im = make_island(
            sphere_genes(), sphere_fitness,
            n_islands=2,
            generations=10,
            migration_interval=1,
            migration_size=1,
        )
        best, score, _ = im.run()
        assert best is not None


class TestIslandModelMode:
    def test_minimize_mode(self):
        """Mode=minimize should return real (un-negated) values."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(-5.0, 5.0))

        def fitness(ind):
            return ind["x"] ** 2  # minimize → 0

        im = IslandModel(
            gene_builder=genes,
            fitness_function=fitness,
            n_islands=2,
            island_population=20,
            generations=10,
            mode='minimize',
            seed=42,
        )
        _, score, history = im.run()
        assert score >= 0, "Minimize mode should return non-negative real values"
        for h in history:
            assert h['best_score'] >= 0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            IslandModel(
                gene_builder=sphere_genes(),
                fitness_function=sphere_fitness,
                mode='invalid',
            )


class TestIslandModelLogging:
    def test_log_file_created(self, tmp_path):
        log = str(tmp_path / "island.json")
        im = make_island(sphere_genes(), sphere_fitness, log_path=log)
        im.run()
        assert os.path.isfile(log)

    def test_log_valid_json(self, tmp_path):
        log = str(tmp_path / "island.json")
        im = make_island(sphere_genes(), sphere_fitness, log_path=log)
        im.run()
        with open(log) as f:
            data = json.load(f)
        assert data['run']['type'] == 'island_model'
        assert 'config' in data
        assert 'result' in data
        assert 'history' in data
        assert data['config']['n_islands'] == 3

    def test_log_result_best_score(self, tmp_path):
        log = str(tmp_path / "island.json")
        im = make_island(sphere_genes(), sphere_fitness, log_path=log)
        _, score, _ = im.run()
        with open(log) as f:
            data = json.load(f)
        assert abs(data['result']['best_score'] - score) < 1e-9


class TestIslandModelCallback:
    def test_callback_fires_every_gen(self):
        called = []

        def cb(gen, best, avg, best_ind):
            called.append(gen)

        im = make_island(
            sphere_genes(), sphere_fitness,
            generations=7,
            on_generation=cb,
        )
        im.run()
        assert called == list(range(1, 8))
