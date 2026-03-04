"""Tests for steering interface, per-generation diagnosis, and from_checkpoint()."""

import json
import os
import tempfile

import pytest

from evogine import (
    GeneBuilder,
    FloatRange,
    IntRange,
    ChoiceList,
    GeneticAlgorithm,
    IslandModel,
    DEOptimizer,
    CMAESOptimizer,
    MultiObjectiveGA,
    MAPElites,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _simple_genes():
    gb = GeneBuilder()
    gb.add('x', FloatRange(0, 10))
    gb.add('y', FloatRange(0, 10))
    return gb


def _fitness(ind):
    return -(ind['x'] - 5) ** 2 - (ind['y'] - 5) ** 2


def _multi_fitness(ind):
    return [ind['x'], ind['y']]


def _behavior_fn(ind):
    return (ind['x'] / 10, ind['y'] / 10)


# ===========================================================================
# Steering Tests
# ===========================================================================

class TestGASteering:
    def test_callback_returns_dict_applies_mutation_rate(self):
        def steer(gen, best, avg, best_ind):
            if gen == 2:
                return {'mutation_rate': 0.99}

        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            population_size=20,
            on_generation=steer,
            seed=42,
        )
        ga.run()
        assert ga.mutation_rate == 0.99

    def test_callback_returns_none_no_change(self):
        original_rate = 0.1

        def steer(gen, best, avg, best_ind):
            return None

        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=20,
            on_generation=steer,
            mutation_rate=original_rate,
            seed=42,
        )
        ga.run()
        assert ga.mutation_rate == original_rate

    def test_callback_returns_non_dict_no_crash(self):
        def steer(gen, best, avg, best_ind):
            return 42  # not a dict

        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=20,
            on_generation=steer,
            seed=42,
        )
        _, _, history = ga.run()
        assert len(history) == 3

    def test_unknown_key_ignored(self):
        def steer(gen, best, avg, best_ind):
            return {'nonexistent_param': 999}

        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=20,
            on_generation=steer,
            seed=42,
        )
        ga.run()
        assert not hasattr(ga, 'nonexistent_param')

    def test_steer_patience(self):
        def steer(gen, best, avg, best_ind):
            if gen == 1:
                return {'patience': 2}

        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=100,
            population_size=20,
            on_generation=steer,
            seed=42,
        )
        _, _, history = ga.run()
        # Should early-stop due to patience=2
        assert len(history) < 100


class TestIslandSteering:
    def test_steer_migration_interval(self):
        steered = {}

        def steer(gen, best, avg, best_ind):
            if gen == 1:
                return {'migration_interval': 1}

        island = IslandModel(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            n_islands=2,
            island_population=10,
            migration_interval=100,
            on_generation=steer,
            seed=42,
        )
        island.run()
        assert island.migration_interval == 1


class TestDESteering:
    def test_steer_strategy(self):
        def steer(gen, best, avg, best_ind):
            if gen == 2:
                return {'strategy': 'rand1'}

        de = DEOptimizer(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            population_size=10,
            on_generation=steer,
            seed=42,
        )
        de.run()
        assert de.strategy == 'rand1'

    def test_steer_invalid_strategy_ignored(self):
        def steer(gen, best, avg, best_ind):
            return {'strategy': 'invalid'}

        de = DEOptimizer(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=10,
            on_generation=steer,
            seed=42,
        )
        de.run()
        assert de.strategy == 'current_to_best'


class TestCMAESSteering:
    def test_steer_patience(self):
        def steer(gen, best, avg, best_ind):
            if gen == 1:
                return {'patience': 2}

        cma = CMAESOptimizer(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=200,
            on_generation=steer,
            seed=42,
        )
        _, _, history = cma.run()
        # Should stop early from patience
        assert len(history) < 200


class TestMultiObjectiveSteering:
    def test_steer_mutation_rate(self):
        def steer(gen, pareto_size, hv, pareto_front):
            if gen == 1:
                return {'mutation_rate': 0.99}

        mo = MultiObjectiveGA(
            gene_builder=_simple_genes(),
            fitness_function=_multi_fitness,
            n_objectives=2,
            generations=3,
            population_size=20,
            on_generation=steer,
            seed=42,
        )
        mo.run()
        assert mo.mutation_rate == 0.99


class TestMAPElitesSteering:
    def test_steer_mutation_rate(self):
        def steer(gen, archive_size, best_score, coverage):
            if gen == 1:
                return {'mutation_rate': 0.99}

        me = MAPElites(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            behavior_fn=_behavior_fn,
            grid_shape=(5, 5),
            generations=10,
            initial_population=20,
            on_generation=steer,
            seed=42,
        )
        me.run()
        assert me.mutation_rate == 0.99


# ===========================================================================
# Diagnosis Tests
# ===========================================================================

class TestGADiagnosis:
    def test_history_has_diagnosis_keys(self):
        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            population_size=20,
            seed=42,
        )
        _, _, history = ga.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h
            assert isinstance(h['diagnosis'], str)
            assert isinstance(h['recommendation'], str)

    def test_low_diversity_diagnosed(self):
        """Directly test the _diagnose_generation method for low diversity."""
        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=20,
            seed=42,
        )
        diag, rec = ga._diagnose_generation(
            diversity=0.01, improved=False, restarted=False,
            gens_without_improvement=0,
        )
        assert diag == 'low_diversity'
        assert 'mutation_rate' in rec

    def test_improving_diagnosed(self):
        ga = GeneticAlgorithm(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=3,
            population_size=50,
            seed=42,
        )
        _, _, history = ga.run()
        diagnoses = [h['diagnosis'] for h in history]
        # First generation should typically show improvement
        assert 'improving' in diagnoses


class TestIslandDiagnosis:
    def test_history_has_diagnosis_keys(self):
        island = IslandModel(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            n_islands=2,
            island_population=10,
            seed=42,
        )
        _, _, history = island.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h

    def test_islands_converged_diagnosed(self):
        """With tiny pop and many gens, islands should converge."""
        island = IslandModel(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=50,
            n_islands=2,
            island_population=10,
            migration_interval=1,
            migration_size=5,
            seed=42,
        )
        _, _, history = island.run()
        diagnoses = [h['diagnosis'] for h in history]
        assert 'islands_converged' in diagnoses


class TestDEDiagnosis:
    def test_history_has_diagnosis_keys(self):
        de = DEOptimizer(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            population_size=10,
            seed=42,
        )
        _, _, history = de.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h


class TestCMAESDiagnosis:
    def test_history_has_diagnosis_keys(self):
        cma = CMAESOptimizer(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            generations=5,
            seed=42,
        )
        _, _, history = cma.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h


class TestMultiObjectiveDiagnosis:
    def test_history_has_diagnosis_keys(self):
        mo = MultiObjectiveGA(
            gene_builder=_simple_genes(),
            fitness_function=_multi_fitness,
            n_objectives=2,
            generations=3,
            population_size=20,
            seed=42,
        )
        _, history = mo.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h


class TestMAPElitesDiagnosis:
    def test_history_has_diagnosis_keys(self):
        me = MAPElites(
            gene_builder=_simple_genes(),
            fitness_function=_fitness,
            behavior_fn=_behavior_fn,
            grid_shape=(5, 5),
            generations=10,
            initial_population=20,
            seed=42,
        )
        _, history = me.run()
        for h in history:
            assert 'diagnosis' in h
            assert 'recommendation' in h


# ===========================================================================
# from_checkpoint() Tests
# ===========================================================================

class TestFromCheckpoint:
    def test_creates_valid_instance(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump({
                'gen': 5,
                'population': [],
                'best_individual': None,
                'best_score_internal': 0.0,
                'gens_without_improvement': 0,
                'convergence_gen': None,
                'history': [],
                'mutation_rate': 0.1,
                'config': {'mode': 'maximize', 'seed': 42, 'generations': 50},
            }, f)
            path = f.name

        try:
            ga = GeneticAlgorithm.from_checkpoint(
                path,
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
            )
            assert isinstance(ga, GeneticAlgorithm)
            assert ga._resume_path == path
        finally:
            os.unlink(path)

    def test_overrides_work(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump({
                'gen': 0,
                'population': [],
                'best_individual': None,
                'best_score_internal': 0.0,
                'gens_without_improvement': 0,
                'convergence_gen': None,
                'history': [],
                'mutation_rate': 0.1,
                'config': {'mode': 'maximize', 'seed': 42, 'generations': 50},
            }, f)
            path = f.name

        try:
            ga = GeneticAlgorithm.from_checkpoint(
                path,
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
                generations=10,
                population_size=15,
            )
            assert ga.generations == 10
            assert ga.population_size == 15
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            GeneticAlgorithm.from_checkpoint(
                '/nonexistent/path.json',
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
            )

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'ckpt.json')

            # Run 5 generations, save checkpoint
            ga1 = GeneticAlgorithm(
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
                generations=5,
                population_size=20,
                seed=42,
                checkpoint_path=ckpt_path,
                checkpoint_every=1,
            )
            best1, score1, history1 = ga1.run()

            # Resume from checkpoint, run 5 more
            ga2 = GeneticAlgorithm.from_checkpoint(
                ckpt_path,
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
                generations=10,
            )
            best2, score2, history2 = ga2.run()

            assert best2 is not None
            assert len(history2) > len(history1)

    def test_resume_path_cleared_after_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, 'ckpt.json')

            ga1 = GeneticAlgorithm(
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
                generations=3,
                population_size=20,
                seed=42,
                checkpoint_path=ckpt_path,
                checkpoint_every=1,
            )
            ga1.run()

            ga2 = GeneticAlgorithm.from_checkpoint(
                ckpt_path,
                gene_builder=_simple_genes(),
                fitness_function=_fitness,
                generations=5,
            )
            assert ga2._resume_path == ckpt_path
            ga2.run()
            assert ga2._resume_path is None
