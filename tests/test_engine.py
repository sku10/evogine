"""
Tests for genetic_engine.py

Covers:
- Gene types: FloatRange, IntRange, ChoiceList
- GeneBuilder
- GeneticAlgorithm: convergence, early stopping, history, logging, reproducibility
- Bug fix regressions
"""

import json
import os
import pytest
import random

from genetic_engine import (
    GeneticAlgorithm,
    GeneBuilder,
    FloatRange,
    IntRange,
    ChoiceList,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_ga(genes, fitness_fn, **kwargs):
    defaults = dict(
        population_size=30,
        generations=20,
        mutation_rate=0.2,
        seed=42,
    )
    defaults.update(kwargs)
    return GeneticAlgorithm(gene_builder=genes, fitness_function=fitness_fn, **defaults)


def sphere_genes():
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("y", FloatRange(-5.0, 5.0))
    return genes


def sphere_fitness(ind):
    """Sphere function — optimum at (0, 0) with score 0."""
    return -(ind["x"] ** 2 + ind["y"] ** 2)


def peak_fitness(ind):
    """Simple peak at x=3.14, y=2.72."""
    return -((ind["x"] - 3.14) ** 2 + (ind["y"] - 2.72) ** 2)


# ---------------------------------------------------------------------------
# FloatRange
# ---------------------------------------------------------------------------

class TestFloatRange:
    def test_sample_within_bounds(self):
        spec = FloatRange(2.0, 8.0)
        for _ in range(200):
            v = spec.sample()
            assert 2.0 <= v <= 8.0

    def test_mutate_respects_bounds(self):
        spec = FloatRange(0.0, 1.0, sigma=0.5)
        for _ in range(200):
            v = spec.mutate(0.5, mutation_rate=1.0)
            assert 0.0 <= v <= 1.0

    def test_mutate_rate_zero_no_change(self):
        spec = FloatRange(0.0, 10.0)
        for _ in range(50):
            v = spec.mutate(5.0, mutation_rate=0.0)
            assert v == 5.0

    def test_mutate_rate_one_always_applied(self):
        """With rate=1.0 and large sigma, most mutations should change the value."""
        spec = FloatRange(0.0, 100.0, sigma=0.5)
        random.seed(0)
        changed = sum(spec.mutate(50.0, mutation_rate=1.0) != 50.0 for _ in range(100))
        # Gaussian noise — extremely unlikely all 100 land exactly on 50.0
        assert changed > 90

    def test_describe(self):
        spec = FloatRange(1.0, 5.0, sigma=0.2)
        d = spec.describe()
        assert d == {"type": "FloatRange", "low": 1.0, "high": 5.0, "sigma": 0.2}


# ---------------------------------------------------------------------------
# IntRange
# ---------------------------------------------------------------------------

class TestIntRange:
    def test_sample_within_bounds(self):
        spec = IntRange(3, 30)
        for _ in range(200):
            v = spec.sample()
            assert 3 <= v <= 30

    def test_sample_is_integer(self):
        spec = IntRange(0, 100)
        for _ in range(50):
            assert isinstance(spec.sample(), int)

    def test_mutate_within_bounds(self):
        spec = IntRange(0, 5)
        for v_start in range(6):
            for _ in range(20):
                v = spec.mutate(v_start, mutation_rate=1.0)
                assert 0 <= v <= 5

    def test_mutate_is_integer(self):
        spec = IntRange(0, 100)
        for _ in range(50):
            assert isinstance(spec.mutate(50, mutation_rate=1.0), int)

    def test_mutate_rate_zero_no_change(self):
        spec = IntRange(0, 100)
        for _ in range(50):
            assert spec.mutate(42, mutation_rate=0.0) == 42

    def test_mutate_boundary_low(self):
        """At lower bound, mutation should never go below it."""
        spec = IntRange(0, 10)
        for _ in range(50):
            v = spec.mutate(0, mutation_rate=1.0)
            assert v >= 0

    def test_mutate_boundary_high(self):
        """At upper bound, mutation should never go above it."""
        spec = IntRange(0, 10)
        for _ in range(50):
            v = spec.mutate(10, mutation_rate=1.0)
            assert v <= 10

    def test_describe(self):
        spec = IntRange(5, 50)
        d = spec.describe()
        assert d == {"type": "IntRange", "low": 5, "high": 50}


# ---------------------------------------------------------------------------
# ChoiceList
# ---------------------------------------------------------------------------

class TestChoiceList:
    def test_sample_from_options(self):
        spec = ChoiceList(["a", "b", "c"])
        for _ in range(100):
            assert spec.sample() in ["a", "b", "c"]

    def test_mutate_from_options(self):
        spec = ChoiceList(["a", "b", "c"])
        for _ in range(100):
            assert spec.mutate("a", mutation_rate=1.0) in ["a", "b", "c"]

    def test_mutate_picks_different_value(self):
        """With rate=1.0, mutate should always pick a different option."""
        spec = ChoiceList(["a", "b", "c"])
        random.seed(0)
        for _ in range(50):
            assert spec.mutate("a", mutation_rate=1.0) != "a"

    def test_mutate_rate_zero_no_change(self):
        spec = ChoiceList(["x", "y", "z"])
        for _ in range(50):
            assert spec.mutate("x", mutation_rate=0.0) == "x"

    def test_single_option_no_crash(self):
        """Bug fix: single-option ChoiceList must not crash."""
        spec = ChoiceList(["only"])
        assert spec.mutate("only", mutation_rate=1.0) == "only"

    def test_two_options_always_switches(self):
        spec = ChoiceList(["yes", "no"])
        random.seed(1)
        for _ in range(50):
            assert spec.mutate("yes", mutation_rate=1.0) == "no"

    def test_describe(self):
        spec = ChoiceList([3, 5, 8])
        d = spec.describe()
        assert d == {"type": "ChoiceList", "options": [3, 5, 8]}


# ---------------------------------------------------------------------------
# GeneBuilder
# ---------------------------------------------------------------------------

class TestGeneBuilder:
    def setup_method(self):
        self.genes = GeneBuilder()
        self.genes.add("f", FloatRange(0.0, 1.0))
        self.genes.add("i", IntRange(1, 10))
        self.genes.add("c", ChoiceList(["a", "b"]))

    def test_sample_has_all_keys(self):
        ind = self.genes.sample()
        assert set(ind.keys()) == {"f", "i", "c"}

    def test_sample_values_correct_types(self):
        for _ in range(20):
            ind = self.genes.sample()
            assert isinstance(ind["f"], float)
            assert isinstance(ind["i"], int)
            assert ind["c"] in ["a", "b"]

    def test_mutate_has_all_keys(self):
        ind = self.genes.sample()
        mutated = self.genes.mutate(ind, 0.5)
        assert set(mutated.keys()) == {"f", "i", "c"}

    def test_keys_preserves_order(self):
        assert self.genes.keys() == ["f", "i", "c"]

    def test_describe_structure(self):
        d = self.genes.describe()
        assert d["f"]["type"] == "FloatRange"
        assert d["i"]["type"] == "IntRange"
        assert d["c"]["type"] == "ChoiceList"


# ---------------------------------------------------------------------------
# GeneticAlgorithm — configuration and output structure
# ---------------------------------------------------------------------------

class TestGeneticAlgorithmOutput:
    def test_run_returns_three_values(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness)
        result = ga.run()
        assert len(result) == 3

    def test_best_individual_has_correct_keys(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness)
        best, _, _ = ga.run()
        assert set(best.keys()) == {"x", "y"}

    def test_best_score_is_float(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness)
        _, score, _ = ga.run()
        assert isinstance(score, float)

    def test_history_length_equals_generations_run(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness, generations=10)
        _, _, history = ga.run()
        assert len(history) == 10

    def test_history_entry_keys(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness, generations=5)
        _, _, history = ga.run()
        for entry in history:
            assert "gen" in entry
            assert "best_score" in entry
            assert "avg_score" in entry
            assert "improved" in entry
            assert "gens_without_improvement" in entry

    def test_history_gen_numbers_sequential(self):
        ga = make_simple_ga(sphere_genes(), sphere_fitness, generations=8)
        _, _, history = ga.run()
        assert [h["gen"] for h in history] == list(range(1, 9))

    def test_history_best_score_never_decreases(self):
        """best_score in history should be monotonically non-decreasing."""
        ga = make_simple_ga(sphere_genes(), sphere_fitness, generations=15)
        _, _, history = ga.run()
        scores = [h["best_score"] for h in history]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# GeneticAlgorithm — reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        genes = sphere_genes()
        ga1 = make_simple_ga(genes, sphere_fitness, seed=99)
        ga2 = make_simple_ga(genes, sphere_fitness, seed=99)
        best1, score1, _ = ga1.run()
        best2, score2, _ = ga2.run()
        assert score1 == score2
        assert best1 == best2

    def test_different_seed_different_result(self):
        genes = sphere_genes()
        ga1 = make_simple_ga(genes, sphere_fitness, seed=1)
        ga2 = make_simple_ga(genes, sphere_fitness, seed=2)
        _, score1, _ = ga1.run()
        _, score2, _ = ga2.run()
        # Extremely unlikely to produce identical scores with different seeds
        assert score1 != score2

    def test_no_seed_does_not_crash(self):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, seed=None)
        best, score, history = ga.run()
        assert best is not None


# ---------------------------------------------------------------------------
# GeneticAlgorithm — early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stops_before_max_generations(self):
        """A quickly-converging problem should stop before the generation limit."""
        genes = sphere_genes()
        ga = make_simple_ga(
            genes, sphere_fitness,
            generations=200,
            patience=5,
            population_size=50,
            seed=42,
        )
        _, _, history = ga.run()
        assert len(history) < 200

    def test_no_patience_runs_all_generations(self):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=10, patience=None)
        _, _, history = ga.run()
        assert len(history) == 10

    def test_gens_without_improvement_resets_on_improvement(self):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=30, seed=42)
        _, _, history = ga.run()
        for entry in history:
            if entry["improved"]:
                assert entry["gens_without_improvement"] == 0

    def test_gens_without_improvement_increments(self):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=30, seed=42)
        _, _, history = ga.run()
        prev = None
        for entry in history:
            if prev is not None and not entry["improved"]:
                assert entry["gens_without_improvement"] == prev["gens_without_improvement"] + 1
            prev = entry


# ---------------------------------------------------------------------------
# GeneticAlgorithm — convergence on benchmark problems
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_sphere_finds_near_optimum(self):
        """Sphere function: optimum is 0 at (0,0). Should get close."""
        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=sphere_fitness,
            population_size=80,
            generations=100,
            mutation_rate=0.2,
            seed=42,
        )
        _, score, _ = ga.run()
        assert score > -0.5  # within 0.5 of optimum

    def test_known_peak_found(self):
        """Peak at (3.14, 2.72) — should find it within tolerance."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))
        genes.add("y", FloatRange(0.0, 10.0))
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=peak_fitness,
            population_size=80,
            generations=100,
            mutation_rate=0.2,
            seed=42,
        )
        best, score, _ = ga.run()
        assert abs(best["x"] - 3.14) < 0.5
        assert abs(best["y"] - 2.72) < 0.5

    def test_intrange_gene_converges(self):
        """Optimum at n=7 — GA should find it."""
        genes = GeneBuilder()
        genes.add("n", IntRange(1, 20))

        def fitness(ind):
            return -(abs(ind["n"] - 7))

        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=fitness,
            population_size=30,
            generations=50,
            mutation_rate=0.3,
            seed=42,
        )
        best, _, _ = ga.run()
        assert best["n"] == 7

    def test_choicelist_gene_converges(self):
        """One option scores much higher — GA should find it."""
        genes = GeneBuilder()
        genes.add("mode", ChoiceList(["bad", "ok", "good", "best"]))

        scores = {"bad": -10, "ok": 0, "good": 5, "best": 100}

        def fitness(ind):
            return scores[ind["mode"]]

        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=fitness,
            population_size=20,
            generations=30,
            mutation_rate=0.5,
            seed=42,
        )
        best, _, _ = ga.run()
        assert best["mode"] == "best"

    def test_mixed_gene_types_converge(self):
        """Mixed FloatRange + IntRange + ChoiceList — all should reach near-optimum."""
        genes = GeneBuilder()
        genes.add("x",    FloatRange(0.0, 10.0))
        genes.add("n",    IntRange(1, 20))
        genes.add("flag", ChoiceList([True, False]))

        def fitness(ind):
            f = -((ind["x"] - 5.0) ** 2)
            f += -(abs(ind["n"] - 10))
            f += 10 if ind["flag"] is True else 0
            return f

        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=fitness,
            population_size=60,
            generations=80,
            mutation_rate=0.2,
            seed=42,
        )
        best, _, _ = ga.run()
        assert abs(best["x"] - 5.0) < 1.0
        assert best["n"] == 10
        assert best["flag"] is True


# ---------------------------------------------------------------------------
# GeneticAlgorithm — structured logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_file_created(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=5)
        ga.run()
        assert os.path.exists(log_file)

    def test_log_is_valid_json(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=5)
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_log_top_level_keys(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=5)
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert "run" in data
        assert "config" in data
        assert "genes" in data
        assert "result" in data
        assert "analysis" in data
        assert "history" in data

    def test_log_config_matches_params(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=sphere_fitness,
            population_size=42,
            generations=7,
            mutation_rate=0.15,
            crossover_rate=0.6,
            elitism=3,
            seed=1,
            patience=5,
            log_path=log_file,
        )
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        cfg = data["config"]
        assert cfg["population_size"] == 42
        assert cfg["mutation_rate"] == 0.15
        assert cfg["crossover_rate"] == 0.6
        assert cfg["elitism"] == 3
        assert cfg["patience"] == 5

    def test_log_genes_structure(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = GeneBuilder()
        genes.add("f", FloatRange(0.0, 1.0))
        genes.add("i", IntRange(1, 10))
        genes.add("c", ChoiceList(["a", "b"]))
        ga = make_simple_ga(genes, lambda ind: 0.0, log_path=log_file, generations=3)
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert data["genes"]["f"]["type"] == "FloatRange"
        assert data["genes"]["i"]["type"] == "IntRange"
        assert data["genes"]["c"]["type"] == "ChoiceList"

    def test_log_result_contains_best(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=5)
        best, score, _ = ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert data["result"]["best_score"] == score
        assert data["result"]["best_individual"] == best

    def test_log_history_length(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=6)
        _, _, history = ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert len(data["history"]) == len(history)

    def test_log_analysis_has_notes(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, log_path=log_file, generations=5)
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert isinstance(data["analysis"]["notes"], list)
        assert len(data["analysis"]["notes"]) > 0

    def test_no_log_when_path_not_set(self, tmp_path):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness)
        ga.run()
        assert not os.path.exists(str(tmp_path / "run.json"))

    def test_log_early_stopped_flag(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=sphere_fitness,
            population_size=50,
            generations=200,
            mutation_rate=0.2,
            seed=42,
            patience=5,
            log_path=log_file,
        )
        ga.run()
        with open(log_file) as f:
            data = json.load(f)
        assert data["result"]["early_stopped"] is True
        assert data["config"]["generations_run"] < 200


# ---------------------------------------------------------------------------
# Bug fix regressions
# ---------------------------------------------------------------------------

class TestBugFixes:
    def test_seed_is_respected(self):
        """Regression: seed was previously set to time.time() instead of the value."""
        genes = sphere_genes()
        ga1 = make_simple_ga(genes, sphere_fitness, seed=7)
        ga2 = make_simple_ga(genes, sphere_fitness, seed=7)
        _, s1, _ = ga1.run()
        _, s2, _ = ga2.run()
        assert s1 == s2

    def test_choicelist_single_option_no_crash(self):
        """Regression: ChoiceList with 1 option used to crash on mutate."""
        genes = GeneBuilder()
        genes.add("only", ChoiceList(["singleton"]))

        def fitness(ind):
            return 1.0

        ga = make_simple_ga(genes, fitness, generations=5)
        best, score, _ = ga.run()
        assert best["only"] == "singleton"
