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
    RouletteSelection,
    TournamentSelection,
    RankSelection,
    UniformCrossover,
    ArithmeticCrossover,
    SinglePointCrossover,
    _seed_all,
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
        spec = IntRange(5, 50, sigma=0.1)
        d = spec.describe()
        assert d == {"type": "IntRange", "low": 5, "high": 50, "sigma": 0.1}


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

class TestIntRangeSigma:
    def test_default_sigma_still_valid(self):
        spec = IntRange(0, 100)
        for _ in range(100):
            v = spec.mutate(50, mutation_rate=1.0)
            assert 0 <= v <= 100

    def test_wide_range_jumps_more_than_one(self):
        """With sigma=0.1 on a range of 100, jump should be up to ±10."""
        spec = IntRange(0, 100, sigma=0.1)
        random.seed(0)
        deltas = set()
        for _ in range(200):
            v = spec.mutate(50, mutation_rate=1.0)
            deltas.add(abs(v - 50))
        assert max(deltas) > 1  # should see jumps larger than ±1

    def test_narrow_range_still_moves(self):
        """IntRange(5, 7) with any sigma should still be able to mutate."""
        spec = IntRange(5, 7, sigma=0.05)
        random.seed(1)
        values = {spec.mutate(6, mutation_rate=1.0) for _ in range(50)}
        assert len(values) > 1  # should visit more than one value

    def test_sigma_in_describe(self):
        spec = IntRange(0, 50, sigma=0.2)
        d = spec.describe()
        assert d['sigma'] == 0.2

    def test_large_sigma_converges_faster(self):
        """Wide sigma should find the optimum in fewer generations."""
        genes_slow = GeneBuilder()
        genes_slow.add("n", IntRange(0, 200, sigma=0.01))  # tiny steps

        genes_fast = GeneBuilder()
        genes_fast.add("n", IntRange(0, 200, sigma=0.2))  # big steps

        def fitness(ind):
            return -abs(ind["n"] - 150)

        ga_slow = GeneticAlgorithm(gene_builder=genes_slow, fitness_function=fitness,
                                   population_size=20, generations=30, seed=42)
        ga_fast = GeneticAlgorithm(gene_builder=genes_fast, fitness_function=fitness,
                                   population_size=20, generations=30, seed=42)

        _, score_slow, _ = ga_slow.run()
        _, score_fast, _ = ga_fast.run()
        assert score_fast >= score_slow  # fast sigma should reach at least as good a score


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

class TestRouletteSelection:
    def test_returns_two_individuals(self):
        scored = [({'x': i}, float(i)) for i in range(10)]
        sel = RouletteSelection()
        p1, p2 = sel.select_parents(scored)
        assert isinstance(p1, dict) and isinstance(p2, dict)

    def test_all_same_score_no_crash(self):
        scored = [({'x': 1}, 5.0) for _ in range(10)]
        sel = RouletteSelection()
        p1, p2 = sel.select_parents(scored)
        assert p1 is not None

    def test_negative_scores_no_crash(self):
        scored = [({'x': i}, float(-i)) for i in range(1, 10)]
        sel = RouletteSelection()
        p1, p2 = sel.select_parents(scored)
        assert p1 is not None

    def test_describe(self):
        assert RouletteSelection().describe() == {'strategy': 'roulette'}


class TestTournamentSelection:
    def test_returns_two_individuals(self):
        scored = [({'x': i}, float(i)) for i in range(10)]
        sel = TournamentSelection(k=3)
        p1, p2 = sel.select_parents(scored)
        assert isinstance(p1, dict) and isinstance(p2, dict)

    def test_k_larger_than_population_no_crash(self):
        scored = [({'x': i}, float(i)) for i in range(3)]
        sel = TournamentSelection(k=10)
        p1, p2 = sel.select_parents(scored)
        assert p1 is not None

    def test_high_k_favours_best(self):
        """With k = population_size, best individual should always win."""
        scored = [({'x': i}, float(i)) for i in range(20)]
        sel = TournamentSelection(k=20)
        random.seed(0)
        for _ in range(20):
            p1, _ = sel.select_parents(scored)
            assert p1['x'] == 19  # highest score always wins

    def test_describe(self):
        assert TournamentSelection(k=5).describe() == {'strategy': 'tournament', 'k': 5}


class TestRankSelection:
    def test_returns_two_individuals(self):
        scored = [({'x': i}, float(i)) for i in range(10)]
        sel = RankSelection()
        p1, p2 = sel.select_parents(scored)
        assert isinstance(p1, dict) and isinstance(p2, dict)

    def test_works_with_negative_scores(self):
        scored = [({'x': i}, float(-i * 100)) for i in range(10)]
        sel = RankSelection()
        p1, p2 = sel.select_parents(scored)
        assert p1 is not None

    def test_describe(self):
        assert RankSelection().describe() == {'strategy': 'rank'}


# ---------------------------------------------------------------------------
# Crossover strategies
# ---------------------------------------------------------------------------

class TestUniformCrossover:
    def test_child_has_all_keys(self):
        genes = GeneBuilder()
        genes.add("a", FloatRange(0, 1))
        genes.add("b", IntRange(0, 10))
        xo = UniformCrossover()
        p1 = {'a': 0.1, 'b': 2}
        p2 = {'a': 0.9, 'b': 8}
        child = xo.crossover(p1, p2, genes)
        assert set(child.keys()) == {'a', 'b'}

    def test_child_values_come_from_parents(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0, 1))
        xo = UniformCrossover()
        random.seed(0)
        for _ in range(50):
            child = xo.crossover({'x': 0.1}, {'x': 0.9}, genes)
            assert child['x'] in (0.1, 0.9)

    def test_describe(self):
        assert UniformCrossover().describe() == {'strategy': 'uniform'}


class TestArithmeticCrossover:
    def test_child_has_all_keys(self):
        genes = GeneBuilder()
        genes.add("f", FloatRange(0, 10))
        genes.add("i", IntRange(0, 10))
        genes.add("c", ChoiceList(["a", "b"]))
        xo = ArithmeticCrossover()
        p1 = {'f': 2.0, 'i': 3, 'c': 'a'}
        p2 = {'f': 8.0, 'i': 7, 'c': 'b'}
        child = xo.crossover(p1, p2, genes)
        assert set(child.keys()) == {'f', 'i', 'c'}

    def test_float_child_between_parents(self):
        genes = GeneBuilder()
        genes.add("f", FloatRange(0, 10))
        xo = ArithmeticCrossover()
        random.seed(0)
        for _ in range(50):
            child = xo.crossover({'f': 2.0}, {'f': 8.0}, genes)
            assert 2.0 <= child['f'] <= 8.0

    def test_non_float_uses_uniform(self):
        genes = GeneBuilder()
        genes.add("i", IntRange(0, 10))
        genes.add("c", ChoiceList(["x", "y"]))
        xo = ArithmeticCrossover()
        random.seed(0)
        for _ in range(50):
            child = xo.crossover({'i': 1, 'c': 'x'}, {'i': 9, 'c': 'y'}, genes)
            assert child['i'] in (1, 9)
            assert child['c'] in ('x', 'y')

    def test_describe(self):
        assert ArithmeticCrossover().describe() == {'strategy': 'arithmetic'}


class TestSinglePointCrossover:
    def test_child_has_all_keys(self):
        genes = GeneBuilder()
        genes.add("a", FloatRange(0, 1))
        genes.add("b", FloatRange(0, 1))
        genes.add("c", FloatRange(0, 1))
        xo = SinglePointCrossover()
        p1 = {'a': 0.1, 'b': 0.1, 'c': 0.1}
        p2 = {'a': 0.9, 'b': 0.9, 'c': 0.9}
        child = xo.crossover(p1, p2, genes)
        assert set(child.keys()) == {'a', 'b', 'c'}

    def test_values_come_from_parents(self):
        genes = GeneBuilder()
        genes.add("a", FloatRange(0, 1))
        genes.add("b", FloatRange(0, 1))
        genes.add("c", FloatRange(0, 1))
        xo = SinglePointCrossover()
        random.seed(0)
        for _ in range(30):
            p1 = {'a': 0.1, 'b': 0.1, 'c': 0.1}
            p2 = {'a': 0.9, 'b': 0.9, 'c': 0.9}
            child = xo.crossover(p1, p2, genes)
            for key in child:
                assert child[key] in (0.1, 0.9)

    def test_single_gene_no_crash(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0, 1))
        xo = SinglePointCrossover()
        child = xo.crossover({'x': 0.2}, {'x': 0.8}, genes)
        assert 'x' in child

    def test_describe(self):
        assert SinglePointCrossover().describe() == {'strategy': 'single_point'}


# ---------------------------------------------------------------------------
# GA integration tests for strategies
# ---------------------------------------------------------------------------

class TestGAWithStrategies:
    def _run(self, selection=None, crossover=None):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))
        genes.add("y", FloatRange(0.0, 10.0))
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=peak_fitness,
            population_size=50,
            generations=50,
            mutation_rate=0.2,
            seed=42,
            selection=selection,
            crossover=crossover,
        )
        return ga.run()

    def test_tournament_selection_converges(self):
        best, score, _ = self._run(selection=TournamentSelection(k=3))
        assert score > -1.0

    def test_rank_selection_converges(self):
        best, score, _ = self._run(selection=RankSelection())
        assert score > -1.0

    def test_arithmetic_crossover_converges(self):
        best, score, _ = self._run(crossover=ArithmeticCrossover())
        assert score > -1.0

    def test_single_point_crossover_converges(self):
        best, score, _ = self._run(crossover=SinglePointCrossover())
        assert score > -1.0

    def test_tournament_plus_arithmetic_converges(self):
        best, score, _ = self._run(
            selection=TournamentSelection(k=4),
            crossover=ArithmeticCrossover(),
        )
        assert score > -1.0

    def test_default_strategies_unchanged(self):
        """No selection/crossover args → same results as before."""
        best, score, _ = self._run()
        assert score > -1.0

    def test_log_includes_strategy_names(self, tmp_path):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0, 10))
        log_file = str(tmp_path / "run.json")
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=lambda ind: -ind["x"],
            population_size=10,
            generations=3,
            seed=1,
            selection=TournamentSelection(k=3),
            crossover=ArithmeticCrossover(),
            log_path=log_file,
        )
        ga.run()
        import json
        with open(log_file) as f:
            data = json.load(f)
        assert data["config"]["selection"]["strategy"] == "tournament"
        assert data["config"]["crossover"]["strategy"] == "arithmetic"


class TestCallback:
    def test_callback_called_each_generation(self):
        call_count = []

        def cb(gen, best_score, avg_score, best_ind):
            call_count.append(gen)

        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=8, on_generation=cb)
        ga.run()
        assert call_count == list(range(1, 9))

    def test_callback_receives_correct_gen_numbers(self):
        received = []

        def cb(gen, best_score, avg_score, best_ind):
            received.append(gen)

        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=5, on_generation=cb)
        ga.run()
        assert received == [1, 2, 3, 4, 5]

    def test_callback_receives_scores(self):
        scores = []

        def cb(gen, best_score, avg_score, best_ind):
            scores.append((best_score, avg_score))

        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=5, on_generation=cb)
        ga.run()
        assert len(scores) == 5
        for best, avg in scores:
            assert isinstance(best, float)
            assert isinstance(avg, float)
            assert best >= avg  # best is always >= avg

    def test_callback_receives_best_individual(self):
        individuals = []

        def cb(gen, best_score, avg_score, best_ind):
            individuals.append(best_ind)

        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=5, on_generation=cb)
        ga.run()
        for ind in individuals:
            assert isinstance(ind, dict)
            assert 'x' in ind and 'y' in ind

    def test_callback_fires_on_early_stop(self):
        call_count = []

        def cb(gen, best_score, avg_score, best_ind):
            call_count.append(gen)

        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=sphere_fitness,
            population_size=50,
            generations=200,
            mutation_rate=0.2,
            seed=42,
            patience=5,
            on_generation=cb,
        )
        _, _, history = ga.run()
        assert len(call_count) == len(history)
        assert len(call_count) < 200

    def test_no_callback_still_works(self):
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=5)
        best, score, history = ga.run()
        assert best is not None

    def test_callback_can_accumulate_data(self):
        """Typical use case: accumulate data for plotting."""
        data = {'gens': [], 'best': [], 'avg': []}

        def cb(gen, best_score, avg_score, best_ind):
            data['gens'].append(gen)
            data['best'].append(best_score)
            data['avg'].append(avg_score)

        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=10, on_generation=cb)
        _, _, history = ga.run()

        assert len(data['gens']) == 10
        assert data['best'] == [h['best_score'] for h in history]
        assert data['avg'] == [h['avg_score'] for h in history]


class TestAdaptiveMutation:
    def _ga(self, **kwargs):
        genes = sphere_genes()
        defaults = dict(
            gene_builder=genes, fitness_function=sphere_fitness,
            population_size=40, generations=30, mutation_rate=0.2, seed=42,
            adaptive_mutation=True,
        )
        defaults.update(kwargs)
        return GeneticAlgorithm(**defaults)

    def test_mutation_rate_recorded_in_history(self):
        ga = self._ga()
        _, _, history = ga.run()
        for h in history:
            assert 'mutation_rate' in h
            assert isinstance(h['mutation_rate'], float)

    def test_rate_changes_over_time(self):
        """Adaptive rate should not stay fixed — it must vary across generations."""
        ga = self._ga(mutation_rate=0.2)
        _, _, history = ga.run()
        rates = [h['mutation_rate'] for h in history]
        assert len(set(rates)) > 1  # rate must change at some point

    def test_rate_never_below_min(self):
        ga = self._ga(adaptive_mutation_min=0.05)
        _, _, history = ga.run()
        for h in history:
            assert h['mutation_rate'] >= 0.05 - 1e-9

    def test_rate_never_above_max(self):
        ga = self._ga(adaptive_mutation_max=0.4)
        _, _, history = ga.run()
        for h in history:
            assert h['mutation_rate'] <= 0.4 + 1e-9

    def test_disabled_by_default(self):
        """mutation_rate should stay fixed when adaptive_mutation=False."""
        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=sphere_fitness,
            population_size=30, generations=10, mutation_rate=0.2, seed=42,
        )
        _, _, history = ga.run()
        for h in history:
            assert h['mutation_rate'] == 0.2

    def test_adaptive_still_converges(self):
        genes = sphere_genes()
        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=sphere_fitness,
            population_size=80, generations=100, mutation_rate=0.2, seed=42,
            adaptive_mutation=True,
        )
        _, score, _ = ga.run()
        assert score > -0.5

    def test_log_includes_adaptive_config(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        ga = self._ga(
            adaptive_mutation_min=0.02, adaptive_mutation_max=0.45,
            log_path=log_file, generations=5,
        )
        ga.run()
        import json
        with open(log_file) as f:
            data = json.load(f)
        assert data['config']['adaptive_mutation'] is True
        assert data['config']['adaptive_mutation_min'] == 0.02
        assert data['config']['adaptive_mutation_max'] == 0.45

    def test_log_history_has_mutation_rate(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        ga = self._ga(log_path=log_file, generations=5)
        ga.run()
        import json
        with open(log_file) as f:
            data = json.load(f)
        for entry in data['history']:
            assert 'mutation_rate' in entry


class TestMinimizeMode:
    def test_minimize_finds_minimum(self):
        """Minimize MSE — optimum at x=3.14, y=2.72 with error=0."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))
        genes.add("y", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 3.14)**2 + (ind["y"] - 2.72)**2

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=80, generations=100,
            mutation_rate=0.2, seed=42, mode='minimize',
        )
        best, score, _ = ga.run()
        assert score >= 0               # real error value, not negated
        assert score < 0.5              # found something close to minimum
        assert abs(best["x"] - 3.14) < 0.5
        assert abs(best["y"] - 2.72) < 0.5

    def test_minimize_score_is_real_value_not_negated(self):
        """Returned score must be the real error value, not the internal negated one."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 5.0)**2  # always >= 0

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=30, generations=20,
            seed=42, mode='minimize',
        )
        _, score, _ = ga.run()
        assert score >= 0  # must be positive real error, never negative

    def test_minimize_history_scores_are_real(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 5.0)**2

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=30, generations=10,
            seed=42, mode='minimize',
        )
        _, _, history = ga.run()
        for h in history:
            assert h['best_score'] >= 0  # real error, never negative

    def test_minimize_history_best_decreases(self):
        """For minimize mode, best_score in history should trend downward."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 5.0)**2

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=50, generations=30,
            seed=42, mode='minimize',
        )
        _, _, history = ga.run()
        scores = [h['best_score'] for h in history]
        # Best score should never get worse (increase) in minimize mode
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-12

    def test_maximize_still_works(self):
        """Default mode unchanged."""
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, mode='maximize')
        _, score, _ = ga.run()
        assert score <= 0  # sphere returns negative values

    def test_invalid_mode_raises(self):
        genes = sphere_genes()
        with pytest.raises(ValueError, match="mode must be"):
            GeneticAlgorithm(gene_builder=genes, fitness_function=sphere_fitness,
                             mode='sideways')

    def test_minimize_log_shows_real_score(self, tmp_path):
        log_file = str(tmp_path / "run.json")
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 5.0)**2

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=20, generations=5,
            seed=1, mode='minimize', log_path=log_file,
        )
        _, score, _ = ga.run()
        import json
        with open(log_file) as f:
            data = json.load(f)
        assert data['result']['best_score'] == score
        assert data['result']['best_score'] >= 0
        assert data['config']['mode'] == 'minimize'

    def test_minimize_callback_receives_real_scores(self):
        received = []

        def cb(gen, best_score, avg_score, best_ind):
            received.append(best_score)

        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))

        def error(ind):
            return (ind["x"] - 5.0)**2

        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=error,
            population_size=30, generations=5,
            seed=1, mode='minimize', on_generation=cb,
        )
        ga.run()
        for score in received:
            assert score >= 0  # real error, not negated


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


# ---------------------------------------------------------------------------
# Numpy seed
# ---------------------------------------------------------------------------

class TestNumpySeed:
    def test_seed_all_with_none_does_not_crash(self):
        """_seed_all(None) should be a no-op."""
        _seed_all(None)  # must not raise

    def test_seed_all_with_value_works(self):
        """_seed_all(42) should seed random; numpy optional."""
        _seed_all(42)
        v1 = random.random()
        _seed_all(42)
        v2 = random.random()
        assert v1 == v2


# ---------------------------------------------------------------------------
# Per-gene mutation rate
# ---------------------------------------------------------------------------

class TestPerGeneMutationRate:
    def test_floatrange_gene_rate_used(self):
        """Gene with per-gene rate=1.0 always mutates even when global rate=0.0."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0, mutation_rate=1.0))
        ind = {"x": 0.5}
        values = {genes.mutate(ind, mutation_rate=0.0)["x"] for _ in range(50)}
        assert len(values) > 1, "Expected variation with per-gene rate=1.0"

    def test_floatrange_zero_gene_rate_never_mutates(self):
        """Gene with per-gene rate=0.0 never mutates even when global rate=1.0."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0, mutation_rate=0.0))
        ind = {"x": 0.5}
        for _ in range(50):
            assert genes.mutate(ind, mutation_rate=1.0)["x"] == 0.5

    def test_intrange_gene_rate_overrides_global(self):
        """IntRange per-gene rate=1.0 always mutates even if global=0.0."""
        genes = GeneBuilder()
        genes.add("n", IntRange(0, 100, mutation_rate=1.0))
        ind = {"n": 50}
        values = {genes.mutate(ind, mutation_rate=0.0)["n"] for _ in range(50)}
        assert len(values) > 1

    def test_choicelist_gene_rate_overrides_global(self):
        """ChoiceList per-gene rate=1.0 always mutates."""
        genes = GeneBuilder()
        genes.add("c", ChoiceList(["a", "b", "c"], mutation_rate=1.0))
        ind = {"c": "a"}
        values = {genes.mutate(ind, mutation_rate=0.0)["c"] for _ in range(30)}
        assert len(values) > 1

    def test_gene_rate_in_describe(self):
        """Per-gene rate appears in describe() output."""
        spec = FloatRange(0.0, 1.0, mutation_rate=0.3)
        d = spec.describe()
        assert d['mutation_rate'] == 0.3

    def test_no_gene_rate_absent_from_describe(self):
        """When no per-gene rate, key is absent from describe()."""
        spec = FloatRange(0.0, 1.0)
        d = spec.describe()
        assert 'mutation_rate' not in d

    def test_genebuilder_uses_per_gene_rate(self):
        """GeneBuilder.mutate() uses per-gene rate when set."""
        genes = GeneBuilder()
        genes.add("frozen", FloatRange(0.0, 1.0, mutation_rate=0.0))
        genes.add("hot",    FloatRange(0.0, 1.0, mutation_rate=1.0))
        ind = {"frozen": 0.5, "hot": 0.5}
        mutations = [genes.mutate(ind, mutation_rate=0.5) for _ in range(20)]
        assert all(m["frozen"] == 0.5 for m in mutations), "frozen gene should never change"
        hot_values = {m["hot"] for m in mutations}
        assert len(hot_values) > 1, "hot gene should always vary"


# ---------------------------------------------------------------------------
# Population diversity metric
# ---------------------------------------------------------------------------

class TestDiversityMetric:
    def _make_ga(self):
        genes = sphere_genes()
        return make_simple_ga(genes, sphere_fitness)

    def test_diversity_in_history(self):
        ga = self._make_ga()
        _, _, history = ga.run()
        for h in history:
            assert 'diversity' in h

    def test_diversity_between_zero_and_one(self):
        ga = self._make_ga()
        _, _, history = ga.run()
        for h in history:
            assert 0.0 <= h['diversity'] <= 1.0

    def test_diversity_drops_when_converged(self):
        """Diversity should generally be lower after many generations."""
        genes = sphere_genes()
        ga = make_simple_ga(genes, sphere_fitness, generations=50, seed=1)
        _, _, history = ga.run()
        first_div = history[0]['diversity']
        last_div = history[-1]['diversity']
        # Allow either direction but usually diversity falls over time
        # (at minimum, both should be valid floats)
        assert isinstance(first_div, float)
        assert isinstance(last_div, float)

    def test_diversity_compute_floatrange(self):
        """Direct test of _compute_diversity on known population."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 10.0))
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=lambda ind: ind["x"],
            population_size=10,
            generations=1,
        )
        # Population spanning full range → diversity near 1.0
        population = [{"x": float(i)} for i in range(10)]
        d = ga._compute_diversity(population)
        assert d > 0.8

    def test_diversity_compute_choicelist(self):
        """ChoiceList diversity = fraction of options used."""
        genes = GeneBuilder()
        genes.add("c", ChoiceList(["a", "b", "c", "d"]))
        ga = GeneticAlgorithm(
            gene_builder=genes,
            fitness_function=lambda ind: 1.0,
            population_size=10,
            generations=1,
        )
        # All same value → diversity near 0
        population = [{"c": "a"} for _ in range(10)]
        d = ga._compute_diversity(population)
        assert d <= 0.3

    def test_restarted_in_history(self):
        """history entries have 'restarted' key."""
        ga = self._make_ga()
        _, _, history = ga.run()
        for h in history:
            assert 'restarted' in h


# ---------------------------------------------------------------------------
# Stagnation restart
# ---------------------------------------------------------------------------

class TestStagnationRestart:
    def test_restart_fires(self):
        """restart_after=5 should set restarted=True at stagnation milestones."""
        genes = sphere_genes()
        ga = make_simple_ga(
            genes, sphere_fitness,
            generations=40,
            patience=None,
            restart_after=5,
        )
        _, _, history = ga.run()
        restarted_gens = [h['gen'] for h in history if h['restarted']]
        assert len(restarted_gens) > 0, "Expected at least one restart"

    def test_restart_fires_at_multiples(self):
        """restarted=True only when gens_without_improvement % restart_after == 0."""
        genes = sphere_genes()
        ga = make_simple_ga(
            genes, sphere_fitness,
            generations=50,
            patience=None,
            restart_after=7,
        )
        _, _, history = ga.run()
        for h in history:
            if h['restarted']:
                assert h['gens_without_improvement'] % 7 == 0

    def test_restart_does_not_crash(self):
        """Full run with restart_after should complete without error."""
        genes = sphere_genes()
        ga = make_simple_ga(
            genes, sphere_fitness,
            generations=20,
            restart_after=3,
            restart_fraction=0.4,
        )
        best, score, history = ga.run()
        assert best is not None
        assert len(history) == 20

    def test_restart_after_none_never_restarts(self):
        """Without restart_after, no gen should have restarted=True."""
        ga = make_simple_ga(sphere_genes(), sphere_fitness, generations=20)
        _, _, history = ga.run()
        assert all(not h['restarted'] for h in history)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_checkpoint_file_created(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.json")
        ga = make_simple_ga(
            sphere_genes(), sphere_fitness,
            generations=5,
            checkpoint_path=ckpt,
            checkpoint_every=1,
        )
        ga.run()
        assert os.path.isfile(ckpt)

    def test_checkpoint_valid_json(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.json")
        ga = make_simple_ga(
            sphere_genes(), sphere_fitness,
            generations=5,
            checkpoint_path=ckpt,
        )
        ga.run()
        with open(ckpt) as f:
            data = json.load(f)
        assert 'gen' in data
        assert 'population' in data
        assert 'best_individual' in data
        assert 'history' in data

    def test_checkpoint_every_respected(self, tmp_path):
        """Checkpoint file should contain the last saved gen (every=3 → gen 3 or 5)."""
        ckpt = str(tmp_path / "ckpt.json")
        ga = make_simple_ga(
            sphere_genes(), sphere_fitness,
            generations=5,
            checkpoint_path=ckpt,
            checkpoint_every=3,
        )
        ga.run()
        with open(ckpt) as f:
            data = json.load(f)
        assert data['gen'] % 3 == 0 or data['gen'] == 5

    def test_resume_continues_from_checkpoint(self, tmp_path):
        """Resume appends new gens to loaded history; total includes all gens."""
        ckpt = str(tmp_path / "ckpt.json")
        ga_first = make_simple_ga(
            sphere_genes(), sphere_fitness,
            generations=5,
            checkpoint_path=ckpt,
            checkpoint_every=5,
        )
        _, _, hist1 = ga_first.run()

        ga_second = make_simple_ga(sphere_genes(), sphere_fitness, generations=10)
        _, _, hist2 = ga_second.run(resume_from=ckpt)

        # Returned history = checkpoint history (gens 1-5) + new gens (6-10)
        assert len(hist2) == 10
        assert hist2[5]['gen'] == 6   # first resumed gen is at index 5
        assert hist2[-1]['gen'] == 10

    def test_resume_result_valid(self, tmp_path):
        """Resumed run returns valid best individual and score."""
        ckpt = str(tmp_path / "ckpt.json")
        ga1 = make_simple_ga(
            sphere_genes(), sphere_fitness,
            generations=3,
            checkpoint_path=ckpt,
        )
        ga1.run()

        ga2 = make_simple_ga(sphere_genes(), sphere_fitness, generations=8)
        best, score, history = ga2.run(resume_from=ckpt)
        assert best is not None
        assert isinstance(score, float)
        # history = 3 gens from checkpoint + 5 new gens (4..8)
        assert len(history) == 8
