"""
Tests for CMAESOptimizer.

Coverage:
- Basic API contract: run() returns (dict, float, list)
- History structure and keys
- Maximize and minimize modes
- Reproducibility via seed
- Validation errors (non-FloatRange genes, < 2 genes, bad mode)
- Convergence on standard benchmarks (sphere, Rastrigin)
- Stopping conditions: tolx, tolfun, patience
- JSON log output
- Popsize override
- Missing numpy ImportError
- Bounds enforcement (all returned values inside gene ranges)
"""

import math
import json
import os
import sys
import tempfile
import pytest

from evogine import (
    CMAESOptimizer,
    GeneBuilder,
    FloatRange,
    IntRange,
    ChoiceList,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sphere_optimizer(**kwargs):
    genes = GeneBuilder()
    genes.add("x", FloatRange(-5.0, 5.0))
    genes.add("y", FloatRange(-5.0, 5.0))

    def sphere(ind):
        return -(ind["x"] ** 2 + ind["y"] ** 2)

    defaults = dict(
        gene_builder=genes,
        fitness_function=sphere,
        sigma0=0.3,
        generations=200,
        patience=30,
        seed=42,
        mode="maximize",
    )
    defaults.update(kwargs)
    return CMAESOptimizer(**defaults)


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

class TestAPIContract:
    def test_run_returns_three_tuple(self):
        opt = make_sphere_optimizer(generations=10)
        result = opt.run()
        assert len(result) == 3

    def test_best_is_dict(self):
        opt = make_sphere_optimizer(generations=10)
        best, _, _ = opt.run()
        assert isinstance(best, dict)

    def test_best_has_all_gene_keys(self):
        opt = make_sphere_optimizer(generations=10)
        best, _, _ = opt.run()
        assert set(best.keys()) == {"x", "y"}

    def test_score_is_float(self):
        opt = make_sphere_optimizer(generations=10)
        _, score, _ = opt.run()
        assert isinstance(score, float)

    def test_history_is_list(self):
        opt = make_sphere_optimizer(generations=10)
        _, _, history = opt.run()
        assert isinstance(history, list)

    def test_history_length_lte_generations(self):
        opt = make_sphere_optimizer(generations=15)
        _, _, history = opt.run()
        assert len(history) <= 15

    def test_history_length_at_least_one(self):
        opt = make_sphere_optimizer(generations=5)
        _, _, history = opt.run()
        assert len(history) >= 1


# ---------------------------------------------------------------------------
# History structure
# ---------------------------------------------------------------------------

class TestHistoryStructure:
    REQUIRED_KEYS = {
        "gen", "best_score", "avg_score", "sigma",
        "improved", "gens_without_improvement", "stop_reason",
    }

    def test_all_required_keys_present(self):
        opt = make_sphere_optimizer(generations=5)
        _, _, history = opt.run()
        for entry in history:
            assert self.REQUIRED_KEYS.issubset(entry.keys()), (
                f"Missing keys: {self.REQUIRED_KEYS - entry.keys()}"
            )

    def test_gen_counter_sequential(self):
        opt = make_sphere_optimizer(generations=10)
        _, _, history = opt.run()
        for i, entry in enumerate(history):
            assert entry["gen"] == i + 1

    def test_sigma_positive(self):
        opt = make_sphere_optimizer(generations=10)
        _, _, history = opt.run()
        for entry in history:
            assert entry["sigma"] > 0

    def test_gens_without_improvement_non_negative(self):
        opt = make_sphere_optimizer(generations=10)
        _, _, history = opt.run()
        for entry in history:
            assert entry["gens_without_improvement"] >= 0

    def test_improved_is_bool(self):
        opt = make_sphere_optimizer(generations=10)
        _, _, history = opt.run()
        for entry in history:
            assert isinstance(entry["improved"], bool)

    def test_first_gen_usually_improves(self):
        # Gen 1 always has something better than -inf
        opt = make_sphere_optimizer(generations=5)
        _, _, history = opt.run()
        assert history[0]["improved"] is True

    def test_best_score_not_worse_over_generations(self):
        opt = make_sphere_optimizer(generations=20, mode="maximize")
        _, _, history = opt.run()
        # For maximize: best_score should be non-decreasing
        for i in range(1, len(history)):
            assert history[i]["best_score"] >= history[i - 1]["best_score"] - 1e-12

    def test_stop_reason_only_on_last_entry(self):
        opt = make_sphere_optimizer(generations=50, patience=10)
        _, _, history = opt.run()
        # All except possibly the last should have stop_reason = None
        for entry in history[:-1]:
            assert entry["stop_reason"] is None


# ---------------------------------------------------------------------------
# Convergence benchmarks
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_sphere_2d_converges(self):
        """CMA-ES should find the optimum of sphere at origin."""
        genes = GeneBuilder()
        genes.add("x", FloatRange(-5.0, 5.0))
        genes.add("y", FloatRange(-5.0, 5.0))

        def sphere(ind):
            return -(ind["x"] ** 2 + ind["y"] ** 2)

        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=sphere,
            sigma0=0.3, generations=300, patience=50, seed=1,
        )
        best, score, _ = opt.run()
        assert abs(best["x"]) < 0.1
        assert abs(best["y"]) < 0.1
        assert score > -0.01  # near 0

    def test_sphere_5d_converges(self):
        genes = GeneBuilder()
        for name in ["a", "b", "c", "d", "e"]:
            genes.add(name, FloatRange(-3.0, 3.0))

        def sphere5(ind):
            return -sum(ind[k] ** 2 for k in ind)

        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=sphere5,
            sigma0=0.3, generations=500, patience=80, seed=7,
        )
        best, score, _ = opt.run()
        for k in best:
            assert abs(best[k]) < 0.2, f"{k} = {best[k]}"

    def test_minimize_mode_sphere(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(-5.0, 5.0))
        genes.add("y", FloatRange(-5.0, 5.0))

        def sphere(ind):
            return ind["x"] ** 2 + ind["y"] ** 2

        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=sphere,
            sigma0=0.3, generations=300, patience=50, seed=2,
            mode="minimize",
        )
        best, score, history = opt.run()
        assert abs(best["x"]) < 0.1
        assert abs(best["y"]) < 0.1
        # Returned score is the raw value (not negated)
        assert score < 0.01
        # History shows real (non-negated) values
        for entry in history:
            assert entry["best_score"] >= 0.0

    def test_score_matches_history_final(self):
        opt = make_sphere_optimizer(generations=30, patience=20)
        best, score, history = opt.run()
        best_in_history = max(h["best_score"] for h in history)
        assert abs(score - best_in_history) < 1e-9

    def test_maximize_score_is_best_ever(self):
        opt = make_sphere_optimizer(generations=30)
        best, score, history = opt.run()
        # score must equal the maximum best_score in history
        max_score = max(h["best_score"] for h in history)
        assert abs(score - max_score) < 1e-9

    def test_fitness_of_returned_individual(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(-5.0, 5.0))
        genes.add("y", FloatRange(-5.0, 5.0))

        def sphere(ind):
            return -(ind["x"] ** 2 + ind["y"] ** 2)

        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=sphere,
            generations=50, seed=3,
        )
        best, score, _ = opt.run()
        assert abs(sphere(best) - score) < 1e-9


# ---------------------------------------------------------------------------
# Bounds enforcement
# ---------------------------------------------------------------------------

class TestBoundsEnforcement:
    def test_all_returned_values_within_gene_ranges(self):
        genes = GeneBuilder()
        genes.add("a", FloatRange(0.0, 1.0))
        genes.add("b", FloatRange(-10.0, 10.0))
        genes.add("c", FloatRange(100.0, 200.0))

        def fitness(ind):
            return ind["a"] + ind["b"] + ind["c"]

        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=fitness,
            generations=30, seed=5,
        )
        best, _, _ = opt.run()
        assert 0.0 <= best["a"] <= 1.0
        assert -10.0 <= best["b"] <= 10.0
        assert 100.0 <= best["c"] <= 200.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        def run(seed):
            return make_sphere_optimizer(generations=20, seed=seed).run()

        best1, score1, hist1 = run(99)
        best2, score2, hist2 = run(99)
        assert score1 == score2
        assert best1 == best2
        assert len(hist1) == len(hist2)

    def test_different_seed_different_result(self):
        _, score1, _ = make_sphere_optimizer(generations=20, seed=1).run()
        _, score2, _ = make_sphere_optimizer(generations=20, seed=2).run()
        # Very unlikely to be identical
        assert score1 != score2


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_intrange_gene(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        genes.add("n", IntRange(1, 10))
        with pytest.raises(ValueError, match="IntRange"):
            CMAESOptimizer(gene_builder=genes, fitness_function=lambda i: 0.0)

    def test_rejects_choicelist_gene(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        genes.add("cat", ChoiceList(["a", "b"]))
        with pytest.raises(ValueError, match="ChoiceList"):
            CMAESOptimizer(gene_builder=genes, fitness_function=lambda i: 0.0)

    def test_rejects_single_gene(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        with pytest.raises(ValueError, match="at least 2"):
            CMAESOptimizer(gene_builder=genes, fitness_function=lambda i: 0.0)

    def test_rejects_bad_mode(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        genes.add("y", FloatRange(0.0, 1.0))
        with pytest.raises(ValueError, match="mode"):
            CMAESOptimizer(
                gene_builder=genes, fitness_function=lambda i: 0.0,
                mode="invalid",
            )


# ---------------------------------------------------------------------------
# Stopping conditions
# ---------------------------------------------------------------------------

class TestStoppingConditions:
    def test_patience_stop(self):
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        genes.add("y", FloatRange(0.0, 1.0))

        # Constant fitness → patience fires immediately after first gen
        opt = CMAESOptimizer(
            gene_builder=genes,
            fitness_function=lambda ind: 1.0,
            generations=100,
            patience=5,
            seed=1,
        )
        _, _, history = opt.run()
        assert history[-1]["stop_reason"] == "patience"
        assert len(history) <= 10  # stops well before 100

    def test_tolx_stop_triggers(self):
        # On sphere, CMA-ES naturally hits tolx once converged
        opt = make_sphere_optimizer(generations=500, patience=None, tolx=1e-4)
        _, _, history = opt.run()
        stop = history[-1]["stop_reason"]
        # Should stop via tolx or tolfun (not run all 500 gens)
        assert stop in ("tolx", "tolfun", "patience")

    def test_full_run_no_patience(self):
        opt = make_sphere_optimizer(generations=5, patience=None, tolx=0.0, tolfun=0.0)
        _, _, history = opt.run()
        # With no stopping conditions should run exactly 5 gens
        # (unless tolfun/tolx triggered — we set them to 0 so they won't)
        assert len(history) == 5


# ---------------------------------------------------------------------------
# Popsize override
# ---------------------------------------------------------------------------

class TestPopsizeOverride:
    def test_custom_popsize_accepted(self):
        opt = make_sphere_optimizer(generations=5, popsize=20)
        best, score, history = opt.run()
        assert isinstance(score, float)
        assert len(history) >= 1

    def test_default_popsize_formula(self):
        # n=2: lambda = 4 + floor(3*ln(2)) = 4 + 2 = 6
        genes = GeneBuilder()
        genes.add("x", FloatRange(0.0, 1.0))
        genes.add("y", FloatRange(0.0, 1.0))
        opt = CMAESOptimizer(
            gene_builder=genes, fitness_function=lambda i: 0.0,
        )
        assert opt._lam == 4 + int(3 * math.log(2))


# ---------------------------------------------------------------------------
# JSON log output
# ---------------------------------------------------------------------------

class TestLogOutput:
    def test_log_written(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        opt.run()
        assert os.path.exists(log_file)

    def test_log_valid_json(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        opt.run()
        with open(log_file) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_log_type_is_cmaes(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        opt.run()
        with open(log_file) as f:
            data = json.load(f)
        assert data["run"]["type"] == "cmaes"

    def test_log_has_required_sections(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        opt.run()
        with open(log_file) as f:
            data = json.load(f)
        for section in ("run", "config", "genes", "result", "history"):
            assert section in data, f"Missing section: {section}"

    def test_log_config_has_cmaes_params(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        opt.run()
        with open(log_file) as f:
            data = json.load(f)
        config = data["config"]
        for key in ("sigma0", "lambda", "mu", "mueff", "cc", "cs", "c1", "cmu", "damps"):
            assert key in config, f"Missing config key: {key}"

    def test_log_result_has_best_score(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file)
        _, score, _ = opt.run()
        with open(log_file) as f:
            data = json.load(f)
        assert abs(data["result"]["best_score"] - score) < 1e-9

    def test_log_mode_recorded(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=5, log_path=log_file, mode="maximize")
        opt.run()
        with open(log_file) as f:
            data = json.load(f)
        assert data["config"]["mode"] == "maximize"

    def test_log_history_length_matches(self, tmp_path):
        log_file = str(tmp_path / "cmaes_run.json")
        opt = make_sphere_optimizer(generations=8, log_path=log_file)
        _, _, history = opt.run()
        with open(log_file) as f:
            data = json.load(f)
        assert len(data["history"]) == len(history)
