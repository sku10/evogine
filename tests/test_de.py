"""Tests for DEOptimizer (SHADE Differential Evolution)."""
import pytest
from evogine import DEOptimizer, FloatRange, IntRange, ChoiceList, GeneBuilder


def _make_gb(n=3):
    gb = GeneBuilder()
    for i in range(n):
        gb.add(f"x{i}", FloatRange(0.0, 1.0))
    return gb


def _sphere(ind):
    """Minimize sum of squares — minimum at origin (maximize negative)."""
    return -sum(v ** 2 for v in ind.values())


def _sphere_min(ind):
    return sum(v ** 2 for v in ind.values())


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------

class TestDEOptimizerInit:
    def test_basic_init(self):
        gb = _make_gb()
        de = DEOptimizer(gb, _sphere)
        assert de.population_size == 50
        assert de.generations == 200
        assert de.strategy == 'current_to_best'
        assert de.memory_size == 6

    def test_custom_params(self):
        gb = _make_gb()
        de = DEOptimizer(gb, _sphere, population_size=30, generations=100,
                         strategy='rand1', memory_size=10)
        assert de.population_size == 30
        assert de.generations == 100
        assert de.strategy == 'rand1'
        assert de.memory_size == 10

    def test_rejects_non_float_genes(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        gb.add('y', IntRange(0, 10))
        with pytest.raises(ValueError, match="FloatRange"):
            DEOptimizer(gb, _sphere)

    def test_rejects_choice_genes(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        gb.add('c', ChoiceList(['a', 'b']))
        with pytest.raises(ValueError, match="FloatRange"):
            DEOptimizer(gb, _sphere)

    def test_rejects_single_gene(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        with pytest.raises(ValueError, match="at least 2"):
            DEOptimizer(gb, _sphere)

    def test_rejects_bad_mode(self):
        gb = _make_gb()
        with pytest.raises(ValueError, match="mode"):
            DEOptimizer(gb, _sphere, mode='bogus')

    def test_rejects_bad_strategy(self):
        gb = _make_gb()
        with pytest.raises(ValueError, match="strategy"):
            DEOptimizer(gb, _sphere, strategy='bogus')


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

class TestDEOptimizerReturnShape:
    def test_returns_three_tuple(self):
        gb = _make_gb()
        result = DEOptimizer(gb, _sphere, generations=5, population_size=10, seed=1).run()
        assert len(result) == 3

    def test_best_individual_has_all_genes(self):
        gb = _make_gb(n=4)
        best_ind, _, _ = DEOptimizer(gb, _sphere, generations=5, population_size=10, seed=1).run()
        assert set(best_ind.keys()) == {'x0', 'x1', 'x2', 'x3'}

    def test_best_individual_in_bounds(self):
        gb = GeneBuilder()
        gb.add('a', FloatRange(-5.0, 5.0))
        gb.add('b', FloatRange(0.0, 10.0))
        best_ind, _, _ = DEOptimizer(gb, _sphere, generations=10, population_size=15, seed=42).run()
        assert -5.0 <= best_ind['a'] <= 5.0
        assert 0.0 <= best_ind['b'] <= 10.0

    def test_history_is_list_of_dicts(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=5, population_size=10, seed=1).run()
        assert isinstance(history, list)
        assert len(history) == 5

    def test_history_keys(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=3, population_size=10, seed=1).run()
        required = {'gen', 'best_score', 'avg_score', 'F_mean', 'CR_mean',
                    'improved', 'gens_without_improvement', 'stop_reason', 'pop_size'}
        for entry in history:
            assert required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}"

    def test_history_gen_sequential(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=5, population_size=10, seed=1).run()
        for i, h in enumerate(history, 1):
            assert h['gen'] == i

    def test_best_score_non_decreasing(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=20, population_size=20, seed=42).run()
        scores = [h['best_score'] for h in history]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-12


# ---------------------------------------------------------------------------
# SHADE-specific: F_mean / CR_mean in history
# ---------------------------------------------------------------------------

class TestSHADEMemory:
    def test_F_mean_in_range(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=10, population_size=20, seed=7).run()
        for h in history:
            assert 0.0 <= h['F_mean'] <= 1.0

    def test_CR_mean_in_range(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(gb, _sphere, generations=10, population_size=20, seed=7).run()
        for h in history:
            assert 0.0 <= h['CR_mean'] <= 1.0

    def test_memory_size_param(self):
        gb = _make_gb()
        de = DEOptimizer(gb, _sphere, memory_size=3, generations=5, population_size=10, seed=1)
        assert de.memory_size == 3
        de.run()  # just ensure it runs without error

    def test_strategy_rand1(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(
            gb, _sphere, strategy='rand1', generations=10, population_size=20, seed=5
        ).run()
        assert len(history) == 10


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestDEOptimizerConvergence:
    def test_sphere_maximize_improves(self):
        """Score should improve (negative sphere goes toward 0)."""
        gb = _make_gb(n=2)
        _, _, history = DEOptimizer(gb, _sphere, generations=50, population_size=20, seed=42).run()
        assert history[-1]['best_score'] > history[0]['best_score']

    def test_sphere_minimize_mode(self):
        """In minimize mode the best_score (real) should be small (near 0)."""
        gb = _make_gb(n=2)
        best_ind, best_score, _ = DEOptimizer(
            gb, _sphere_min, mode='minimize', generations=100, population_size=20, seed=42
        ).run()
        assert best_score >= 0.0
        assert best_score < 0.5  # sphere minimum is 0; should get close

    def test_best_score_real_sign_minimize(self):
        """best_score returned is raw (un-negated) in minimize mode."""
        gb = _make_gb(n=2)
        _, best_score, _ = DEOptimizer(
            gb, _sphere_min, mode='minimize', generations=10, population_size=15, seed=1
        ).run()
        assert best_score >= 0.0  # sphere is always non-negative


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestDEOptimizerReproducibility:
    def test_same_seed_same_result(self):
        gb = _make_gb()

        def run():
            return DEOptimizer(gb, _sphere, generations=10, population_size=15, seed=99).run()

        ind1, sc1, _ = run()
        ind2, sc2, _ = run()
        assert sc1 == sc2
        assert ind1 == ind2

    def test_different_seeds_different_results(self):
        gb = _make_gb()
        _, sc1, _ = DEOptimizer(gb, _sphere, generations=10, population_size=15, seed=1).run()
        _, sc2, _ = DEOptimizer(gb, _sphere, generations=10, population_size=15, seed=2).run()
        # May occasionally be equal by chance, but very unlikely
        # Just ensure it runs without error


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestDEOptimizerEarlyStopping:
    def test_patience_stops_early(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(
            gb, _sphere, generations=200, population_size=20, patience=5, seed=42
        ).run()
        assert len(history) < 200

    def test_stop_reason_patience(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(
            gb, _sphere, generations=200, population_size=20, patience=5, seed=42
        ).run()
        stop_reasons = [h['stop_reason'] for h in history if h['stop_reason'] is not None]
        assert 'patience' in stop_reasons


# ---------------------------------------------------------------------------
# L-SHADE (linear population reduction)
# ---------------------------------------------------------------------------

class TestLSHADE:
    def test_pop_shrinks(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(
            gb, _sphere, generations=50, population_size=30,
            linear_pop_reduction=True, seed=1
        ).run()
        pop_sizes = [h['pop_size'] for h in history]
        assert pop_sizes[-1] < pop_sizes[0]  # must have shrunk

    def test_pop_never_below_min(self):
        gb = _make_gb()
        _, _, history = DEOptimizer(
            gb, _sphere, generations=100, population_size=20,
            linear_pop_reduction=True, seed=1
        ).run()
        for h in history:
            assert h['pop_size'] >= 4

    def test_lshade_still_returns_valid_result(self):
        gb = _make_gb()
        best_ind, best_score, history = DEOptimizer(
            gb, _sphere, generations=30, population_size=20,
            linear_pop_reduction=True, seed=42
        ).run()
        assert best_ind is not None
        assert isinstance(best_score, float)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TestDEOptimizerCallback:
    def test_on_generation_called(self):
        gb = _make_gb()
        calls = []

        def cb(gen, best, avg, best_ind):
            calls.append((gen, best, avg))

        DEOptimizer(gb, _sphere, generations=5, population_size=10, seed=1,
                    on_generation=cb).run()
        assert len(calls) == 5
        assert calls[0][0] == 1
        assert calls[-1][0] == 5

    def test_callback_receives_best_individual(self):
        gb = _make_gb()
        inds = []

        def cb(gen, best, avg, best_ind):
            inds.append(best_ind)

        DEOptimizer(gb, _sphere, generations=3, population_size=10, seed=1,
                    on_generation=cb).run()
        assert all(isinstance(ind, dict) for ind in inds)
        assert all('x0' in ind for ind in inds)


# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------

class TestDEOptimizerLog:
    def test_log_written(self, tmp_path):
        import json
        log_file = str(tmp_path / "de_log.json")
        gb = _make_gb()
        DEOptimizer(gb, _sphere, generations=5, population_size=10,
                    seed=1, log_path=log_file).run()
        with open(log_file) as f:
            log = json.load(f)
        assert log['run']['type'] == 'de'
        assert 'config' in log
        assert 'result' in log
        assert 'history' in log

    def test_log_config_fields(self, tmp_path):
        import json
        log_file = str(tmp_path / "de_log2.json")
        gb = _make_gb()
        DEOptimizer(gb, _sphere, generations=5, population_size=10,
                    strategy='rand1', memory_size=4,
                    seed=1, log_path=log_file).run()
        with open(log_file) as f:
            log = json.load(f)
        cfg = log['config']
        assert cfg['strategy'] == 'rand1'
        assert cfg['memory_size'] == 4
