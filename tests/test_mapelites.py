"""Tests for MAPElites."""
import pytest
from evogine import MAPElites, FloatRange, IntRange, ChoiceList, GeneBuilder


def _make_gb_2d():
    gb = GeneBuilder()
    gb.add('x', FloatRange(0.0, 1.0))
    gb.add('y', FloatRange(0.0, 1.0))
    return gb


def _fitness(ind):
    """Simple fitness: higher x+y is better."""
    return ind['x'] + ind['y']


def _behavior(ind):
    """Behavior = (x, y), both already in [0,1]."""
    return (ind['x'], ind['y'])


def _fitness_min(ind):
    """For minimize tests."""
    return ind['x'] ** 2 + ind['y'] ** 2


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestMAPElitesInit:
    def test_basic_init(self):
        gb = _make_gb_2d()
        me = MAPElites(gb, _fitness, _behavior, grid_shape=(10, 10))
        assert me.grid_shape == (10, 10)
        assert me._total_cells == 100
        assert me.initial_population == 200
        assert me.generations == 1000

    def test_custom_params(self):
        gb = _make_gb_2d()
        me = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                       initial_population=50, generations=100, mutation_rate=0.2)
        assert me.initial_population == 50
        assert me.generations == 100
        assert me.mutation_rate == 0.2

    def test_rejects_bad_mode(self):
        gb = _make_gb_2d()
        with pytest.raises(ValueError, match="mode"):
            MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5), mode='bogus')

    def test_rejects_invalid_grid_shape(self):
        gb = _make_gb_2d()
        with pytest.raises(ValueError):
            MAPElites(gb, _fitness, _behavior, grid_shape=(0, 10))

    def test_total_cells_1d(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0, 1))
        me = MAPElites(gb, lambda ind: ind['x'], lambda ind: (ind['x'],),
                       grid_shape=(20,))
        assert me._total_cells == 20

    def test_total_cells_3d(self):
        gb = _make_gb_2d()
        gb.add('z', FloatRange(0, 1))
        me = MAPElites(gb, lambda ind: 1.0, lambda ind: (ind['x'], ind['y'], ind['z']),
                       grid_shape=(4, 5, 6))
        assert me._total_cells == 120


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

class TestMAPElitesReturnShape:
    def test_returns_two_tuple(self):
        gb = _make_gb_2d()
        result = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                           initial_population=20, generations=10, seed=1).run()
        assert len(result) == 2

    def test_archive_is_dict(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=10, seed=1).run()
        assert isinstance(archive, dict)

    def test_archive_keys_are_tuples(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=10, seed=1).run()
        for key in archive.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_archive_entry_fields(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=10, seed=1).run()
        for cell, entry in archive.items():
            assert 'individual' in entry
            assert 'score' in entry
            assert 'behavior' in entry
            assert '_internal' not in entry  # internal key must be stripped

    def test_history_structure(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=10, seed=1).run()
        assert isinstance(history, list)
        # gen 0 (seeding) + 10 generations = 11 entries
        assert len(history) == 11

    def test_history_has_required_keys(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=10, generations=5, seed=1).run()
        for entry in history:
            assert 'gen' in entry
            assert 'archive_size' in entry
            assert 'best_score' in entry
            assert 'coverage' in entry

    def test_history_gen_0_is_seeding(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=5, seed=1).run()
        assert history[0]['gen'] == 0

    def test_history_gen_sequential(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=10, generations=5, seed=1).run()
        for i, entry in enumerate(history):
            assert entry['gen'] == i


# ---------------------------------------------------------------------------
# Archive properties
# ---------------------------------------------------------------------------

class TestMAPElitesArchive:
    def test_archive_fills_over_time(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(10, 10),
                                initial_population=100, generations=200, seed=42).run()
        early_size = history[1]['archive_size']
        late_size  = history[-1]['archive_size']
        assert late_size >= early_size

    def test_coverage_between_0_and_1(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=20, seed=1).run()
        for entry in history:
            assert 0.0 <= entry['coverage'] <= 1.0

    def test_best_score_non_decreasing(self):
        gb = _make_gb_2d()
        _, history = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=30, seed=42).run()
        scores = [h['best_score'] for h in history]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 1e-12

    def test_cells_in_bounds(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(gb, _fitness, _behavior, grid_shape=(8, 8),
                                initial_population=50, generations=50, seed=1).run()
        for cell in archive.keys():
            assert 0 <= cell[0] < 8
            assert 0 <= cell[1] < 8

    def test_individual_in_bounds(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                                initial_population=20, generations=20, seed=1).run()
        for entry in archive.values():
            assert 0.0 <= entry['individual']['x'] <= 1.0
            assert 0.0 <= entry['individual']['y'] <= 1.0


# ---------------------------------------------------------------------------
# Minimize mode
# ---------------------------------------------------------------------------

class TestMAPElitesMinimize:
    def test_minimize_mode(self):
        gb = _make_gb_2d()
        archive, history = MAPElites(
            gb, _fitness_min, _behavior, grid_shape=(5, 5),
            mode='minimize', initial_population=30, generations=20, seed=1
        ).run()
        # best_score should be small (near 0 for sphere)
        assert history[-1]['best_score'] <= history[0]['best_score'] + 1e-9

    def test_minimize_scores_are_real(self):
        gb = _make_gb_2d()
        archive, _ = MAPElites(
            gb, _fitness_min, _behavior, grid_shape=(5, 5),
            mode='minimize', initial_population=20, generations=10, seed=1
        ).run()
        for entry in archive.values():
            assert entry['score'] >= 0.0  # sphere is always non-negative


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestMAPElitesReproducibility:
    def test_same_seed_same_archive(self):
        gb = _make_gb_2d()

        def run():
            return MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                             initial_population=20, generations=10, seed=7).run()

        archive1, h1 = run()
        archive2, h2 = run()
        assert set(archive1.keys()) == set(archive2.keys())
        assert h1[-1]['best_score'] == h2[-1]['best_score']


# ---------------------------------------------------------------------------
# Mixed gene types
# ---------------------------------------------------------------------------

class TestMAPElitesMixedGenes:
    def test_mixed_int_float(self):
        gb = GeneBuilder()
        gb.add('x', FloatRange(0.0, 1.0))
        gb.add('n', IntRange(0, 9))

        def fitness(ind):
            return ind['x'] + ind['n'] / 9.0

        def behavior(ind):
            return (ind['x'], ind['n'] / 9.0)

        archive, history = MAPElites(
            gb, fitness, behavior, grid_shape=(5, 5),
            initial_population=30, generations=20, seed=1
        ).run()
        assert len(archive) > 0


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TestMAPElitesCallback:
    def test_on_generation_called(self):
        gb = _make_gb_2d()
        calls = []

        def cb(gen, archive_size, best_score, coverage):
            calls.append((gen, archive_size, best_score, coverage))

        MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                  initial_population=10, generations=5, seed=1,
                  on_generation=cb).run()
        assert len(calls) == 5
        assert calls[0][0] == 1
        assert calls[-1][0] == 5

    def test_callback_coverage_in_range(self):
        gb = _make_gb_2d()
        coverages = []

        def cb(gen, archive_size, best_score, coverage):
            coverages.append(coverage)

        MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                  initial_population=10, generations=5, seed=1,
                  on_generation=cb).run()
        assert all(0.0 <= c <= 1.0 for c in coverages)


# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------

class TestMAPElitesLog:
    def test_log_written(self, tmp_path):
        import json
        log_file = str(tmp_path / "mapelites_log.json")
        gb = _make_gb_2d()
        MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                  initial_population=10, generations=5, seed=1,
                  log_path=log_file).run()
        with open(log_file) as f:
            log = json.load(f)
        assert log['run']['type'] == 'map_elites'
        assert 'config' in log
        assert 'result' in log
        assert 'history' in log

    def test_log_result_fields(self, tmp_path):
        import json
        log_file = str(tmp_path / "mapelites_log2.json")
        gb = _make_gb_2d()
        MAPElites(gb, _fitness, _behavior, grid_shape=(5, 5),
                  initial_population=10, generations=5, seed=1,
                  log_path=log_file).run()
        with open(log_file) as f:
            log = json.load(f)
        result = log['result']
        assert 'archive_size' in result
        assert 'coverage' in result
        assert 'best_score' in result
