"""Tests for the evogine benchmark suite.

Verifies function correctness at known optima, constraint validity,
multi-objective output shapes, and runner integration.

Usage::

    pytest tests/test_benchmarks.py -v
"""

import math

import pytest

from evogine.benchmarks import functions as F
from evogine.benchmarks import engineering as E
from evogine.benchmarks import multi_objective as MO
from evogine.benchmarks.problems import CLASSIC, ENGINEERING, MULTI_OBJECTIVE, QD, ALL
from evogine.benchmarks.runner import (
    run_suite, _make_genes, _wrap_fn, _wrap_constraints, _make_behavior_fn,
)


# -----------------------------------------------------------------------
# Single-objective functions — values at known optima
# -----------------------------------------------------------------------

class TestFunctionsAtOptima:
    """Each function should return (approximately) its known minimum at the optimal point."""

    def test_sphere(self):
        assert F.sphere([0.0, 0.0, 0.0]) == 0.0

    def test_rosenbrock(self):
        assert F.rosenbrock([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_rastrigin(self):
        assert F.rastrigin([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_ackley(self):
        assert F.ackley([0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-10)

    def test_schwefel(self):
        opt = [420.9687] * 3
        assert F.schwefel(opt) == pytest.approx(0.0, abs=0.01)

    def test_griewank(self):
        assert F.griewank([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_levy(self):
        assert F.levy([1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_michalewicz_2d(self):
        # Known optimum for 2D: approximately -1.8013
        # at (2.20, 1.57)
        val = F.michalewicz([2.20, 1.57])
        assert val < -1.7  # should be close to -1.8013

    def test_styblinski_tang(self):
        opt = [-2.903534] * 3
        expected = -39.16617 * 3
        assert F.styblinski_tang(opt) == pytest.approx(expected, rel=1e-4)

    def test_zakharov(self):
        assert F.zakharov([0.0, 0.0, 0.0]) == 0.0

    def test_dixon_price(self):
        # Known minimum is 0 at x_i = 2^(-(2^i - 2)/2^i)
        x = [2 ** (-(2 ** i - 2) / 2 ** i) for i in range(1, 4)]
        assert F.dixon_price(x) == pytest.approx(0.0, abs=1e-6)


class TestFunctionsAwayFromOptima:
    """Each function should return a positive value away from the optimum."""

    def test_sphere_nonzero(self):
        assert F.sphere([1.0, 1.0]) > 0

    def test_rosenbrock_nonzero(self):
        assert F.rosenbrock([0.0, 0.0]) > 0

    def test_rastrigin_nonzero(self):
        assert F.rastrigin([1.0, 1.0]) > 0

    def test_ackley_nonzero(self):
        assert F.ackley([1.0, 1.0]) > 0

    def test_schwefel_nonzero(self):
        assert F.schwefel([0.0, 0.0]) > 0

    def test_griewank_nonzero(self):
        assert F.griewank([10.0, 10.0]) > 0

    def test_levy_nonzero(self):
        assert F.levy([0.0, 0.0]) > 0

    def test_zakharov_nonzero(self):
        assert F.zakharov([1.0, 1.0]) > 0

    def test_dixon_price_nonzero(self):
        assert F.dixon_price([0.0, 0.0]) > 0


class TestFunctionsDimensionality:
    """Functions should work at any dimension >= 2."""

    @pytest.mark.parametrize("dim", [2, 5, 10, 20])
    def test_sphere_dims(self, dim):
        assert F.sphere([0.0] * dim) == 0.0

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_rastrigin_dims(self, dim):
        assert F.rastrigin([0.0] * dim) == pytest.approx(0.0)

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_ackley_dims(self, dim):
        assert F.ackley([0.0] * dim) == pytest.approx(0.0, abs=1e-10)


# -----------------------------------------------------------------------
# Engineering constraints
# -----------------------------------------------------------------------

class TestWeldedBeam:
    def test_cost_at_known_optimum(self):
        x = [0.2057, 3.4705, 9.0366, 0.2057]
        cost = E.welded_beam_cost(x)
        assert 1.5 < cost < 2.5

    def test_constraints_near_optimum(self):
        x = [0.2057, 3.4705, 9.0366, 0.2057]
        c = E.welded_beam_constraints(x)
        assert len(c) == 7
        assert all(isinstance(v, bool) for v in c)

    def test_infeasible_point(self):
        # Very thin weld, long beam — should violate stress constraints
        x = [0.1, 10.0, 0.1, 0.1]
        c = E.welded_beam_constraints(x)
        assert not all(c)


class TestPressureVessel:
    def test_cost_positive(self):
        x = [1.0, 1.0, 50.0, 100.0]
        assert E.pressure_vessel_cost(x) > 0

    def test_constraints_count(self):
        x = [1.0, 1.0, 50.0, 100.0]
        c = E.pressure_vessel_constraints(x)
        assert len(c) == 4

    def test_length_constraint(self):
        # L=250 > 240 → constraint 4 violated
        x = [1.0, 1.0, 50.0, 250.0]
        c = E.pressure_vessel_constraints(x)
        assert c[3] is False


class TestSpring:
    def test_cost_at_known_optimum(self):
        x = [0.05169, 0.35673, 11.2885]
        cost = E.spring_cost(x)
        assert cost == pytest.approx(0.012665, rel=1e-3)

    def test_constraints_count(self):
        x = [0.1, 0.5, 10.0]
        c = E.spring_constraints(x)
        assert len(c) == 4


# -----------------------------------------------------------------------
# Multi-objective functions
# -----------------------------------------------------------------------

class TestZDT:
    def test_zdt1_shape(self):
        result = MO.zdt1([0.5] + [0.0] * 29)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.5)
        assert result[1] >= 0

    def test_zdt1_pareto_point(self):
        # On Pareto front: x_i=0 for i>=1
        r = MO.zdt1([0.25] + [0.0] * 29)
        assert r[0] == pytest.approx(0.25)
        assert r[1] == pytest.approx(1 - math.sqrt(0.25))

    def test_zdt2_shape(self):
        r = MO.zdt2([0.5] + [0.0] * 29)
        assert len(r) == 2

    def test_zdt2_pareto_point(self):
        r = MO.zdt2([0.25] + [0.0] * 29)
        assert r[1] == pytest.approx(1 - 0.25 ** 2)

    def test_zdt3_shape(self):
        r = MO.zdt3([0.5] + [0.0] * 29)
        assert len(r) == 2

    def test_zdt6_shape(self):
        r = MO.zdt6([0.5] + [0.0] * 9)
        assert len(r) == 2
        assert r[0] > 0


class TestDTLZ:
    def test_dtlz1_shape(self):
        # n=7, n_obj=3 → k=5
        r = MO.dtlz1([0.5] * 7, n_obj=3)
        assert len(r) == 3

    def test_dtlz1_pareto_point(self):
        # On Pareto front: x_m = 0.5 → g=0, sum(f_i) = 0.5
        x = [0.5, 0.5] + [0.5] * 5
        r = MO.dtlz1(x, n_obj=3)
        assert sum(r) == pytest.approx(0.5, abs=0.01)

    def test_dtlz2_shape(self):
        r = MO.dtlz2([0.5] * 12, n_obj=3)
        assert len(r) == 3

    def test_dtlz2_pareto_point(self):
        # On Pareto front: x_m = 0.5 → g=0, sum(f_i^2) = 1
        x = [0.5, 0.5] + [0.5] * 10
        r = MO.dtlz2(x, n_obj=3)
        assert sum(fi ** 2 for fi in r) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("n_obj", [2, 3, 5])
    def test_dtlz2_scalable(self, n_obj):
        n = n_obj - 1 + 10
        r = MO.dtlz2([0.5] * n, n_obj=n_obj)
        assert len(r) == n_obj


# -----------------------------------------------------------------------
# Problem registry
# -----------------------------------------------------------------------

class TestProblemRegistry:
    def test_classic_count(self):
        assert len(CLASSIC) >= 13

    def test_engineering_count(self):
        assert len(ENGINEERING) == 3

    def test_multi_objective_count(self):
        assert len(MULTI_OBJECTIVE) >= 6

    def test_qd_count(self):
        assert len(QD) >= 2

    def test_all_keys(self):
        assert set(ALL.keys()) == {'classic', 'engineering', 'multi_objective', 'qd'}

    def test_every_problem_has_name(self):
        for cat, problems in ALL.items():
            for p in problems:
                assert p.name, f"Problem in {cat} has no name"
                assert p.dim > 0
                assert len(p.bounds) == p.dim


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class TestHelpers:
    def test_make_genes(self):
        p = CLASSIC[0]  # Sphere 2D
        genes = _make_genes(p)
        assert len(genes.specs) == p.dim

    def test_wrap_fn(self):
        p = CLASSIC[0]
        fn = _wrap_fn(p)
        ind = {'x0': 0.0, 'x1': 0.0}
        assert fn(ind) == 0.0

    def test_wrap_fn_nonzero(self):
        p = CLASSIC[0]
        fn = _wrap_fn(p)
        ind = {'x0': 1.0, 'x1': 2.0}
        assert fn(ind) == 5.0

    def test_wrap_constraints(self):
        p = ENGINEERING[0]  # Welded Beam
        constraints = _wrap_constraints(p)
        assert len(constraints) == 7
        ind = {f'x{i}': v for i, v in enumerate([0.2057, 3.4705, 9.0366, 0.2057])}
        results = [c(ind) for c in constraints]
        assert all(isinstance(r, bool) for r in results)

    def test_make_behavior_fn(self):
        p = QD[0]
        bfn = _make_behavior_fn(p)
        ind = {'x0': 0.0, 'x1': 0.0, 'x2': 0.0, 'x3': 0.0, 'x4': 0.0}
        b = bfn(ind)
        assert len(b) == 2
        assert 0.0 <= b[0] <= 1.0
        assert 0.0 <= b[1] <= 1.0


# -----------------------------------------------------------------------
# Runner integration (small budget, fast)
# -----------------------------------------------------------------------

class TestRunnerIntegration:
    def test_classic_single(self):
        results = run_suite(
            categories=['classic'],
            eval_budget=500,
            save=False,
        )
        assert len(results) > 0
        assert all(r.category == 'classic' for r in results)

    def test_engineering_runs(self):
        results = run_suite(
            categories=['engineering'],
            eval_budget=500,
            save=False,
        )
        assert len(results) == 3
        assert all(r.category == 'engineering' for r in results)

    def test_multi_objective_runs(self):
        results = run_suite(
            categories=['multi_objective'],
            eval_budget=500,
            save=False,
        )
        assert len(results) >= 6
        for r in results:
            assert r.extra.get('pareto_size', 0) > 0

    def test_qd_runs(self):
        results = run_suite(
            categories=['qd'],
            eval_budget=500,
            save=False,
        )
        assert len(results) == 2
        for r in results:
            assert r.extra.get('archive_size', 0) > 0
            assert r.extra.get('coverage', 0) > 0
