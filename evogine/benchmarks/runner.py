"""Benchmark suite runner.

Runs evogine optimizers against registered problems and produces formatted
tables + JSON results.

Usage::

    from evogine.benchmarks import run_suite
    results = run_suite()                          # all categories
    results = run_suite(categories=['classic'])     # just classic
    results = run_suite(eval_budget=10000)          # more evaluations
"""

import contextlib
import io
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from evogine import (
    GeneticAlgorithm, CMAESOptimizer, DEOptimizer, IslandModel,
    MultiObjectiveGA, MAPElites,
    GeneBuilder, FloatRange, TournamentSelection, ArithmeticCrossover,
)
from .problems import Problem, ALL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genes(problem):
    gb = GeneBuilder()
    for i, (lo, hi) in enumerate(problem.bounds):
        gb.add(f'x{i}', FloatRange(lo, hi))
    return gb


def _wrap_fn(problem):
    """Wrap list-based function for evogine's dict interface."""
    names = [f'x{i}' for i in range(problem.dim)]
    raw = problem.fn

    def wrapped(ind):
        return raw([ind[n] for n in names])
    return wrapped


def _wrap_constraints(problem):
    """Create list of evogine constraint functions from a batch constraint fn."""
    if not problem.constraints_fn:
        return None
    names = [f'x{i}' for i in range(problem.dim)]
    raw = problem.constraints_fn

    def _make(idx):
        def constraint(ind):
            return raw([ind[n] for n in names])[idx]
        return constraint

    return [_make(i) for i in range(problem.n_constraints)]


def _make_behavior_fn(problem):
    """Behavior function for QD: normalize first 2 genes to [0, 1]."""
    lo, hi = problem.bounds[0]
    span = hi - lo

    def behavior(ind):
        return (
            max(0.0, min(1.0, (ind['x0'] - lo) / span)),
            max(0.0, min(1.0, (ind['x1'] - lo) / span)),
        )
    return behavior


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    optimizer: str
    problem: str
    category: str
    best_score: float
    evals: int
    converged: bool
    close: bool
    elapsed: float
    extra: dict = None  # category-specific metrics

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


# ---------------------------------------------------------------------------
# Classic single-objective runner
# ---------------------------------------------------------------------------

_GA_POP = 50
_DE_POP = 50
_ISL_N = 4
_ISL_POP = 25
_OPTIMIZERS = ['GA', 'CMA-ES', 'DE', 'Island']


def _cmaes_lambda(dim):
    return 4 + int(3 * math.log(dim))


def _run_classic_one(opt_name, problem, budget, seed):
    genes = _make_genes(problem)
    fn = _wrap_fn(problem)
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        if opt_name == 'GA':
            gens = max(1, budget // _GA_POP)
            opt = GeneticAlgorithm(
                gene_builder=genes, fitness_function=fn,
                population_size=_GA_POP, generations=gens,
                mutation_rate=0.15, crossover_rate=0.7, elitism=2,
                mode='minimize', seed=seed,
                selection=TournamentSelection(k=3),
                crossover=ArithmeticCrossover(),
            )
            _, score, hist = opt.run()
            evals = len(hist) * _GA_POP

        elif opt_name == 'CMA-ES':
            lam = _cmaes_lambda(problem.dim)
            gens = max(1, budget // lam)
            opt = CMAESOptimizer(
                gene_builder=genes, fitness_function=fn,
                sigma0=0.3, generations=gens,
                mode='minimize', seed=seed,
            )
            _, score, hist = opt.run()
            evals = len(hist) * lam

        elif opt_name == 'DE':
            gens = max(1, (budget // _DE_POP) - 1)
            opt = DEOptimizer(
                gene_builder=genes, fitness_function=fn,
                population_size=_DE_POP, generations=gens,
                mode='minimize', seed=seed,
            )
            _, score, hist = opt.run()
            evals = (len(hist) + 1) * _DE_POP

        elif opt_name == 'Island':
            gens = max(1, budget // (_ISL_N * _ISL_POP))
            opt = IslandModel(
                gene_builder=genes, fitness_function=fn,
                n_islands=_ISL_N, island_population=_ISL_POP,
                generations=gens, migration_interval=5, migration_size=2,
                mutation_rate=0.15, crossover_rate=0.7, elitism=2,
                mode='minimize', seed=seed,
                selection=TournamentSelection(k=3),
                crossover=ArithmeticCrossover(),
                topology='ring',
            )
            _, score, hist = opt.run()
            evals = len(hist) * _ISL_N * _ISL_POP

    elapsed = time.perf_counter() - t0
    opt_val = problem.known_optimum or 0.0
    tol = problem.tolerance
    converged = score <= opt_val + tol
    close = not converged and score <= opt_val + tol * 10
    return Result(opt_name, problem.name, 'classic', score, evals, converged, close, elapsed)


# ---------------------------------------------------------------------------
# Engineering (constrained) runner
# ---------------------------------------------------------------------------

def _run_engineering_one(problem, budget, seed):
    genes = _make_genes(problem)
    fn = _wrap_fn(problem)
    constraints = _wrap_constraints(problem)
    pop = 100
    gens = max(1, budget // pop)
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        ga = GeneticAlgorithm(
            gene_builder=genes, fitness_function=fn,
            population_size=pop, generations=gens,
            mutation_rate=0.2, crossover_rate=0.7, elitism=2,
            mode='minimize', seed=seed,
            selection=TournamentSelection(k=3),
            crossover=ArithmeticCrossover(),
            constraints=constraints,
        )
        best, score, hist = ga.run()

    elapsed = time.perf_counter() - t0

    # Check if best solution is feasible
    names = [f'x{i}' for i in range(problem.dim)]
    x_vals = [best[n] for n in names]
    feasible = all(problem.constraints_fn(x_vals))

    opt_val = problem.known_optimum or 0.0
    tol = problem.tolerance
    converged = feasible and score <= opt_val + tol
    close = not converged and feasible and score <= opt_val + tol * 5
    return Result(
        'GA (constrained)', problem.name, 'engineering',
        score, len(hist) * pop, converged, close, elapsed,
        extra={'feasible': feasible},
    )


# ---------------------------------------------------------------------------
# Multi-objective runner
# ---------------------------------------------------------------------------

def _run_mo_one(problem, budget, seed, algorithm='nsga2'):
    genes = _make_genes(problem)
    fn = _wrap_fn(problem)
    pop = 100
    gens = max(1, budget // pop)
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        mo = MultiObjectiveGA(
            gene_builder=genes, fitness_function=fn,
            n_objectives=problem.n_objectives,
            population_size=pop, generations=gens,
            seed=seed, algorithm=algorithm,
        )
        pareto, hist = mo.run()

    elapsed = time.perf_counter() - t0
    return Result(
        f'NSGA-{"III" if algorithm == "nsga3" else "II"}',
        problem.name, 'multi_objective',
        0.0, len(hist) * pop, False, False, elapsed,
        extra={
            'pareto_size': len(pareto),
            'f1_range': (
                min(p['scores'][0] for p in pareto),
                max(p['scores'][0] for p in pareto),
            ) if pareto else None,
        },
    )


# ---------------------------------------------------------------------------
# QD runner
# ---------------------------------------------------------------------------

def _run_qd_one(problem, budget, seed):
    genes = _make_genes(problem)
    fn = _wrap_fn(problem)
    behavior = _make_behavior_fn(problem)
    init_pop = 100
    gens = max(1, (budget - init_pop) // 1)  # batch_size=1
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        me = MAPElites(
            gene_builder=genes, fitness_function=fn,
            behavior_fn=behavior, grid_shape=problem.grid_shape,
            initial_population=init_pop, generations=gens,
            mode='minimize', seed=seed,
        )
        archive, hist = me.run()

    elapsed = time.perf_counter() - t0
    total_cells = 1
    for s in problem.grid_shape:
        total_cells *= s
    coverage = len(archive) / total_cells

    best_score = min(e['score'] for e in archive.values()) if archive else None
    return Result(
        'MAP-Elites', problem.name, 'qd',
        best_score or 0.0, init_pop + gens, False, False, elapsed,
        extra={'coverage': round(coverage, 3), 'archive_size': len(archive)},
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _status_str(r):
    if r.converged:
        return 'SOLVED'
    if r.close:
        return 'close'
    return '-'


def _print_classic(problems, results, budget):
    print(f"\n{'=' * 82}")
    print("Classic Single-Objective Benchmarks".center(82))
    print(f"{'Eval budget: ' + str(budget) + ' per optimizer per problem':^82}")
    print('=' * 82)

    for p in problems:
        pr = [r for r in results if r.problem == p.name and r.category == 'classic']
        desc = f"  ({p.description})" if p.description else ""
        print(f"\n{p.name}{desc}")
        print(f"  {'Optimizer':<18} {'Best Score':>14} {'Evals':>8} {'Status':>8} {'Time':>8}")
        print(f"  {'-'*18} {'-'*14} {'-'*8} {'-'*8} {'-'*8}")
        for r in pr:
            print(
                f"  {r.optimizer:<18} {r.best_score:>14.6f} {r.evals:>8d} "
                f"{_status_str(r):>8} {r.elapsed:>7.2f}s"
            )


def _print_engineering(problems, results):
    print(f"\n{'=' * 82}")
    print("Constrained Engineering Benchmarks".center(82))
    print('=' * 82)

    for p in problems:
        pr = [r for r in results if r.problem == p.name and r.category == 'engineering']
        desc = f"  ({p.description})" if p.description else ""
        print(f"\n{p.name}{desc}")
        print(f"  Known optimum: {p.known_optimum:.4f}, tolerance: {p.tolerance}")
        print(f"  {'Optimizer':<22} {'Cost':>12} {'Feasible':>10} {'Status':>8} {'Time':>8}")
        print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        for r in pr:
            feas = 'yes' if r.extra.get('feasible') else 'no'
            print(
                f"  {r.optimizer:<22} {r.best_score:>12.4f} {feas:>10} "
                f"{_status_str(r):>8} {r.elapsed:>7.2f}s"
            )


def _print_mo(problems, results):
    print(f"\n{'=' * 82}")
    print("Multi-Objective Benchmarks".center(82))
    print('=' * 82)

    for p in problems:
        pr = [r for r in results if r.problem == p.name and r.category == 'multi_objective']
        desc = f"  ({p.description})" if p.description else ""
        print(f"\n{p.name}{desc}")
        print(f"  {'Algorithm':<12} {'Pareto Size':>12} {'f1 Range':>20} {'Time':>8}")
        print(f"  {'-'*12} {'-'*12} {'-'*20} {'-'*8}")
        for r in pr:
            ps = r.extra.get('pareto_size', 0)
            f1r = r.extra.get('f1_range')
            f1_str = f"[{f1r[0]:.4f}, {f1r[1]:.4f}]" if f1r else "n/a"
            print(f"  {r.optimizer:<12} {ps:>12d} {f1_str:>20} {r.elapsed:>7.2f}s")


def _print_qd(problems, results):
    print(f"\n{'=' * 82}")
    print("Quality-Diversity Benchmarks".center(82))
    print('=' * 82)

    for p in problems:
        pr = [r for r in results if r.problem == p.name and r.category == 'qd']
        total = 1
        for s in p.grid_shape:
            total *= s
        desc = f"  (grid {p.grid_shape}, {total} cells)"
        print(f"\n{p.name}{desc}")
        print(f"  {'Optimizer':<14} {'Best Score':>12} {'Coverage':>10} {'Cells Filled':>14} {'Time':>8}")
        print(f"  {'-'*14} {'-'*12} {'-'*10} {'-'*14} {'-'*8}")
        for r in pr:
            cov = r.extra.get('coverage', 0)
            arch = r.extra.get('archive_size', 0)
            print(
                f"  {r.optimizer:<14} {r.best_score:>12.4f} {cov:>9.1%} "
                f"{arch:>14d} {r.elapsed:>7.2f}s"
            )


def _print_summary(results):
    classic = [r for r in results if r.category == 'classic']
    if not classic:
        return

    print(f"\n{'=' * 82}")
    print("Summary — Classic".center(82))
    print('=' * 82)

    problems = sorted(set(r.problem for r in classic),
                       key=lambda n: next(i for i, r in enumerate(classic) if r.problem == n))
    optimizers = sorted(set(r.optimizer for r in classic),
                         key=lambda n: next(i for i, r in enumerate(classic) if r.optimizer == n))

    short = [n.replace(" ", "") for n in problems]
    col_w = max(len(s) for s in short) + 1

    header = f"  {'':18}" + "".join(f"{s:>{col_w}}" for s in short)
    print(header)
    print(f"  {'-' * 18}" + "-" * (col_w * len(short)))

    for opt in optimizers:
        row = f"  {opt:<18}"
        for pname in problems:
            r = next((x for x in classic if x.optimizer == opt and x.problem == pname), None)
            if r is None:
                cell = ""
            elif r.converged:
                cell = "yes"
            elif r.close:
                cell = "~"
            else:
                cell = "no"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  yes = converged, ~ = close, no = did not converge")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_suite(
    categories: Optional[list] = None,
    eval_budget: int = 5000,
    seed: int = 42,
    save: bool = True,
    output_dir: Optional[str] = None,
) -> list:
    """Run the benchmark suite.

    Args:
        categories: List of 'classic', 'engineering', 'multi_objective', 'qd'.
                    None = all categories.
        eval_budget: Total fitness evaluations per optimizer per problem.
        seed: Random seed for reproducibility.
        save: Save results to JSON.
        output_dir: Directory for JSON output. Default: current directory.

    Returns:
        List of Result dataclasses.
    """
    cats = categories or list(ALL.keys())
    results = []
    total_t0 = time.perf_counter()

    if 'classic' in cats:
        problems = ALL['classic']
        for p in problems:
            for opt_name in _OPTIMIZERS:
                r = _run_classic_one(opt_name, p, eval_budget, seed)
                results.append(r)
        _print_classic(problems, results, eval_budget)
        _print_summary(results)

    if 'engineering' in cats:
        problems = ALL['engineering']
        eng_budget = max(eval_budget, 50000)
        for p in problems:
            r = _run_engineering_one(p, eng_budget, seed)
            results.append(r)
        _print_engineering(problems, results)

    if 'multi_objective' in cats:
        problems = ALL['multi_objective']
        for p in problems:
            algo = 'nsga3' if p.n_objectives >= 3 else 'nsga2'
            r = _run_mo_one(p, eval_budget, seed, algorithm=algo)
            results.append(r)
        _print_mo(problems, results)

    if 'qd' in cats:
        problems = ALL['qd']
        for p in problems:
            r = _run_qd_one(p, eval_budget, seed)
            results.append(r)
        _print_qd(problems, results)

    total_elapsed = time.perf_counter() - total_t0
    print(f"\nTotal time: {total_elapsed:.1f}s")

    if save:
        log = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'eval_budget': eval_budget,
            'seed': seed,
            'total_elapsed': round(total_elapsed, 2),
            'categories': cats,
            'results': [asdict(r) for r in results],
        }
        out_dir = output_dir or '.'
        out_path = os.path.join(out_dir, 'benchmark_suite_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        print(f"Results saved to {out_path}")

    return results
