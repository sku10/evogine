#!/usr/bin/env python3
"""
evogine Benchmark Suite
=======================

Runs standard optimization benchmark functions against evogine's optimizers
and prints a comparison table showing where each optimizer excels.

Usage:
    python examples/benchmarks.py

Each optimizer gets the same evaluation budget per problem (5000 fitness
evaluations by default). This makes the comparison fair — a fast optimizer
that converges in 500 evals is genuinely better than one that needs all 5000.

Benchmark functions (all minimization, global minimum = 0):

  Sphere      Unimodal, separable. The easiest test. Any optimizer should solve it.
  Rosenbrock  Unimodal but non-separable. Narrow curved valley toward (1,1,...,1).
              Tests ability to follow correlated variables.
  Rastrigin   Highly multimodal. ~10^n local minima. Tests escape from local optima.
  Ackley      Multimodal with a smooth global funnel. Tests local vs global balance.

Expected winners:
  CMA-ES      Sphere (all dims), Rosenbrock, Ackley — covariance adaptation
              follows valleys and funnels efficiently.
  IslandModel Rastrigin — independent populations explore different basins.
  DE          Strong all-rounder on continuous problems.
  GA          Competitive but typically needs more evaluations than CMA-ES/DE.
"""

import contextlib
import io
import json
import math
import sys
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evogine import (
    GeneticAlgorithm, CMAESOptimizer, DEOptimizer, IslandModel,
    GeneBuilder, FloatRange, TournamentSelection, ArithmeticCrossover,
)


# ---------------------------------------------------------------------------
# Benchmark function definitions
# ---------------------------------------------------------------------------

def sphere(ind: dict) -> float:
    """f(x) = sum(x_i^2). Minimum 0 at origin. Unimodal, separable."""
    return sum(v ** 2 for v in ind.values())


def rosenbrock(ind: dict) -> float:
    """f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2). Minimum 0 at (1,...,1).
    Non-separable narrow valley — the classic hard unimodal problem."""
    vals = list(ind.values())
    return sum(
        100 * (vals[i + 1] - vals[i] ** 2) ** 2 + (1 - vals[i]) ** 2
        for i in range(len(vals) - 1)
    )


def rastrigin(ind: dict) -> float:
    """f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i)). Minimum 0 at origin.
    Highly multimodal — many evenly-spaced local minima."""
    vals = list(ind.values())
    n = len(vals)
    return 10 * n + sum(v ** 2 - 10 * math.cos(2 * math.pi * v) for v in vals)


def ackley(ind: dict) -> float:
    """f(x) = -20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e.
    Minimum 0 at origin. Multimodal with a smooth global funnel."""
    vals = list(ind.values())
    n = len(vals)
    sum_sq = sum(v ** 2 for v in vals)
    sum_cos = sum(math.cos(2 * math.pi * v) for v in vals)
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20 + math.e
    )


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

@dataclass
class Benchmark:
    name: str
    dim: int
    fn: Callable[[dict], float]
    lo: float
    hi: float
    known_min: float
    tolerance: float  # "converged" if best <= known_min + tolerance


BENCHMARKS = [
    Benchmark("Sphere 2D",      2,  sphere,     -5.12,  5.12, 0.0, 0.01),
    Benchmark("Sphere 5D",      5,  sphere,     -5.12,  5.12, 0.0, 0.01),
    Benchmark("Sphere 10D",    10,  sphere,     -5.12,  5.12, 0.0, 0.1),
    Benchmark("Rosenbrock 5D",  5,  rosenbrock, -5.0,  10.0,  0.0, 1.0),
    Benchmark("Rastrigin 5D",   5,  rastrigin,  -5.12,  5.12, 0.0, 1.0),
    Benchmark("Ackley 5D",      5,  ackley,     -5.0,   5.0,  0.0, 0.5),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_genes(dim: int, lo: float, hi: float) -> GeneBuilder:
    gb = GeneBuilder()
    for i in range(dim):
        gb.add(f"x{i}", FloatRange(lo, hi))
    return gb


@dataclass
class Result:
    optimizer: str
    benchmark: str
    best_score: float
    evals: int
    converged: bool
    close: bool  # within 10x tolerance but not converged
    elapsed: float


# ---------------------------------------------------------------------------
# Budget → generations conversion
# ---------------------------------------------------------------------------

GA_POP = 50
DE_POP = 50
ISLAND_N = 4
ISLAND_POP = 25
EVAL_BUDGET = 5000


def ga_gens(budget: int) -> int:
    return max(1, budget // GA_POP)


def cmaes_lambda(dim: int) -> int:
    return 4 + int(3 * math.log(dim))


def cmaes_gens(budget: int, dim: int) -> int:
    return max(1, budget // cmaes_lambda(dim))


def de_gens(budget: int) -> int:
    return max(1, (budget // DE_POP) - 1)


def island_gens(budget: int) -> int:
    return max(1, budget // (ISLAND_N * ISLAND_POP))


# ---------------------------------------------------------------------------
# Run a single optimizer on a single benchmark
# ---------------------------------------------------------------------------

def run_one(name: str, bm: Benchmark, seed: int = 42) -> Result:
    genes = make_genes(bm.dim, bm.lo, bm.hi)
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        if name == "GA":
            opt = GeneticAlgorithm(
                gene_builder=genes,
                fitness_function=bm.fn,
                population_size=GA_POP,
                generations=ga_gens(EVAL_BUDGET),
                mutation_rate=0.15,
                crossover_rate=0.7,
                elitism=2,
                mode='minimize',
                seed=seed,
                selection=TournamentSelection(k=3),
                crossover=ArithmeticCrossover(),
            )
            best, score, history = opt.run()
            evals = len(history) * GA_POP

        elif name == "CMA-ES":
            gens = cmaes_gens(EVAL_BUDGET, bm.dim)
            opt = CMAESOptimizer(
                gene_builder=genes,
                fitness_function=bm.fn,
                sigma0=0.3,
                generations=gens,
                mode='minimize',
                seed=seed,
            )
            best, score, history = opt.run()
            lam = cmaes_lambda(bm.dim)
            evals = len(history) * lam

        elif name == "DE":
            opt = DEOptimizer(
                gene_builder=genes,
                fitness_function=bm.fn,
                population_size=DE_POP,
                generations=de_gens(EVAL_BUDGET),
                mode='minimize',
                seed=seed,
            )
            best, score, history = opt.run()
            evals = (len(history) + 1) * DE_POP

        elif name == "Island":
            opt = IslandModel(
                gene_builder=genes,
                fitness_function=bm.fn,
                n_islands=ISLAND_N,
                island_population=ISLAND_POP,
                generations=island_gens(EVAL_BUDGET),
                migration_interval=5,
                migration_size=2,
                mutation_rate=0.15,
                crossover_rate=0.7,
                elitism=2,
                mode='minimize',
                seed=seed,
                selection=TournamentSelection(k=3),
                crossover=ArithmeticCrossover(),
                topology='ring',
            )
            best, score, history = opt.run()
            evals = len(history) * ISLAND_N * ISLAND_POP

    elapsed = time.perf_counter() - t0
    converged = score <= bm.known_min + bm.tolerance
    close = not converged and score <= bm.known_min + bm.tolerance * 10

    return Result(name, bm.name, score, evals, converged, close, elapsed)


# ---------------------------------------------------------------------------
# Run all and print
# ---------------------------------------------------------------------------

OPTIMIZERS = ["GA", "CMA-ES", "DE", "Island"]


def run_all() -> list[Result]:
    results = []
    total_t0 = time.perf_counter()

    print("=" * 78)
    print("evogine Benchmark Suite".center(78))
    print(f"Eval budget: {EVAL_BUDGET} per optimizer per problem".center(78))
    print("=" * 78)

    for bm in BENCHMARKS:
        print(f"\n{bm.name}  (min={bm.known_min}, tol={bm.tolerance})")
        print("-" * 78)
        print(f"  {'Optimizer':<18} {'Best Score':>12} {'Evals':>8} {'Status':>10} {'Time':>8}")
        print(f"  {'-'*18} {'-'*12} {'-'*8} {'-'*10} {'-'*8}")

        for opt_name in OPTIMIZERS:
            r = run_one(opt_name, bm)
            results.append(r)

            if r.converged:
                status = "SOLVED"
            elif r.close:
                status = "close"
            else:
                status = "-"

            print(
                f"  {r.optimizer:<18} {r.best_score:>12.6f} {r.evals:>8d} "
                f"{status:>10} {r.elapsed:>7.2f}s"
            )

    total_elapsed = time.perf_counter() - total_t0

    # Summary matrix
    print(f"\n{'=' * 78}")
    print("Summary".center(78))
    print("=" * 78)

    bm_names = [bm.name for bm in BENCHMARKS]
    short_names = [n.replace(" ", "") for n in bm_names]
    col_w = max(len(s) for s in short_names) + 2

    header = f"  {'':18}" + "".join(f"{s:>{col_w}}" for s in short_names)
    print(header)
    print(f"  {'-' * 18}" + "-" * (col_w * len(short_names)))

    for opt_name in OPTIMIZERS:
        row = f"  {opt_name:<18}"
        for bm_name in bm_names:
            r = next(x for x in results if x.optimizer == opt_name and x.benchmark == bm_name)
            if r.converged:
                cell = "yes"
            elif r.close:
                cell = "~"
            else:
                cell = "no"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n  yes = converged within tolerance")
    print(f"    ~ = close (within 10x tolerance)")
    print(f"   no = did not converge")
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Save JSON
    log = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'eval_budget': EVAL_BUDGET,
        'seed': 42,
        'total_elapsed': round(total_elapsed, 2),
        'results': [asdict(r) for r in results],
        'summary': {
            opt: {
                bm.name: next(
                    "yes" if x.converged else ("close" if x.close else "no")
                    for x in results
                    if x.optimizer == opt and x.benchmark == bm.name
                )
                for bm in BENCHMARKS
            }
            for opt in OPTIMIZERS
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    run_all()
