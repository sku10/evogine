"""Problem registry — defines benchmark problems and organizes them by category.

Each Problem wraps a raw math function with metadata (bounds, known optimum,
constraints, number of objectives). The runner uses this to configure evogine
optimizers automatically.

To add a new problem: define the function in the appropriate module, then
append a Problem to the relevant list (CLASSIC, ENGINEERING, etc.).
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

from . import functions as F
from . import engineering as E
from . import multi_objective as MO


@dataclass
class Problem:
    """A benchmark problem definition."""
    name: str
    category: str
    dim: int
    bounds: list  # [(lo, hi), ...] per dimension
    fn: Callable  # list[float] -> float or list[float]
    known_optimum: Optional[float] = None
    tolerance: float = 0.01
    n_objectives: int = 1
    constraints_fn: Optional[Callable] = None
    n_constraints: int = 0
    behavior_dim: int = 0
    grid_shape: Optional[tuple] = None
    description: str = ""


def _uniform(dim, lo, hi):
    return [(lo, hi)] * dim


# ---------------------------------------------------------------------------
# Classic single-objective
# ---------------------------------------------------------------------------

CLASSIC = [
    Problem("Sphere 2D", "classic", 2, _uniform(2, -5.12, 5.12),
            F.sphere, 0.0, 0.01,
            description="Unimodal, separable — baseline"),
    Problem("Sphere 5D", "classic", 5, _uniform(5, -5.12, 5.12),
            F.sphere, 0.0, 0.01),
    Problem("Sphere 10D", "classic", 10, _uniform(10, -5.12, 5.12),
            F.sphere, 0.0, 0.1),
    Problem("Rosenbrock 5D", "classic", 5, _uniform(5, -5.0, 10.0),
            F.rosenbrock, 0.0, 1.0,
            description="Narrow curved valley, non-separable"),
    Problem("Rastrigin 5D", "classic", 5, _uniform(5, -5.12, 5.12),
            F.rastrigin, 0.0, 1.0,
            description="Highly multimodal — ~10^5 local minima"),
    Problem("Ackley 5D", "classic", 5, _uniform(5, -5.0, 5.0),
            F.ackley, 0.0, 0.5,
            description="Multimodal with smooth global funnel"),
    Problem("Schwefel 5D", "classic", 5, _uniform(5, -500.0, 500.0),
            F.schwefel, 0.0, 50.0,
            description="Deceptive — global min far from local structure"),
    Problem("Griewank 5D", "classic", 5, _uniform(5, -600.0, 600.0),
            F.griewank, 0.0, 0.1,
            description="Variable interactions via product term"),
    Problem("Levy 5D", "classic", 5, _uniform(5, -10.0, 10.0),
            F.levy, 0.0, 0.1,
            description="Multimodal, non-separable"),
    Problem("Michalewicz 5D", "classic", 5, _uniform(5, 0.0, math.pi),
            F.michalewicz, -4.687658, 0.5,
            description="Steep ridges — needle in haystack"),
    Problem("Styblinski-Tang 5D", "classic", 5, _uniform(5, -5.0, 5.0),
            F.styblinski_tang, -39.16617 * 5, 5.0,
            description="Simple multimodal"),
    Problem("Zakharov 5D", "classic", 5, _uniform(5, -5.0, 10.0),
            F.zakharov, 0.0, 0.1,
            description="Unimodal, plate-shaped"),
    Problem("Dixon-Price 5D", "classic", 5, _uniform(5, -10.0, 10.0),
            F.dixon_price, 0.0, 1.0,
            description="Valley-shaped, non-separable"),
]


# ---------------------------------------------------------------------------
# Constrained engineering
# ---------------------------------------------------------------------------

ENGINEERING = [
    Problem("Welded Beam", "engineering", 4, E.WB_BOUNDS,
            E.welded_beam_cost, E.WB_OPTIMUM, E.WB_TOLERANCE,
            constraints_fn=E.welded_beam_constraints, n_constraints=7,
            description="4 vars, 7 constraints — structural optimization"),
    Problem("Pressure Vessel", "engineering", 4, E.PV_BOUNDS,
            E.pressure_vessel_cost, E.PV_OPTIMUM, E.PV_TOLERANCE,
            constraints_fn=E.pressure_vessel_constraints, n_constraints=4,
            description="4 vars, 4 constraints — minimum cost design"),
    Problem("Spring Design", "engineering", 3, E.SP_BOUNDS,
            E.spring_cost, E.SP_OPTIMUM, E.SP_TOLERANCE,
            constraints_fn=E.spring_constraints, n_constraints=4,
            description="3 vars, 4 constraints — minimum weight"),
]


# ---------------------------------------------------------------------------
# Multi-objective
# ---------------------------------------------------------------------------

MULTI_OBJECTIVE = [
    Problem("ZDT1", "multi_objective", 30, _uniform(30, 0.0, 1.0),
            MO.zdt1, n_objectives=2,
            description="Convex Pareto front"),
    Problem("ZDT2", "multi_objective", 30, _uniform(30, 0.0, 1.0),
            MO.zdt2, n_objectives=2,
            description="Concave Pareto front"),
    Problem("ZDT3", "multi_objective", 30, _uniform(30, 0.0, 1.0),
            MO.zdt3, n_objectives=2,
            description="Disconnected Pareto front — 5 segments"),
    Problem("ZDT6", "multi_objective", 10, _uniform(10, 0.0, 1.0),
            MO.zdt6, n_objectives=2,
            description="Non-uniform density near front"),
    Problem("DTLZ1 (3-obj)", "multi_objective", 7, _uniform(7, 0.0, 1.0),
            lambda x: MO.dtlz1(x, 3), n_objectives=3,
            description="Linear front, many local fronts"),
    Problem("DTLZ2 (3-obj)", "multi_objective", 12, _uniform(12, 0.0, 1.0),
            lambda x: MO.dtlz2(x, 3), n_objectives=3,
            description="Spherical Pareto front"),
]


# ---------------------------------------------------------------------------
# Quality-diversity (MAP-Elites)
# ---------------------------------------------------------------------------

QD = [
    Problem("Sphere QD", "qd", 5, _uniform(5, -5.12, 5.12),
            F.sphere, behavior_dim=2, grid_shape=(20, 20),
            description="QD on sphere — explore diverse low-cost regions"),
    Problem("Rastrigin QD", "qd", 5, _uniform(5, -5.12, 5.12),
            F.rastrigin, behavior_dim=2, grid_shape=(20, 20),
            description="QD on multimodal landscape"),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL = {
    'classic': CLASSIC,
    'engineering': ENGINEERING,
    'multi_objective': MULTI_OBJECTIVE,
    'qd': QD,
}
