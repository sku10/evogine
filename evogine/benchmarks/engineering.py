"""Constrained engineering design benchmarks.

Real-world structural optimization problems with nonlinear constraints.
Each problem has a cost function and constraint functions.
Constraints return a list of bools (True = feasible).

Usage::

    from evogine.benchmarks.engineering import welded_beam_cost, welded_beam_constraints
    x = [0.2057, 3.4705, 9.0366, 0.2057]
    cost = welded_beam_cost(x)
    feasible = all(welded_beam_constraints(x))
"""

import math


# ---------------------------------------------------------------------------
# Welded Beam Design (Ragsdell & Phillips, 1976)
# ---------------------------------------------------------------------------
# Variables: h (weld size), l (weld length), t (beam height), b (beam width)
# Bounds: h∈[0.1,2], l∈[0.1,10], t∈[0.1,10], b∈[0.1,2]
# Best known: f* ≈ 1.7249 at (0.2057, 3.4705, 9.0366, 0.2057)

_WB_P = 6000.0          # applied force (lb)
_WB_L = 14.0            # beam length (in)
_WB_E = 30e6            # Young's modulus (psi)
_WB_G = 12e6            # shear modulus (psi)
_WB_TAU_MAX = 13600.0   # max shear stress (psi)
_WB_SIGMA_MAX = 30000.0 # max bending stress (psi)
_WB_DELTA_MAX = 0.25    # max deflection (in)

WB_BOUNDS = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]
WB_OPTIMUM = 1.7249
WB_TOLERANCE = 0.5


def _wb_tau(h, l, t, b):
    tau_prime = _WB_P / (math.sqrt(2) * h * l)
    M = _WB_P * (_WB_L + l / 2)
    R = math.sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
    J = 2 * math.sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2)
    tau_double_prime = M * R / J
    return math.sqrt(
        tau_prime ** 2
        + 2 * tau_prime * tau_double_prime * (l / (2 * R))
        + tau_double_prime ** 2
    )


def welded_beam_cost(x):
    """Fabrication cost. x = [h, l, t, b]."""
    h, l, t, b = x
    return 1.10471 * h ** 2 * l + 0.04811 * t * b * (14 + l)


def welded_beam_constraints(x):
    """7 constraints. Returns list of 7 bools (True = feasible)."""
    h, l, t, b = x
    tau = _wb_tau(h, l, t, b)
    sigma = 6 * _WB_P * _WB_L / (b * t ** 2)
    delta = 4 * _WB_P * _WB_L ** 3 / (_WB_E * t ** 3 * b)
    pc = (
        4.013 * _WB_E * math.sqrt(t ** 2 * b ** 6 / 36) / _WB_L ** 2
        * (1 - t / (2 * _WB_L) * math.sqrt(_WB_E / (4 * _WB_G)))
    )
    return [
        tau <= _WB_TAU_MAX,
        sigma <= _WB_SIGMA_MAX,
        h <= b,
        0.10471 * h ** 2 + 0.04811 * t * b * (14 + l) <= 5,
        h >= 0.125,
        delta <= _WB_DELTA_MAX,
        _WB_P <= pc,
    ]


# ---------------------------------------------------------------------------
# Pressure Vessel Design (Sandgren, 1990)
# ---------------------------------------------------------------------------
# Variables: Ts (shell thick), Th (head thick), R (radius), L (length)
# Bounds: Ts∈[0.0625,6.1875], Th∈[0.0625,6.1875], R∈[10,200], L∈[10,200]
# Best known: f* ≈ 6059.71

PV_BOUNDS = [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 200.0)]
PV_OPTIMUM = 6059.71
PV_TOLERANCE = 500.0


def pressure_vessel_cost(x):
    """Total cost. x = [Ts, Th, R, L]."""
    ts, th, r, l = x
    return (
        0.6224 * ts * r * l
        + 1.7781 * th * r ** 2
        + 3.1661 * ts ** 2 * l
        + 19.84 * ts ** 2 * r
    )


def pressure_vessel_constraints(x):
    """4 constraints. Returns list of 4 bools (True = feasible)."""
    ts, th, r, l = x
    return [
        -ts + 0.0193 * r <= 0,
        -th + 0.00954 * r <= 0,
        -math.pi * r ** 2 * l - (4 / 3) * math.pi * r ** 3 + 1296000 <= 0,
        l - 240 <= 0,
    ]


# ---------------------------------------------------------------------------
# Tension/Compression Spring Design (Belegundu, 1982)
# ---------------------------------------------------------------------------
# Variables: d (wire diameter), D (coil diameter), N (active coils)
# Bounds: d∈[0.05,2.0], D∈[0.25,1.3], N∈[2,15]
# Best known: f* ≈ 0.012665

SP_BOUNDS = [(0.05, 0.2), (0.25, 1.3), (2.0, 15.0)]
SP_OPTIMUM = 0.012665
SP_TOLERANCE = 0.005


def spring_cost(x):
    """Weight of spring. x = [d, D, N]."""
    d, D, N = x
    return (N + 2) * D * d ** 2


def spring_constraints(x):
    """4 constraints. Returns list of 4 bools (True = feasible).
    Standard formulation: g_i <= 0 means feasible."""
    d, D, N = x
    return [
        1 - D ** 3 * N / (71785 * d ** 4) <= 0,
        (4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4)) + 1 / (5108 * d ** 2) - 1 <= 0,
        1 - 140.45 * d / (D ** 2 * N) <= 0,
        (d + D) / 1.5 - 1 <= 0,
    ]
