"""Multi-objective benchmark functions.

ZDT (2-objective) and DTLZ (scalable) test suites.
All functions take a list of floats and return a list of floats (objectives to minimize).

Usage::

    from evogine.benchmarks.multi_objective import zdt1, dtlz2
    zdt1([0.5] + [0.0] * 29)   # -> [0.5, 0.293...]
    dtlz2([0.5] * 12, n_obj=3) # -> [f1, f2, f3]
"""

import math


# ---------------------------------------------------------------------------
# ZDT (Zitzler-Deb-Thiele) — 2 objectives, vars in [0, 1]
# ---------------------------------------------------------------------------

def zdt1(x):
    """Convex Pareto front. Standard: n=30.
    Front: f2 = 1 - sqrt(f1), f1 in [0,1]."""
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * sum(x[1:])
    f2 = g * (1 - math.sqrt(f1 / g))
    return [f1, f2]


def zdt2(x):
    """Concave (non-convex) Pareto front. Standard: n=30.
    Front: f2 = 1 - (f1)^2, f1 in [0,1]."""
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * sum(x[1:])
    f2 = g * (1 - (f1 / g) ** 2)
    return [f1, f2]


def zdt3(x):
    """Disconnected Pareto front (5 segments). Standard: n=30.
    Tests diversity maintenance across gaps."""
    n = len(x)
    f1 = x[0]
    g = 1 + 9 / (n - 1) * sum(x[1:])
    f2 = g * (1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1))
    return [f1, f2]


def zdt6(x):
    """Non-convex front with non-uniform density. Standard: n=10.
    Solutions are sparse near the Pareto front — tests convergence pressure."""
    n = len(x)
    f1 = 1 - math.exp(-4 * x[0]) * math.sin(6 * math.pi * x[0]) ** 6
    g = 1 + 9 * (sum(x[1:]) / (n - 1)) ** 0.25
    f2 = g * (1 - (f1 / g) ** 2)
    return [f1, f2]


# ---------------------------------------------------------------------------
# DTLZ (Deb-Thiele-Laumanns-Zitzler) — scalable objectives
# ---------------------------------------------------------------------------

def dtlz1(x, n_obj=3):
    """Linear Pareto front (sum f_i = 0.5).
    Has (3^k - 1) local fronts — tests convergence.
    Standard: n = n_obj - 1 + k, k=5."""
    k = len(x) - n_obj + 1
    x_m = x[n_obj - 1:]
    g = 100 * (
        k + sum((xi - 0.5) ** 2 - math.cos(20 * math.pi * (xi - 0.5)) for xi in x_m)
    )
    f = []
    for i in range(n_obj):
        fi = 0.5 * (1 + g)
        for j in range(n_obj - 1 - i):
            fi *= x[j]
        if i > 0:
            fi *= (1 - x[n_obj - 1 - i])
        f.append(fi)
    return f


def dtlz2(x, n_obj=3):
    """Spherical Pareto front (sum f_i^2 = 1).
    Tests scalability to many objectives.
    Standard: n = n_obj - 1 + k, k=10."""
    k = len(x) - n_obj + 1
    x_m = x[n_obj - 1:]
    g = sum((xi - 0.5) ** 2 for xi in x_m)
    f = []
    for i in range(n_obj):
        fi = 1 + g
        for j in range(n_obj - 1 - i):
            fi *= math.cos(x[j] * math.pi / 2)
        if i > 0:
            fi *= math.sin(x[n_obj - 1 - i] * math.pi / 2)
        f.append(fi)
    return f
