"""Standard single-objective benchmark functions.

All functions take a list of floats and return a float to minimize.
Global minimum is 0.0 unless noted otherwise.

Usage::

    from evogine.benchmarks.functions import sphere, rastrigin
    sphere([0.0, 0.0])      # -> 0.0
    rastrigin([0.0, 0.0])   # -> 0.0
"""

import math


def sphere(x):
    """sum(x_i^2). Min 0 at origin. Unimodal, separable."""
    return sum(xi ** 2 for xi in x)


def rosenbrock(x):
    """sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2). Min 0 at (1,...,1).
    Narrow curved valley, non-separable."""
    return sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def rastrigin(x):
    """10n + sum(x_i^2 - 10*cos(2*pi*x_i)). Min 0 at origin.
    Highly multimodal — ~10^n local minima."""
    n = len(x)
    return 10 * n + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def ackley(x):
    """-20*exp(-0.2*sqrt(mean(x_i^2))) - exp(mean(cos(2*pi*x_i))) + 20 + e.
    Min 0 at origin. Multimodal with smooth global funnel."""
    n = len(x)
    sum_sq = sum(xi ** 2 for xi in x)
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x)
    return (
        -20 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20 + math.e
    )


def schwefel(x):
    """418.9829*n - sum(x_i*sin(sqrt(|x_i|))). Min 0 at (420.9687,...).
    Deceptive — global minimum is geometrically far from local structure."""
    n = len(x)
    return 418.9829 * n - sum(
        xi * math.sin(math.sqrt(abs(xi))) for xi in x
    )


def griewank(x):
    """sum(x_i^2/4000) - prod(cos(x_i/sqrt(i+1))) + 1. Min 0 at origin.
    Variable interactions via product term."""
    sum_sq = sum(xi ** 2 for xi in x)
    prod_cos = 1.0
    for i, xi in enumerate(x):
        prod_cos *= math.cos(xi / math.sqrt(i + 1))
    return sum_sq / 4000 - prod_cos + 1


def levy(x):
    """Levy function. Min 0 at (1,...,1). Multimodal, non-separable."""
    w = [1 + (xi - 1) / 4 for xi in x]
    term1 = math.sin(math.pi * w[0]) ** 2
    term2 = sum(
        (wi - 1) ** 2 * (1 + 10 * math.sin(math.pi * wi + 1) ** 2)
        for wi in w[:-1]
    )
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def michalewicz(x, m=10):
    """Michalewicz function. Min at dimension-dependent point.
    d=2: -1.8013, d=5: -4.687658, d=10: -9.66015.
    Steep ridges — 'needle in haystack' with large m."""
    return -sum(
        math.sin(xi) * math.sin((i + 1) * xi ** 2 / math.pi) ** (2 * m)
        for i, xi in enumerate(x)
    )


def styblinski_tang(x):
    """sum(x_i^4 - 16*x_i^2 + 5*x_i) / 2. Min -39.16617*d at (-2.9035,...).
    Simple multimodal."""
    return sum(xi ** 4 - 16 * xi ** 2 + 5 * xi for xi in x) / 2


def zakharov(x):
    """sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4.
    Min 0 at origin. Unimodal, non-separable, plate-shaped."""
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2 ** 2 + sum2 ** 4


def dixon_price(x):
    """(x_1-1)^2 + sum(i*(2*x_i^2 - x_{i-1})^2). Min 0.
    Valley-shaped, non-separable."""
    term1 = (x[0] - 1) ** 2
    term2 = sum(
        (i + 2) * (2 * x[i + 1] ** 2 - x[i]) ** 2
        for i in range(len(x) - 1)
    )
    return term1 + term2
