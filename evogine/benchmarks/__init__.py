"""evogine benchmark suite — standard test functions for evolutionary optimizers.

Quick start::

    from evogine.benchmarks import run_suite
    run_suite()                              # all categories
    run_suite(categories=['classic'])         # just classic functions
    run_suite(categories=['engineering'])     # constrained engineering

Individual functions::

    from evogine.benchmarks.functions import sphere, rastrigin
    from evogine.benchmarks.engineering import welded_beam_cost
    from evogine.benchmarks.multi_objective import zdt1, dtlz2

Problem registry::

    from evogine.benchmarks.problems import CLASSIC, ENGINEERING, ALL
"""

from .runner import run_suite
from .problems import Problem, CLASSIC, ENGINEERING, MULTI_OBJECTIVE, QD, ALL

__all__ = [
    'run_suite',
    'Problem', 'CLASSIC', 'ENGINEERING', 'MULTI_OBJECTIVE', 'QD', 'ALL',
]
