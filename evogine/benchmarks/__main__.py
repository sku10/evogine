"""Run: python -m evogine.benchmarks [classic|engineering|multi_objective|qd]"""

import sys
from .runner import run_suite

categories = sys.argv[1:] or None
run_suite(categories=categories)
