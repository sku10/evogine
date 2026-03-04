import random
from typing import Callable, Optional

from .genes import GeneSpec, FloatRange, IntRange, ChoiceList, GeneBuilder


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

class SelectionStrategy:
    """Base class for parent selection. Subclass and implement select_parents()."""
    def select_parents(
        self, scored: list[tuple[dict, float]]
    ) -> tuple[dict, dict]:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class RouletteSelection(SelectionStrategy):
    """
    Fitness-proportionate (roulette wheel) selection.
    Scores are shifted so the worst individual gets a tiny positive weight.
    Default selection strategy.
    """
    def select_parents(self, scored):
        min_score = min(s for _, s in scored)
        weights = [(s - min_score + 1e-12) for _, s in scored]
        parents = random.choices(scored, weights=weights, k=2)
        return parents[0][0], parents[1][0]

    def describe(self):
        return {'strategy': 'roulette'}


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection. Randomly picks k individuals and returns the best.
    Repeat twice for two parents.

    Args:
        k: Tournament size. Higher k = stronger selection pressure.
           k=2 is gentle, k=5+ is aggressive.
    """
    def __init__(self, k: int = 3):
        self.k = k

    def select_parents(self, scored):
        def tournament():
            competitors = random.sample(scored, min(self.k, len(scored)))
            return max(competitors, key=lambda x: x[1])[0]
        return tournament(), tournament()

    def describe(self):
        return {'strategy': 'tournament', 'k': self.k}


class RankSelection(SelectionStrategy):
    """
    Rank-based selection. Assigns weights by rank (best = N, worst = 1).
    Prevents one superstar individual from dominating when fitness values
    differ wildly in magnitude.
    """
    def select_parents(self, scored):
        n = len(scored)
        weights = [n - i for i in range(n)]
        parents = random.choices(scored, weights=weights, k=2)
        return parents[0][0], parents[1][0]

    def describe(self):
        return {'strategy': 'rank'}


# ---------------------------------------------------------------------------
# Crossover strategies
# ---------------------------------------------------------------------------

class CrossoverStrategy:
    """Base class for crossover. Subclass and implement crossover()."""
    def crossover(
        self, p1: dict, p2: dict, gene_builder: GeneBuilder
    ) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class UniformCrossover(CrossoverStrategy):
    """
    Uniform crossover. Each gene is independently inherited from either
    parent with equal probability (50/50). Default crossover strategy.
    """
    def crossover(self, p1, p2, gene_builder):
        return {
            key: p1[key] if random.random() > 0.5 else p2[key]
            for key in gene_builder.keys()
        }

    def describe(self):
        return {'strategy': 'uniform'}


class ArithmeticCrossover(CrossoverStrategy):
    """
    Arithmetic (blend) crossover for FloatRange genes.
    child = t*p1 + (1-t)*p2 where t is random in [0, 1].
    Non-float genes fall back to uniform selection.
    """
    def crossover(self, p1, p2, gene_builder):
        child = {}
        for key in gene_builder.keys():
            spec = gene_builder.specs[key]
            if isinstance(spec, FloatRange):
                t = random.random()
                child[key] = t * p1[key] + (1 - t) * p2[key]
            else:
                child[key] = p1[key] if random.random() > 0.5 else p2[key]
        return child

    def describe(self):
        return {'strategy': 'arithmetic'}


class SinglePointCrossover(CrossoverStrategy):
    """
    Single-point crossover. Genes before a random split come from p1, after from p2.
    Preserves gene co-dependencies.
    """
    def crossover(self, p1, p2, gene_builder):
        keys = gene_builder.keys()
        split = random.randint(1, len(keys) - 1) if len(keys) > 1 else 1
        child = {}
        for i, key in enumerate(keys):
            child[key] = p1[key] if i < split else p2[key]
        return child

    def describe(self):
        return {'strategy': 'single_point'}


class LLMCrossover(CrossoverStrategy):
    """
    LLM-assisted crossover. Delegates the crossover to a user-supplied function
    that takes two parent dicts and returns a child dict (e.g., via an LLM API).

    The result is validated (all gene keys present) and clamped to gene bounds.
    On failure, falls back to UniformCrossover and increments fallback_count.

    Args:
        llm_fn:           Callable[[dict, dict], dict]. Takes two parent individuals,
                          returns a child individual dict.
        raise_on_failure: If True, re-raise exceptions from llm_fn instead of
                          falling back. Default False.

    Attributes:
        fallback_count:   Number of times the fallback was triggered.
    """

    def __init__(
        self,
        llm_fn: Callable[[dict, dict], dict],
        raise_on_failure: bool = False,
    ):
        self._llm_fn = llm_fn
        self._raise_on_failure = raise_on_failure
        self.fallback_count = 0
        self._fallback = UniformCrossover()

    def crossover(self, p1: dict, p2: dict, gene_builder: GeneBuilder) -> dict:
        try:
            child = self._llm_fn(p1, p2)
            missing = [k for k in gene_builder.keys() if k not in child]
            if missing:
                raise ValueError(f"LLMCrossover result missing keys: {missing}")
            for name, spec in gene_builder.specs.items():
                if isinstance(spec, FloatRange):
                    child[name] = max(spec.low, min(float(child[name]), spec.high))
                elif isinstance(spec, IntRange):
                    child[name] = max(spec.low, min(int(round(child[name])), spec.high))
                elif isinstance(spec, ChoiceList):
                    if child[name] not in spec.options:
                        raise ValueError(
                            f"LLMCrossover returned invalid value {child[name]!r} "
                            f"for gene '{name}' (options: {spec.options})"
                        )
            return child
        except Exception as exc:
            if self._raise_on_failure:
                raise
            self.fallback_count += 1
            print(f"[LLMCrossover WARNING] fallback #{self.fallback_count}: {exc}")
            return self._fallback.crossover(p1, p2, gene_builder)

    def describe(self) -> dict:
        return {
            'strategy': 'llm',
            'raise_on_failure': self._raise_on_failure,
            'fallback_count': self.fallback_count,
        }
