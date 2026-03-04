import random
from typing import Any, Optional


class GeneSpec:
    """
    Abstract base for all gene types.
    Subclass and implement sample(), mutate(), describe().

    Optional per-gene mutation rate:
        Set gene_mutation_rate on the instance to override the GA's global rate.
        Example:
            spec = FloatRange(0.0, 1.0, mutation_rate=0.5)
        When set, this gene always uses this rate regardless of the global rate.
    """
    gene_mutation_rate: Optional[float] = None

    def sample(self) -> Any:
        raise NotImplementedError

    def mutate(self, value: Any, mutation_rate: float) -> Any:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class FloatRange(GeneSpec):
    def __init__(
        self,
        low: float,
        high: float,
        sigma: float = 0.1,
        mutation_rate: Optional[float] = None,
        mutation_dist: str = 'gaussian',
    ):
        if mutation_dist not in ('gaussian', 'levy'):
            raise ValueError(
                f"mutation_dist must be 'gaussian' or 'levy', got {mutation_dist!r}"
            )
        self.low = low
        self.high = high
        self.sigma = sigma
        self.gene_mutation_rate = mutation_rate
        self.mutation_dist = mutation_dist

    def sample(self):
        return random.uniform(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            if self.mutation_dist == 'levy':
                # Chambers-Mallows-Stuck approximation for Levy(0,1)
                u = random.gauss(0, 1)
                v = random.gauss(0, 1)
                if v == 0:
                    v = 1e-300
                noise = u / abs(v) ** 0.5
                noise *= self.sigma * (self.high - self.low)
            else:
                noise = random.gauss(0, self.sigma * (self.high - self.low))
            value += noise
            return max(min(value, self.high), self.low)
        return value

    def describe(self):
        d = {
            'type': 'FloatRange', 'low': self.low, 'high': self.high,
            'sigma': self.sigma, 'mutation_dist': self.mutation_dist,
        }
        if self.gene_mutation_rate is not None:
            d['mutation_rate'] = self.gene_mutation_rate
        return d


class IntRange(GeneSpec):
    def __init__(
        self,
        low: int,
        high: int,
        sigma: float = 0.05,
        mutation_rate: Optional[float] = None,
    ):
        """
        Args:
            low:           Minimum value (inclusive).
            high:          Maximum value (inclusive).
            sigma:         Jump size as a fraction of the range width.
                           Default 0.05 = 5% of range per mutation step.
                           Example: IntRange(0, 100, sigma=0.05) jumps up to ±5.
            mutation_rate: Per-gene override for the global mutation rate.
        """
        self.low = low
        self.high = high
        self.sigma = sigma
        self.gene_mutation_rate = mutation_rate

    def sample(self):
        return random.randint(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            jump = max(1, round(self.sigma * (self.high - self.low)))
            delta = random.randint(-jump, jump)
            value = max(min(value + delta, self.high), self.low)
        return value

    def describe(self):
        d = {'type': 'IntRange', 'low': self.low, 'high': self.high, 'sigma': self.sigma}
        if self.gene_mutation_rate is not None:
            d['mutation_rate'] = self.gene_mutation_rate
        return d


class ChoiceList(GeneSpec):
    def __init__(self, options: list, mutation_rate: Optional[float] = None):
        if len(options) == 0:
            raise ValueError("ChoiceList requires at least one option.")
        self.options = list(options)
        self.gene_mutation_rate = mutation_rate

    def sample(self):
        return random.choice(self.options)

    def mutate(self, value, mutation_rate):
        if len(self.options) <= 1:
            return value
        if random.random() < mutation_rate:
            idx = self.options.index(value)
            choices = [i for i in range(len(self.options)) if i != idx]
            return self.options[random.choice(choices)]
        return value

    def describe(self):
        d = {'type': 'ChoiceList', 'options': self.options}
        if self.gene_mutation_rate is not None:
            d['mutation_rate'] = self.gene_mutation_rate
        return d


class GeneBuilder:
    def __init__(self):
        self.specs = {}
        self.order = []

    def add(self, name: str, spec: GeneSpec):
        if name in self.specs:
            raise ValueError(f"Gene '{name}' already exists in this GeneBuilder.")
        self.specs[name] = spec
        self.order.append(name)

    def sample(self) -> dict:
        return {name: self.specs[name].sample() for name in self.order}

    def mutate(self, individual: dict, mutation_rate: float) -> dict:
        """
        Mutate each gene, using the gene's own mutation_rate if set,
        otherwise the global mutation_rate passed in.
        """
        result = {}
        for name in self.order:
            spec = self.specs[name]
            effective_rate = (
                spec.gene_mutation_rate
                if spec.gene_mutation_rate is not None
                else mutation_rate
            )
            result[name] = spec.mutate(individual[name], effective_rate)
        return result

    def keys(self):
        return self.order

    def describe(self) -> dict:
        return {name: self.specs[name].describe() for name in self.order}
