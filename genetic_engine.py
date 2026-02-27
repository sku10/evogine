import random
import multiprocessing as mp
import json
import time
from datetime import datetime, timezone
from typing import Callable, List, Tuple, Optional, Any

class GeneSpec:
    def sample(self) -> Any:
        raise NotImplementedError

    def mutate(self, value: Any, mutation_rate: float) -> Any:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class FloatRange(GeneSpec):
    def __init__(self, low: float, high: float, sigma: float = 0.1):
        self.low = low
        self.high = high
        self.sigma = sigma

    def sample(self):
        return random.uniform(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            noise = random.gauss(0, self.sigma * (self.high - self.low))
            value += noise
            return max(min(value, self.high), self.low)
        return value

    def describe(self):
        return {'type': 'FloatRange', 'low': self.low, 'high': self.high, 'sigma': self.sigma}


class IntRange(GeneSpec):
    def __init__(self, low: int, high: int, sigma: float = 0.05):
        """
        Args:
            low:   Minimum value (inclusive).
            high:  Maximum value (inclusive).
            sigma: Jump size as a fraction of the range width.
                   Default 0.05 = 5% of range per mutation step.
                   Example: IntRange(0, 100, sigma=0.05) jumps up to ±5.
                   Use smaller sigma for fine-tuning, larger for wide exploration.
        """
        self.low = low
        self.high = high
        self.sigma = sigma

    def sample(self):
        return random.randint(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            jump = max(1, round(self.sigma * (self.high - self.low)))
            delta = random.randint(-jump, jump)
            value = max(min(value + delta, self.high), self.low)
        return value

    def describe(self):
        return {'type': 'IntRange', 'low': self.low, 'high': self.high, 'sigma': self.sigma}


class ChoiceList(GeneSpec):
    def __init__(self, options: list):
        self.options = options

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
        return {'type': 'ChoiceList', 'options': self.options}


class GeneBuilder:
    def __init__(self):
        self.specs = {}
        self.order = []

    def add(self, name: str, spec: GeneSpec):
        self.specs[name] = spec
        self.order.append(name)

    def sample(self) -> dict:
        return {name: self.specs[name].sample() for name in self.order}

    def mutate(self, individual: dict, mutation_rate: float) -> dict:
        return {
            name: self.specs[name].mutate(individual[name], mutation_rate)
            for name in self.order
        }

    def keys(self):
        return self.order

    def describe(self) -> dict:
        return {name: self.specs[name].describe() for name in self.order}


Individual = List[float]
GeneRange = Tuple[float, float]
Population = List[Individual]


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
           k=2 is gentle, k=5+ is aggressive (best individual dominates).
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
    differ wildly in magnitude. Steady selection pressure throughout the run.
    """
    def select_parents(self, scored):
        n = len(scored)
        # scored is already sorted best-first; assign rank weights accordingly
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
        self, p1: dict, p2: dict, gene_builder: 'GeneBuilder'
    ) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class UniformCrossover(CrossoverStrategy):
    """
    Uniform crossover. Each gene is independently inherited from either
    parent with equal probability (50/50).
    Default crossover strategy.
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
    Produces offspring that interpolates between parents: child = t*p1 + (1-t)*p2
    where t is random in [0, 1]. Non-float genes fall back to uniform selection.

    Better than uniform for continuous landscapes — avoids hard jumps between
    parent values and explores the space between them.
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
    Single-point crossover. A random split index is chosen; genes before the
    split come from p1, genes after from p2.

    Preserves gene co-dependencies — genes that appear together often stay
    together. Useful when gene order is meaningful (e.g. a sequence of
    thresholds that build on each other).
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


class GeneticAlgorithm:
    def __init__(
        self,
        gene_builder: GeneBuilder,
        fitness_function: Callable[[dict], float],
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        elitism: int = 2,
        use_multiprocessing: bool = False,
        seed: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        log_path: Optional[str] = None,
        selection: Optional[SelectionStrategy] = None,
        crossover: Optional[CrossoverStrategy] = None,
        on_generation: Optional[Callable] = None,
    ):
        """
        Args:
            on_generation: Optional callback called after each generation.
                Signature: fn(gen, best_score, avg_score, best_individual) -> None
                Use for live plotting, custom logging, progress bars, etc.
                Example:
                    def my_callback(gen, best_score, avg_score, best_ind):
                        print(f"Gen {gen}: {best_score:.4f}")
        """
        self.genes = gene_builder
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.use_multiprocessing = use_multiprocessing
        self.patience = patience
        self.min_delta = min_delta
        self.log_path = log_path
        self._seed = seed
        self._selection = selection or RouletteSelection()
        self._crossover = crossover or UniformCrossover()
        self._on_generation = on_generation

    def create_individual(self) -> dict:
        return self.genes.sample()

    def mutate(self, ind: dict) -> dict:
        return self.genes.mutate(ind, self.mutation_rate)

    def evaluate_population(self, population: list[dict]) -> list[tuple[dict, float]]:
        if self.use_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                fitnesses = pool.map(self.fitness_function, population)
        else:
            fitnesses = [self.fitness_function(ind) for ind in population]
        return list(zip(population, fitnesses))

    def run(self) -> tuple[dict, float, list[dict]]:
        """
        Run the genetic algorithm.

        Returns:
            best_individual: dict of gene name -> value for the best solution found
            best_score:      fitness score of that individual
            history:         list of dicts per generation with keys:
                               gen, best_score, avg_score, improved, gens_without_improvement
        """
        if self._seed is not None:
            random.seed(self._seed)

        t_start = time.time()
        population = [self.create_individual() for _ in range(self.population_size)]

        best_overall = None
        best_score = float('-inf')
        history = []
        gens_without_improvement = 0
        convergence_gen = None

        for gen in range(1, self.generations + 1):
            scored = self.evaluate_population(population)
            scored.sort(key=lambda x: -x[1])  # Maximize fitness

            gen_best = scored[0][1]
            gen_avg = sum(x[1] for x in scored) / len(scored)
            improved = gen_best > best_score + self.min_delta

            if improved:
                best_score = gen_best
                best_overall = scored[0][0]
                gens_without_improvement = 0
                convergence_gen = gen
            else:
                gens_without_improvement += 1

            history.append({
                'gen': gen,
                'best_score': gen_best,
                'avg_score': gen_avg,
                'improved': improved,
                'gens_without_improvement': gens_without_improvement,
            })

            print(f"[GEN {gen:05}] Best: {gen_best:.11f} | Avg: {gen_avg:.11f}")

            if self._on_generation is not None:
                self._on_generation(gen, gen_best, gen_avg, best_overall)

            # Early stopping
            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

            # Elitism: carry over top N unchanged
            next_gen = [ind for ind, _ in scored[:self.elitism]]

            while len(next_gen) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1, p2 = self._selection.select_parents(scored)
                    child = self._crossover.crossover(p1, p2, self.genes)
                else:
                    child = random.choice(scored)[0].copy()

                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        elapsed = time.time() - t_start
        early_stopped = (
            self.patience is not None and
            gens_without_improvement >= self.patience
        )

        if self.log_path:
            self._write_log(best_overall, best_score, history, elapsed, early_stopped, convergence_gen)

        return best_overall, best_score, history

    def _write_log(
        self,
        best_individual: dict,
        best_score: float,
        history: list[dict],
        elapsed: float,
        early_stopped: bool,
        convergence_gen: Optional[int],
    ):
        scores = [h['best_score'] for h in history]
        first_score = scores[0] if scores else None
        last_score = scores[-1] if scores else None
        total_gens = len(history)
        improvements = sum(1 for h in history if h['improved'])

        # Classify convergence behaviour for AI readability
        if total_gens < 3:
            convergence_pattern = 'too_short_to_assess'
        elif improvements == 1:
            convergence_pattern = 'no_progress_after_gen1'
        elif early_stopped and convergence_gen and convergence_gen < total_gens * 0.3:
            convergence_pattern = 'converged_early'
        elif early_stopped:
            convergence_pattern = 'converged_midway'
        elif improvements > total_gens * 0.5:
            convergence_pattern = 'still_improving'
        else:
            convergence_pattern = 'converged_at_end'

        log = {
            'run': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
            },
            'config': {
                'population_size': self.population_size,
                'generations_max': self.generations,
                'generations_run': total_gens,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism': self.elitism,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'use_multiprocessing': self.use_multiprocessing,
                'selection': self._selection.describe(),
                'crossover': self._crossover.describe(),
            },
            'genes': self.genes.describe(),
            'result': {
                'best_score': best_score,
                'best_individual': best_individual,
                'early_stopped': early_stopped,
                'convergence_gen': convergence_gen,
            },
            'analysis': {
                'score_initial': first_score,
                'score_final': last_score,
                'score_improvement_total': (
                    round(last_score - first_score, 10) if first_score is not None else None
                ),
                'improvement_events': improvements,
                'convergence_pattern': convergence_pattern,
                'notes': self._analysis_notes(convergence_pattern, early_stopped, improvements, total_gens),
            },
            'history': history,
        }

        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)

        print(f"[LOG] Written to {self.log_path}")

    def _analysis_notes(
        self,
        pattern: str,
        early_stopped: bool,
        improvements: int,
        total_gens: int,
    ) -> list[str]:
        notes = []
        if pattern == 'no_progress_after_gen1':
            notes.append("Score only improved in the first generation. Population may have converged to a local optimum immediately. Consider increasing mutation_rate or population_size.")
        if pattern == 'converged_early':
            notes.append("Algorithm converged well before the generation limit. If the result is good, parameters are well-tuned. If not, increase mutation_rate to escape local optima.")
        if pattern == 'still_improving':
            notes.append("Score was still improving when the run ended. Consider increasing generations — the optimum may not have been reached yet.")
        if pattern == 'converged_midway':
            notes.append("Algorithm converged in the middle of the run. The remaining generation budget was unused. You could reduce generations or increase patience to save time.")
        if not early_stopped and improvements < total_gens * 0.1:
            notes.append("Very few improvement events relative to total generations. The search space may be too flat or mutation_rate too low to explore effectively.")
        if improvements > total_gens * 0.8:
            notes.append("Score improved in most generations, suggesting the problem is still being explored aggressively. This is healthy for early runs.")
        if not notes:
            notes.append("Run completed normally with no obvious issues detected.")
        return notes
