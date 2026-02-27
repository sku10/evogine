import random
import math
import multiprocessing as mp
import json
import time
import os
from datetime import datetime, timezone
from typing import Callable, Optional, Any


# ---------------------------------------------------------------------------
# Seeding helper
# ---------------------------------------------------------------------------

def _seed_all(seed: Optional[int]) -> None:
    """Seed random and numpy.random (if numpy is available)."""
    if seed is not None:
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Gene specs
# ---------------------------------------------------------------------------

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
    ):
        self.low = low
        self.high = high
        self.sigma = sigma
        self.gene_mutation_rate = mutation_rate

    def sample(self):
        return random.uniform(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            noise = random.gauss(0, self.sigma * (self.high - self.low))
            value += noise
            return max(min(value, self.high), self.low)
        return value

    def describe(self):
        d = {'type': 'FloatRange', 'low': self.low, 'high': self.high, 'sigma': self.sigma}
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
        self, p1: dict, p2: dict, gene_builder: 'GeneBuilder'
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


# ---------------------------------------------------------------------------
# GeneticAlgorithm
# ---------------------------------------------------------------------------

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
        mode: str = 'maximize',
        adaptive_mutation: bool = False,
        adaptive_mutation_min: float = 0.01,
        adaptive_mutation_max: float = 0.5,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 1,
        restart_after: Optional[int] = None,
        restart_fraction: float = 0.3,
    ):
        """
        Args:
            mode: 'maximize' (default) or 'minimize'.
                  With 'minimize', your fitness function returns the value to
                  minimize directly — no negation needed.

            adaptive_mutation: Automatically adjust mutation_rate each generation.
                  On improvement: rate *= 0.95 (fine-tune). On stagnation: rate *= 1.10
                  (explore). Stays within [adaptive_mutation_min, adaptive_mutation_max].

            checkpoint_path: Path to save a checkpoint JSON after every
                  checkpoint_every generations and on early stop. Allows resuming
                  an interrupted run via ga.run(resume_from=checkpoint_path).

            checkpoint_every: Save checkpoint every N generations (default 1).

            restart_after: Inject fresh random individuals into the population
                  after this many consecutive generations without improvement.
                  Helps escape local optima without discarding good individuals.
                  Example: restart_after=20 injects fresh blood every 20 stagnant gens.

            restart_fraction: Fraction of population to replace on restart (default 0.3).
                  Elites are always preserved. Only non-elite slots are replaced.

            on_generation: Callback called after each generation.
                  Signature: fn(gen, best_score, avg_score, best_individual) -> None

            adaptive_mutation_min: Lower bound for adaptive mutation rate (default 0.01).
            adaptive_mutation_max: Upper bound for adaptive mutation rate (default 0.5).
        """
        if mode not in ('maximize', 'minimize'):
            raise ValueError(f"mode must be 'maximize' or 'minimize', got {mode!r}")
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
        self._mode = mode
        self._sign = -1 if mode == 'minimize' else 1
        self._adaptive_mutation = adaptive_mutation
        self._adaptive_mutation_min = adaptive_mutation_min
        self._adaptive_mutation_max = adaptive_mutation_max
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._restart_after = restart_after
        self._restart_fraction = restart_fraction

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
        if self._mode == 'minimize':
            fitnesses = [-f for f in fitnesses]
        return list(zip(population, fitnesses))

    def _compute_diversity(self, population: list[dict]) -> float:
        """
        Population diversity as the average normalized spread per gene.

        FloatRange / IntRange: spread = (max_val - min_val) / (high - low)
        ChoiceList: fraction of unique values present in the population

        Returns a value in [0.0, 1.0].
        """
        if len(population) < 2:
            return 0.0
        diversities = []
        for name in self.genes.order:
            spec = self.genes.specs[name]
            values = [ind[name] for ind in population]
            if isinstance(spec, (FloatRange, IntRange)):
                range_width = spec.high - spec.low
                if range_width > 0:
                    spread = (max(values) - min(values)) / range_width
                    diversities.append(min(1.0, spread))
            elif isinstance(spec, ChoiceList):
                if len(spec.options) > 1:
                    unique_fraction = len(set(str(v) for v in values)) / len(spec.options)
                    diversities.append(min(1.0, unique_fraction))
        return round(sum(diversities) / len(diversities), 6) if diversities else 0.0

    def _save_checkpoint(
        self,
        gen: int,
        population: list[dict],
        best_individual: Optional[dict],
        best_score_internal: float,
        gens_without_improvement: int,
        convergence_gen: Optional[int],
        history: list[dict],
    ) -> None:
        ckpt = {
            'gen': gen,
            'population': population,
            'best_individual': best_individual,
            'best_score_internal': best_score_internal,
            'gens_without_improvement': gens_without_improvement,
            'convergence_gen': convergence_gen,
            'history': history,
            'mutation_rate': self.mutation_rate,
            'config': {
                'generations': self.generations,
                'mode': self._mode,
                'seed': self._seed,
            },
        }
        dir_part = os.path.dirname(self._checkpoint_path)
        if dir_part:
            os.makedirs(dir_part, exist_ok=True)
        with open(self._checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, indent=2)
        print(f"[CHECKPOINT] Saved gen {gen} → {self._checkpoint_path}")

    def _load_checkpoint(self, path: str) -> dict:
        with open(path, encoding='utf-8') as f:
            ckpt = json.load(f)
        print(f"[CHECKPOINT] Resuming from gen {ckpt['gen'] + 1} ← {path}")
        return ckpt

    def run(self, resume_from: Optional[str] = None) -> tuple[dict, float, list[dict]]:
        """
        Run the genetic algorithm.

        Args:
            resume_from: Optional path to a checkpoint file. When provided,
                         resumes the run from the saved state instead of starting fresh.

        Returns:
            best_individual: dict of gene name -> value for the best solution found
            best_score:      fitness score of that individual (real value)
            history:         list of dicts per generation with keys:
                               gen, best_score, avg_score, improved,
                               gens_without_improvement, mutation_rate,
                               diversity, restarted
        """
        _seed_all(self._seed)
        t_start = time.time()

        if resume_from is not None:
            state = self._load_checkpoint(resume_from)
            population = state['population']
            best_overall = state['best_individual']
            best_score = state['best_score_internal']
            history = state['history']
            gens_without_improvement = state['gens_without_improvement']
            convergence_gen = state['convergence_gen']
            start_gen = state['gen'] + 1
            if self._adaptive_mutation and 'mutation_rate' in state:
                self.mutation_rate = state['mutation_rate']
        else:
            population = [self.create_individual() for _ in range(self.population_size)]
            best_overall = None
            best_score = float('-inf')
            history = []
            gens_without_improvement = 0
            convergence_gen = None
            start_gen = 1

        for gen in range(start_gen, self.generations + 1):
            scored = self.evaluate_population(population)
            scored.sort(key=lambda x: -x[1])

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

            if self._adaptive_mutation:
                if improved:
                    self.mutation_rate *= 0.95
                else:
                    self.mutation_rate *= 1.1
                self.mutation_rate = max(
                    self._adaptive_mutation_min,
                    min(self.mutation_rate, self._adaptive_mutation_max),
                )

            diversity = self._compute_diversity([ind for ind, _ in scored])
            # best_score in history is the all-time running best (non-decreasing
            # in maximize, non-increasing in minimize), consistent across all optimizers.
            running_best_real = best_score * self._sign
            real_avg  = gen_avg  * self._sign

            # Stagnation restart check
            restarted = (
                self._restart_after is not None
                and gens_without_improvement > 0
                and gens_without_improvement % self._restart_after == 0
            )

            history.append({
                'gen': gen,
                'best_score': running_best_real,
                'avg_score': real_avg,
                'improved': improved,
                'gens_without_improvement': gens_without_improvement,
                'mutation_rate': round(self.mutation_rate, 6),
                'diversity': diversity,
                'restarted': restarted,
            })

            print(f"[GEN {gen:05}] Best: {running_best_real:.11f} | Avg: {real_avg:.11f} | Diversity: {diversity:.4f}")

            if self._on_generation is not None:
                self._on_generation(gen, running_best_real, real_avg, best_overall)

            # Checkpoint
            if (
                self._checkpoint_path is not None
                and gen % self._checkpoint_every == 0
            ):
                self._save_checkpoint(
                    gen, population, best_overall, best_score,
                    gens_without_improvement, convergence_gen, history,
                )

            # Early stopping
            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                if self._checkpoint_path is not None:
                    self._save_checkpoint(
                        gen, population, best_overall, best_score,
                        gens_without_improvement, convergence_gen, history,
                    )
                break

            # Build next generation
            next_gen = [ind for ind, _ in scored[:self.elitism]]

            # Stagnation restart: inject fresh individuals (after elites)
            if restarted:
                n_inject = max(1, int(self._restart_fraction * self.population_size))
                for _ in range(n_inject):
                    next_gen.append(self.create_individual())

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
            self.patience is not None
            and gens_without_improvement >= self.patience
        )
        real_best_score = best_score * self._sign

        if self.log_path:
            self._write_log(
                best_overall, real_best_score, history, elapsed,
                early_stopped, convergence_gen,
            )

        return best_overall, real_best_score, history

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
                'type': 'single_objective',
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
                'mode': self._mode,
                'adaptive_mutation': self._adaptive_mutation,
                'adaptive_mutation_min': self._adaptive_mutation_min if self._adaptive_mutation else None,
                'adaptive_mutation_max': self._adaptive_mutation_max if self._adaptive_mutation else None,
                'restart_after': self._restart_after,
                'restart_fraction': self._restart_fraction if self._restart_after else None,
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
                    round(last_score - first_score, 10)
                    if first_score is not None else None
                ),
                'improvement_events': improvements,
                'convergence_pattern': convergence_pattern,
                'notes': self._analysis_notes(
                    convergence_pattern, early_stopped, improvements, total_gens,
                ),
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


# ---------------------------------------------------------------------------
# Island Model
# ---------------------------------------------------------------------------

class IslandModel:
    """
    Multiple independent GA sub-populations (islands) with periodic migration.

    Each island evolves independently. Every migration_interval generations,
    the top migration_size individuals from each island are copied into the
    next island (ring topology: 0→1→2→...→n-1→0).

    This maintains diversity (islands explore different regions) while
    sharing good solutions via migration.

    Args:
        n_islands:          Number of sub-populations (default 4).
        island_population:  Individuals per island (default 50).
        migration_interval: Migrate every N generations (default 10).
        migration_size:     Top K individuals to copy per migration (default 2).

    All other parameters (mutation_rate, crossover_rate, etc.) work the same
    as GeneticAlgorithm and are applied identically to every island.

    Returns from run():
        best_individual: Best solution found across all islands.
        best_score:      Its fitness score.
        history:         Per-generation dicts with island_bests list.
    """

    def __init__(
        self,
        gene_builder: GeneBuilder,
        fitness_function: Callable[[dict], float],
        n_islands: int = 4,
        island_population: int = 50,
        generations: int = 100,
        migration_interval: int = 10,
        migration_size: int = 2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        elitism: int = 2,
        use_multiprocessing: bool = False,
        seed: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        selection: Optional[SelectionStrategy] = None,
        crossover: Optional[CrossoverStrategy] = None,
        mode: str = 'maximize',
        log_path: Optional[str] = None,
        on_generation: Optional[Callable] = None,
    ):
        if mode not in ('maximize', 'minimize'):
            raise ValueError(f"mode must be 'maximize' or 'minimize', got {mode!r}")
        self.genes = gene_builder
        self.fitness_function = fitness_function
        self.n_islands = n_islands
        self.island_population = island_population
        self.generations = generations
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.use_multiprocessing = use_multiprocessing
        self._seed = seed
        self.patience = patience
        self.min_delta = min_delta
        self._selection = selection or RouletteSelection()
        self._crossover = crossover or UniformCrossover()
        self._mode = mode
        self._sign = -1 if mode == 'minimize' else 1
        self.log_path = log_path
        self._on_generation = on_generation

    def _evaluate_population(self, population: list[dict]) -> list[tuple[dict, float]]:
        if self.use_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                fitnesses = pool.map(self.fitness_function, population)
        else:
            fitnesses = [self.fitness_function(ind) for ind in population]
        if self._mode == 'minimize':
            fitnesses = [-f for f in fitnesses]
        return list(zip(population, fitnesses))

    def _evolve_island(
        self, population: list[dict]
    ) -> tuple[list[dict], list[tuple[dict, float]]]:
        """Evolve one island for one generation. Returns (new_population, scored)."""
        scored = self._evaluate_population(population)
        scored.sort(key=lambda x: -x[1])

        next_gen = [ind for ind, _ in scored[:self.elitism]]
        while len(next_gen) < self.island_population:
            if random.random() < self.crossover_rate:
                p1, p2 = self._selection.select_parents(scored)
                child = self._crossover.crossover(p1, p2, self.genes)
            else:
                child = random.choice(scored)[0].copy()
            child = self.genes.mutate(child, self.mutation_rate)
            next_gen.append(child)

        return next_gen, scored

    def run(self) -> tuple[dict, float, list[dict]]:
        """
        Run the island model.

        Returns:
            best_individual: Best solution found across all islands.
            best_score:      Its fitness score (real value).
            history:         Per-generation dicts with keys:
                               gen, best_score, avg_score, island_bests,
                               improved, gens_without_improvement.
        """
        _seed_all(self._seed)
        t_start = time.time()

        populations = [
            [self.genes.sample() for _ in range(self.island_population)]
            for _ in range(self.n_islands)
        ]

        best_overall = None
        best_score = float('-inf')
        history = []
        gens_without_improvement = 0
        convergence_gen = None

        for gen in range(1, self.generations + 1):
            all_island_scored: list[list[tuple[dict, float]]] = []
            for i in range(self.n_islands):
                populations[i], island_scored = self._evolve_island(populations[i])
                all_island_scored.append(island_scored)

            all_scored = [
                (ind, sc)
                for island in all_island_scored
                for ind, sc in island
            ]
            gen_best_sc = max(sc for _, sc in all_scored)
            gen_avg_sc = sum(sc for _, sc in all_scored) / len(all_scored)
            island_bests = [
                round(max(sc for _, sc in isl) * self._sign, 8)
                for isl in all_island_scored
            ]

            improved = gen_best_sc > best_score + self.min_delta
            if improved:
                best_score = gen_best_sc
                best_overall = max(all_scored, key=lambda x: x[1])[0]
                gens_without_improvement = 0
                convergence_gen = gen
            else:
                gens_without_improvement += 1

            # best_score = all-time running best, consistent across all optimizers
            running_best_real = best_score * self._sign
            real_avg  = gen_avg_sc  * self._sign

            history.append({
                'gen': gen,
                'best_score': running_best_real,
                'avg_score': real_avg,
                'island_bests': island_bests,
                'improved': improved,
                'gens_without_improvement': gens_without_improvement,
            })

            print(
                f"[GEN {gen:05}] Best: {running_best_real:.8f} | "
                f"Avg: {real_avg:.8f} | Islands: {island_bests}"
            )

            if self._on_generation is not None:
                self._on_generation(gen, running_best_real, real_avg, best_overall)

            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

            # Migration every migration_interval gens (ring topology)
            if gen % self.migration_interval == 0 and self.n_islands > 1:
                for i in range(self.n_islands):
                    source = sorted(all_island_scored[i], key=lambda x: -x[1])
                    migrants = [ind.copy() for ind, _ in source[:self.migration_size]]
                    target_idx = (i + 1) % self.n_islands
                    # Sort target island by fitness (worst last), replace worst
                    target_scored = list(zip(
                        populations[target_idx],
                        [sc for _, sc in all_island_scored[target_idx]]
                    ))
                    target_scored.sort(key=lambda x: -x[1])
                    populations[target_idx] = (
                        [ind for ind, _ in target_scored[:-self.migration_size]]
                        + migrants
                    )
                print(
                    f"[MIGRATION] Gen {gen}: "
                    f"{self.migration_size} migrants between {self.n_islands} islands"
                )

        elapsed = time.time() - t_start
        early_stopped = (
            self.patience is not None
            and gens_without_improvement >= self.patience
        )
        real_best_score = best_score * self._sign

        if self.log_path:
            self._write_log(
                best_overall, real_best_score, history, elapsed,
                early_stopped, convergence_gen,
            )

        return best_overall, real_best_score, history

    def _write_log(
        self,
        best_individual: dict,
        best_score: float,
        history: list[dict],
        elapsed: float,
        early_stopped: bool,
        convergence_gen: Optional[int],
    ) -> None:
        log = {
            'run': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'type': 'island_model',
            },
            'config': {
                'n_islands': self.n_islands,
                'island_population': self.island_population,
                'total_population': self.n_islands * self.island_population,
                'generations_max': self.generations,
                'generations_run': len(history),
                'migration_interval': self.migration_interval,
                'migration_size': self.migration_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism': self.elitism,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'mode': self._mode,
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
            'history': history,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        print(f"[LOG] Written to {self.log_path}")


# ---------------------------------------------------------------------------
# Multi-objective GA (NSGA-II style)
# ---------------------------------------------------------------------------

class MultiObjectiveGA:
    """
    Multi-objective genetic algorithm using NSGA-II Pareto ranking.

    Fitness function must return a list/tuple of floats — one per objective.
    Each objective can be independently maximized or minimized.

    Returns a Pareto front: a list of non-dominated solutions — rather than
    a single best individual. The caller chooses the preferred trade-off.

    Args:
        fitness_function: fn(dict) -> list[float]  (one value per objective)
        n_objectives:     Number of objectives (must match length of fitness output).
        objectives:       List of 'maximize'/'minimize' per objective.
                          Default: all 'maximize'.

    Example:
        def fitness(ind):
            sharpe = run_backtest(ind).sharpe_ratio
            drawdown = run_backtest(ind).max_drawdown  # positive = bad
            return [sharpe, -drawdown]   # maximize both

        ga = MultiObjectiveGA(
            gene_builder=genes,
            fitness_function=fitness,
            n_objectives=2,
            objectives=['maximize', 'maximize'],
        )
        pareto_front, history = ga.run()
        # pareto_front: [{'individual': dict, 'scores': [sharpe, -drawdown]}, ...]

    Return value of run():
        pareto_front: list of dicts, each with:
            'individual': gene value dict
            'scores':     list of real objective values (un-negated)
        history: list of per-generation dicts with:
            gen, pareto_size, hypervolume_proxy, improved, gens_without_improvement
    """

    def __init__(
        self,
        gene_builder: GeneBuilder,
        fitness_function: Callable[[dict], list[float]],
        n_objectives: int,
        objectives: Optional[list[str]] = None,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.5,
        use_multiprocessing: bool = False,
        seed: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        crossover: Optional[CrossoverStrategy] = None,
        log_path: Optional[str] = None,
        on_generation: Optional[Callable] = None,
    ):
        if objectives is None:
            objectives = ['maximize'] * n_objectives
        if len(objectives) != n_objectives:
            raise ValueError(
                f"len(objectives)={len(objectives)} != n_objectives={n_objectives}"
            )
        for obj in objectives:
            if obj not in ('maximize', 'minimize'):
                raise ValueError(
                    f"Each objective must be 'maximize' or 'minimize', got {obj!r}"
                )

        self.genes = gene_builder
        self.fitness_function = fitness_function
        self.n_objectives = n_objectives
        self.objectives = objectives
        self._signs = [-1 if obj == 'minimize' else 1 for obj in objectives]
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.use_multiprocessing = use_multiprocessing
        self._seed = seed
        self.patience = patience
        self.min_delta = min_delta
        self._crossover = crossover or UniformCrossover()
        self.log_path = log_path
        self._on_generation = on_generation

    def _evaluate(self, population: list[dict]) -> list[list[float]]:
        """
        Evaluate population. Returns sign-adjusted score vectors
        (negate 'minimize' objectives so we always maximize internally).
        """
        if self.use_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                raw = pool.map(self.fitness_function, population)
        else:
            raw = [self.fitness_function(ind) for ind in population]
        return [
            [score * sign for score, sign in zip(scores, self._signs)]
            for scores in raw
        ]

    def _dominates(self, a: list[float], b: list[float]) -> bool:
        """Return True if a Pareto-dominates b."""
        no_worse = all(ai >= bi for ai, bi in zip(a, b))
        strictly_better = any(ai > bi for ai, bi in zip(a, b))
        return no_worse and strictly_better

    def _non_dominated_sort(
        self, scored: list[tuple[dict, list[float]]]
    ) -> list[list[int]]:
        """
        NSGA-II non-dominated sorting.
        Returns fronts: list of lists of indices into scored.
        Front 0 is the non-dominated Pareto front.
        """
        n = len(scored)
        domination_count = [0] * n
        dominated_by: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(scored[i][1], scored[j][1]):
                    dominated_by[i].append(j)
                elif self._dominates(scored[j][1], scored[i][1]):
                    domination_count[i] += 1

        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        return fronts

    def _crowding_distance(
        self,
        front_indices: list[int],
        scored: list[tuple[dict, list[float]]],
    ) -> dict[int, float]:
        """Compute crowding distance for individuals in a front."""
        distances: dict[int, float] = {i: 0.0 for i in front_indices}
        n = len(front_indices)

        if n <= 2:
            for i in front_indices:
                distances[i] = float('inf')
            return distances

        for obj_idx in range(self.n_objectives):
            sorted_front = sorted(
                front_indices, key=lambda i: scored[i][1][obj_idx]
            )
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            obj_min = scored[sorted_front[0]][1][obj_idx]
            obj_max = scored[sorted_front[-1]][1][obj_idx]
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            for k in range(1, n - 1):
                distances[sorted_front[k]] += (
                    scored[sorted_front[k + 1]][1][obj_idx]
                    - scored[sorted_front[k - 1]][1][obj_idx]
                ) / obj_range

        return distances

    def _nsga2_select(
        self, scored: list[tuple[dict, list[float]]]
    ) -> list[dict]:
        """Select next generation using NSGA-II ranking + crowding distance."""
        fronts = self._non_dominated_sort(scored)
        next_gen_indices: list[int] = []

        for front in fronts:
            if len(next_gen_indices) + len(front) <= self.population_size:
                next_gen_indices.extend(front)
            else:
                distances = self._crowding_distance(front, scored)
                remaining = self.population_size - len(next_gen_indices)
                sorted_front = sorted(front, key=lambda i: -distances[i])
                next_gen_indices.extend(sorted_front[:remaining])
                break

        return [scored[i][0] for i in next_gen_indices]

    def _tournament_crowding(
        self,
        scored: list[tuple[dict, list[float]]],
        rank_map: dict[int, int],
        dist_map: dict[int, float],
    ) -> dict:
        """Binary tournament: prefer lower rank, then higher crowding distance."""
        i1, i2 = random.sample(range(len(scored)), min(2, len(scored)))
        r1, r2 = rank_map.get(i1, 999), rank_map.get(i2, 999)
        if r1 < r2:
            return scored[i1][0]
        elif r2 < r1:
            return scored[i2][0]
        elif dist_map.get(i1, 0) >= dist_map.get(i2, 0):
            return scored[i1][0]
        else:
            return scored[i2][0]

    def run(self) -> tuple[list[dict], list[dict]]:
        """
        Run the multi-objective genetic algorithm.

        Returns:
            pareto_front: Non-dominated solutions. Each entry:
                          {'individual': dict, 'scores': list[float]}
                          Scores are real values (un-negated).
            history:      Per-generation dicts with keys:
                          gen, pareto_size, hypervolume_proxy,
                          improved, gens_without_improvement.
        """
        _seed_all(self._seed)
        t_start = time.time()

        population = [self.genes.sample() for _ in range(self.population_size)]
        history = []
        best_hypervolume = float('-inf')
        gens_without_improvement = 0

        for gen in range(1, self.generations + 1):
            score_vecs = self._evaluate(population)
            scored = list(zip(population, score_vecs))

            fronts = self._non_dominated_sort(scored)
            front_0 = fronts[0] if fronts else []
            pareto_size = len(front_0)

            # Hypervolume proxy: mean sum-of-objectives on the Pareto front
            if front_0:
                hv_proxy = sum(
                    sum(scored[i][1]) / self.n_objectives
                    for i in front_0
                ) / len(front_0)
            else:
                hv_proxy = float('-inf')

            improved = hv_proxy > best_hypervolume + self.min_delta
            if improved:
                best_hypervolume = hv_proxy
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            # Real values for reporting (un-negate minimized objectives)
            pareto_front_real = [
                {
                    'individual': scored[i][0],
                    'scores': [
                        scored[i][1][j] * self._signs[j]
                        for j in range(self.n_objectives)
                    ],
                }
                for i in front_0
            ]

            history.append({
                'gen': gen,
                'pareto_size': pareto_size,
                'hypervolume_proxy': round(hv_proxy, 8),
                'improved': improved,
                'gens_without_improvement': gens_without_improvement,
            })

            print(
                f"[GEN {gen:05}] Pareto front: {pareto_size} solutions | "
                f"HV proxy: {hv_proxy:.6f}"
            )

            if self._on_generation is not None:
                self._on_generation(gen, pareto_size, hv_proxy, pareto_front_real)

            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

            # --- NSGA-II (mu+lambda): generate offspring, merge with parents, select ---
            # 1. Generate offspring via crowding-tournament selection on parents
            offspring = []
            # Pre-compute ranks and crowding distances for tournament
            rank_map: dict[int, int] = {}
            dist_map: dict[int, float] = {}
            for rank, front in enumerate(fronts):
                distances = self._crowding_distance(front, scored)
                for idx in front:
                    rank_map[idx] = rank
                    dist_map[idx] = distances[idx]

            while len(offspring) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1 = self._tournament_crowding(scored, rank_map, dist_map)
                    p2 = self._tournament_crowding(scored, rank_map, dist_map)
                    child = self._crossover.crossover(p1, p2, self.genes)
                else:
                    child = self._tournament_crowding(scored, rank_map, dist_map).copy()
                child = self.genes.mutate(child, self.mutation_rate)
                offspring.append(child)

            # 2. Merge parents + offspring (mu+lambda)
            combined = [ind for ind, _ in scored] + offspring
            combined_vecs = self._evaluate(combined)
            combined_scored = list(zip(combined, combined_vecs))

            # 3. Select next generation from combined pool
            population = self._nsga2_select(combined_scored)

        # Final Pareto front
        final_vecs = self._evaluate(population)
        final_scored = list(zip(population, final_vecs))
        final_fronts = self._non_dominated_sort(final_scored)
        final_front_0 = final_fronts[0] if final_fronts else []

        final_pareto = [
            {
                'individual': final_scored[i][0],
                'scores': [
                    final_scored[i][1][j] * self._signs[j]
                    for j in range(self.n_objectives)
                ],
            }
            for i in final_front_0
        ]

        elapsed = time.time() - t_start
        if self.log_path:
            self._write_log(final_pareto, history, elapsed)

        return final_pareto, history

    def _write_log(
        self,
        pareto_front: list[dict],
        history: list[dict],
        elapsed: float,
    ) -> None:
        log = {
            'run': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'type': 'multi_objective',
            },
            'config': {
                'n_objectives': self.n_objectives,
                'objectives': self.objectives,
                'population_size': self.population_size,
                'generations_max': self.generations,
                'generations_run': len(history),
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'use_multiprocessing': self.use_multiprocessing,
                'crossover': self._crossover.describe(),
            },
            'genes': self.genes.describe(),
            'result': {
                'pareto_front_size': len(pareto_front),
                'pareto_front': pareto_front,
            },
            'history': history,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        print(f"[LOG] Written to {self.log_path}")


# ---------------------------------------------------------------------------
# CMA-ES Optimizer
# ---------------------------------------------------------------------------

class CMAESOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

    For FloatRange-only problems. CMA-ES learns the shape of the fitness landscape
    by maintaining and adapting a covariance matrix — effectively learning which
    directions in gene space lead toward better solutions.

    Why use CMA-ES over GeneticAlgorithm?
    - 10-100x faster convergence on problems with only FloatRange genes
    - Automatically adapts to correlations between genes
    - Especially effective when the optimum lies in a curved or diagonal valley

    When to use GeneticAlgorithm instead:
    - You have IntRange or ChoiceList genes (CMA-ES does not support these)
    - Your fitness landscape is highly multimodal (IslandModel helps more)

    Args:
        gene_builder:      GeneBuilder containing only FloatRange genes (min 2).
        fitness_function:  Callable (dict -> float). Same interface as GeneticAlgorithm.
        sigma0:            Initial step size, relative to gene range (default 0.3).
                           Higher means more exploration initially. Valid range: 0.0-0.5.
        generations:       Maximum number of generations (default 200).
        popsize:           Population size lambda. Default: 4 + floor(3*ln(n)).
                           Increase for multimodal landscapes.
        patience:          Stop after this many generations without improvement.
                           None means run to full generation limit.
        min_delta:         Minimum improvement to reset the patience counter.
        mode:              'maximize' (default) or 'minimize'.
        seed:              Random seed for reproducibility.
        log_path:          Path to write a JSON log (same format as GeneticAlgorithm logs).
        tolx:              Stop when sigma * max(eigenvalues) < tolx (step size too small).
        tolfun:            Stop when best score has not changed meaningfully in recent gens.

    Returns from run():
        (best_individual, best_score, history)
        Same shape as GeneticAlgorithm.run(). history entries contain:
            gen, best_score, avg_score, sigma, improved,
            gens_without_improvement, stop_reason
    """

    def __init__(
        self,
        gene_builder: 'GeneBuilder',
        fitness_function: Callable[[dict], float],
        sigma0: float = 0.3,
        generations: int = 200,
        popsize: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 1e-9,
        mode: str = 'maximize',
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        tolx: float = 1e-8,
        tolfun: float = 1e-10,
        on_generation: Optional[Callable] = None,
    ):
        if mode not in ('maximize', 'minimize'):
            raise ValueError("mode must be 'maximize' or 'minimize'")

        for name, spec in gene_builder.specs.items():
            if not isinstance(spec, FloatRange):
                raise ValueError(
                    f"CMAESOptimizer only supports FloatRange genes. "
                    f"Gene '{name}' is {type(spec).__name__}. "
                    f"Use GeneticAlgorithm for mixed gene types."
                )

        n = len(gene_builder.order)
        if n < 2:
            raise ValueError(
                "CMAESOptimizer requires at least 2 genes. "
                "For 1-dimensional problems use GeneticAlgorithm."
            )

        self.genes            = gene_builder
        self.fitness_function = fitness_function
        self.sigma0           = sigma0
        self.generations      = generations
        self.patience         = patience
        self.min_delta        = min_delta
        self._mode            = mode
        self._sign            = 1.0 if mode == 'maximize' else -1.0
        self._seed            = seed
        self.log_path         = log_path
        self.tolx             = tolx
        self.tolfun           = tolfun
        self._on_generation   = on_generation

        self._n   = n
        self._lam = popsize if popsize is not None else (4 + int(3 * math.log(n)))
        self._mu  = self._lam // 2

    def _to_individual(self, x) -> dict:
        """Convert a normalized [0,1]^n vector to a gene dict, clamped to gene ranges."""
        ind = {}
        for i, name in enumerate(self.genes.order):
            spec = self.genes.specs[name]
            v = max(0.0, min(1.0, float(x[i])))
            ind[name] = spec.low + v * (spec.high - spec.low)
        return ind

    def _evaluate(self, x) -> float:
        """Evaluate a normalized vector. Returns internal score (always-maximize sign)."""
        return self._sign * self.fitness_function(self._to_individual(x))

    def run(self) -> tuple[dict, float, list]:
        """
        Run the CMA-ES optimization. Requires numpy.

        Returns:
            (best_individual, best_score, history)
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                "CMAESOptimizer requires numpy. Install it with: pip install numpy"
            )

        _seed_all(self._seed)

        n   = self._n
        lam = self._lam
        mu  = self._mu

        # --- Recombination weights (log-based, normalized to sum=1) ---
        weights_raw = np.array(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        )
        weights = weights_raw / weights_raw.sum()
        mueff   = 1.0 / float((weights ** 2).sum())

        # --- Strategy parameters (Hansen canonical defaults) ---
        cc    = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs    = (mueff + 2) / (n + mueff + 5)
        c1    = 2 / ((n + 1.3) ** 2 + mueff)
        cmu   = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN  = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # --- CMA-ES state ---
        m         = np.full(n, 0.5)    # mean in normalized [0,1]^n space
        sigma     = self.sigma0        # global step size
        pc        = np.zeros(n)        # evolution path for C
        ps        = np.zeros(n)        # evolution path for sigma
        B         = np.eye(n)          # eigenvectors of C
        D         = np.ones(n)         # sqrt of eigenvalues of C
        C         = np.eye(n)          # covariance matrix
        invsqrtC  = np.eye(n)          # C^{-1/2}
        eigeneval = 0                  # generation of last eigendecomposition

        history                  = []
        best_score_internal      = -math.inf
        best_individual          = None
        gen_best_ind             = None
        gen_best_internal        = -math.inf
        gens_without_improvement = 0
        t_start                  = time.time()

        for gen in range(1, self.generations + 1):

            # --- Sample lambda candidates: x_k = m + sigma * B * (D * z_k) ---
            arz = np.random.randn(lam, n)
            arx = m + sigma * (B @ (D * arz).T).T   # shape (lam, n)

            # --- Evaluate all candidates ---
            fitvals    = np.array([self._evaluate(arx[k]) for k in range(lam)])
            sorted_idx = np.argsort(fitvals)[::-1]   # best first

            gen_best_internal = float(fitvals[sorted_idx[0]])
            gen_best_ind      = self._to_individual(arx[sorted_idx[0]])
            gen_avg_real      = float(fitvals.mean()) * self._sign

            improved = False
            if gen_best_internal > best_score_internal + self.min_delta:
                best_score_internal      = gen_best_internal
                best_individual          = gen_best_ind
                improved                 = True
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            # best_score records the all-time best seen so far (non-decreasing),
            # matching the GA's history convention.
            running_best_real = best_score_internal * self._sign
            history.append({
                'gen':                      gen,
                'best_score':               running_best_real,
                'avg_score':                gen_avg_real,
                'sigma':                    float(sigma),
                'improved':                 improved,
                'gens_without_improvement': gens_without_improvement,
                'stop_reason':              None,
            })

            print(
                f"[GEN {gen:05}] Best: {running_best_real:.11f} | "
                f"Avg: {gen_avg_real:.11f} | σ: {sigma:.6f}"
            )

            if self._on_generation is not None:
                self._on_generation(gen, running_best_real, gen_avg_real, best_individual)

            # --- CMA-ES update ---
            m_old = m.copy()

            # New mean: weighted centroid of top-mu candidates
            m = sum(weights[i] * arx[sorted_idx[i]] for i in range(mu))

            # Evolution path ps (controls step-size adaptation)
            ps = (
                (1 - cs) * ps
                + math.sqrt(cs * (2 - cs) * mueff)
                * (invsqrtC @ (m - m_old)) / sigma
            )

            # Heaviside indicator: suppress pc rank-one update when ps is too large
            hsig = (
                1.0
                if (np.linalg.norm(ps)
                    / math.sqrt(max(1e-300, 1 - (1 - cs) ** (2 * gen)))
                    / chiN) < (1.4 + 2 / (n + 1))
                else 0.0
            )

            # Evolution path pc (controls rank-one C update)
            pc = (
                (1 - cc) * pc
                + hsig * math.sqrt(cc * (2 - cc) * mueff)
                * (m - m_old) / sigma
            )

            # Deviations of selected points from old mean (rank-mu material)
            artmp = (arx[sorted_idx[:mu]] - m_old) / sigma   # shape (mu, n)

            # Full covariance matrix update: rank-one + rank-mu
            C = (
                (1 - c1 - cmu) * C
                + c1 * (
                    np.outer(pc, pc)
                    + (1 - hsig) * cc * (2 - cc) * C
                )
                + cmu * sum(
                    weights[i] * np.outer(artmp[i], artmp[i])
                    for i in range(mu)
                )
            )

            # Step-size adaptation (cumulative step-size control)
            sigma *= math.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))

            # Eigendecomposition — amortised over multiple generations
            if gen - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = gen
                C         = np.triu(C) + np.triu(C, 1).T    # enforce symmetry
                D2, B     = np.linalg.eigh(C)
                D         = np.sqrt(np.maximum(D2, 1e-20))
                invsqrtC  = B @ np.diag(1.0 / D) @ B.T

            # --- Stopping conditions ---
            stop_reason = None

            if self.patience is not None and gens_without_improvement >= self.patience:
                stop_reason = 'patience'

            elif sigma * float(D.max()) < self.tolx:
                stop_reason = 'tolx'

            else:
                win = min(len(history), 10 + 30 * n)
                if win >= 10:
                    recent = [h['best_score'] * self._sign for h in history[-win:]]
                    if max(recent) - min(recent) < self.tolfun:
                        stop_reason = 'tolfun'

            if stop_reason:
                history[-1]['stop_reason'] = stop_reason
                break

        elapsed = time.time() - t_start

        # Fallback if fitness was constant throughout (best_individual never set)
        if best_individual is None:
            best_individual     = gen_best_ind
            best_score_internal = gen_best_internal

        best_score_real = best_score_internal * self._sign

        if self.log_path:
            self._write_log(
                best_individual, best_score_real, history, elapsed,
                lam=lam, mu=mu, mueff=mueff,
                cc=cc, cs=cs, c1=c1, cmu=cmu, damps=damps,
            )

        return best_individual, best_score_real, history

    def _write_log(
        self,
        best_individual: dict,
        best_score: float,
        history: list,
        elapsed: float,
        **cma_params,
    ) -> None:
        convergence_gen = next(
            (h['gen'] for h in reversed(history) if h['improved']),
            history[-1]['gen'] if history else 0,
        )
        log = {
            'run': {
                'timestamp':       datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'type':            'cmaes',
            },
            'config': {
                'sigma0':          self.sigma0,
                'generations_max': self.generations,
                'generations_run': len(history),
                'patience':        self.patience,
                'min_delta':       self.min_delta,
                'mode':            self._mode,
                'tolx':            self.tolx,
                'tolfun':          self.tolfun,
                'lambda':          cma_params['lam'],
                'mu':              cma_params['mu'],
                'mueff':           round(cma_params['mueff'], 4),
                'cc':              round(cma_params['cc'], 6),
                'cs':              round(cma_params['cs'], 6),
                'c1':              round(cma_params['c1'], 6),
                'cmu':             round(cma_params['cmu'], 6),
                'damps':           round(cma_params['damps'], 6),
            },
            'genes': self.genes.describe(),
            'result': {
                'best_score':      best_score,
                'best_individual': best_individual,
                'sigma_final':     history[-1]['sigma'] if history else self.sigma0,
                'convergence_gen': convergence_gen,
                'stop_reason':     history[-1]['stop_reason'] if history else None,
            },
            'history': history,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        print(f"[LOG] Written to {self.log_path}")
