import random
import multiprocessing as mp
import json
import time
import os
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all
from .genes import FloatRange, IntRange, ChoiceList, GeneBuilder
from .operators import SelectionStrategy, CrossoverStrategy, RouletteSelection, UniformCrossover


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
        linear_pop_reduction: bool = False,
        min_population: int = 4,
        constraints: Optional[list] = None,
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

            restart_fraction: Fraction of population to replace on restart (default 0.3).
                  Elites are always preserved. Only non-elite slots are replaced.

            on_generation: Callback called after each generation.
                  Signature: fn(gen, best_score, avg_score, best_individual) -> None

            adaptive_mutation_min: Lower bound for adaptive mutation rate (default 0.01).
            adaptive_mutation_max: Upper bound for adaptive mutation rate (default 0.5).

            linear_pop_reduction: Linearly shrink the population from population_size
                  down to min_population over the course of all generations (L-SHADE style).

            min_population: Minimum population size when linear_pop_reduction=True (default 4).

            constraints: Optional list of callables fn(individual) -> bool.
                  An individual is feasible if all constraints return True.
                  Infeasible individuals are penalised using Deb's feasibility rules:
                  any feasible individual outranks any infeasible one; among infeasible
                  individuals, fewer violations is better.
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
        self._linear_pop_reduction = linear_pop_reduction
        self._min_population = max(2, min_population)
        self._constraints = constraints or []

    def create_individual(self) -> dict:
        return self.genes.sample()

    def mutate(self, ind: dict) -> dict:
        return self.genes.mutate(ind, self.mutation_rate)

    def _count_violations(self, ind: dict) -> int:
        """Return the number of constraints violated by ind (0 = fully feasible)."""
        if not self._constraints:
            return 0
        return sum(1 for fn in self._constraints if not fn(ind))

    def evaluate_population(self, population: list[dict]) -> list[tuple[dict, float]]:
        if self.use_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                fitnesses = pool.map(self.fitness_function, population)
        else:
            fitnesses = [self.fitness_function(ind) for ind in population]
        if self._mode == 'minimize':
            fitnesses = [-f for f in fitnesses]
        if self._constraints:
            # Deb's feasibility rules: infeasible get score = -1e18 - violations
            adjusted = []
            for ind, fit in zip(population, fitnesses):
                v = self._count_violations(ind)
                adjusted.append((ind, fit if v == 0 else (-1e18 - v)))
            return adjusted
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
            resume_from: Optional path to a checkpoint file.

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
            running_best_real = best_score * self._sign
            real_avg = gen_avg * self._sign

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

            if (
                self._checkpoint_path is not None
                and gen % self._checkpoint_every == 0
            ):
                self._save_checkpoint(
                    gen, population, best_overall, best_score,
                    gens_without_improvement, convergence_gen, history,
                )

            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                if self._checkpoint_path is not None:
                    self._save_checkpoint(
                        gen, population, best_overall, best_score,
                        gens_without_improvement, convergence_gen, history,
                    )
                break

            # Linear population reduction
            if self._linear_pop_reduction:
                effective_pop = max(
                    self._min_population,
                    round(
                        self.population_size
                        - (self.population_size - self._min_population)
                        * (gen / self.generations)
                    ),
                )
            else:
                effective_pop = self.population_size

            elite_count = min(self.elitism, effective_pop)
            next_gen = [ind for ind, _ in scored[:elite_count]]

            if restarted:
                n_inject = max(1, int(self._restart_fraction * effective_pop))
                for _ in range(n_inject):
                    next_gen.append(self.create_individual())

            while len(next_gen) < effective_pop:
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
