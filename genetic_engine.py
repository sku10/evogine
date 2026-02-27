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
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def sample(self):
        return random.randint(self.low, self.high)

    def mutate(self, value, mutation_rate):
        if random.random() < mutation_rate:
            delta = random.choice([-1, 1])
            value = max(min(value + delta, self.high), self.low)
        return value

    def describe(self):
        return {'type': 'IntRange', 'low': self.low, 'high': self.high}


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
    ):
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

    def create_individual(self) -> dict:
        return self.genes.sample()

    def mutate(self, ind: dict) -> dict:
        return self.genes.mutate(ind, self.mutation_rate)
    
    def crossover(self, p1: dict, p2: dict) -> dict:
        child = {}
        for key in self.genes.keys():
            child[key] = p1[key] if random.random() > 0.5 else p2[key]
        return child

    def evaluate_population(self, population: list[dict]) -> list[tuple[dict, float]]:
        if self.use_multiprocessing:
            with mp.Pool(mp.cpu_count()) as pool:
                fitnesses = pool.map(self.fitness_function, population)
        else:
            fitnesses = [self.fitness_function(ind) for ind in population]
        return list(zip(population, fitnesses))

    def select_parents(self, scored: list[tuple[dict, float]]) -> tuple[dict, dict]:
        min_score = min(score for _, score in scored)
        weights = [(score - min_score + 1e-12) for _, score in scored]
        parents = random.choices(scored, weights=weights, k=2)
        return parents[0][0], parents[1][0]

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

            # Early stopping
            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

            # Elitism: carry over top N unchanged
            next_gen = [ind for ind, _ in scored[:self.elitism]]

            while len(next_gen) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1, p2 = self.select_parents(scored)
                    child = self.crossover(p1, p2)
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
