import random
import multiprocessing as mp
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all, _resolve_workers
from .genes import FloatRange, GeneBuilder


class DEOptimizer:
    """
    Differential Evolution optimizer using the SHADE algorithm.

    The on_generation callback may return a dict of parameter overrides.
    Steerable keys: strategy, patience, min_delta.

    For FloatRange-only problems. SHADE maintains a history memory of successful
    mutation factor (F) and crossover rate (CR) values to adapt these parameters
    automatically over time.

    Strategies:
        'current_to_best': v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
        'rand1':           v = x_r1 + F*(x_r2 - x_r3)

    Optional L-SHADE: set linear_pop_reduction=True to linearly shrink the
    population from population_size down to 4 over all generations.

    Args:
        gene_builder:         GeneBuilder with FloatRange genes only (min 2).
        fitness_function:     Callable (dict -> float).
        population_size:      Initial population size (default 50).
        generations:          Max generations (default 200).
        strategy:             'current_to_best' (default) or 'rand1'.
        memory_size:          SHADE history memory size H (default 6).
        linear_pop_reduction: L-SHADE linear population reduction (default False).
        patience:             Early stopping patience.
        min_delta:            Minimum improvement to reset patience counter.
        mode:                 'maximize' (default) or 'minimize'.
        seed:                 Random seed.
        log_path:             Path for JSON log output.
        on_generation:        Callback fn(gen, best_score, avg_score, best_individual).

    Returns from run():
        (best_individual, best_score, history)
        history entries: gen, best_score, avg_score, F_mean, CR_mean,
                         improved, gens_without_improvement, stop_reason, pop_size
    """

    _STEERABLE_KEYS = {'strategy', 'patience', 'min_delta'}

    def _apply_steering(self, result):
        if not isinstance(result, dict):
            return
        for key, value in result.items():
            if key == 'strategy' and value not in ('current_to_best', 'rand1'):
                continue
            if key in self._STEERABLE_KEYS:
                setattr(self, key, value)

    def __init__(
        self,
        gene_builder: GeneBuilder,
        fitness_function: Callable[[dict], float],
        population_size: int = 50,
        generations: int = 200,
        strategy: str = 'current_to_best',
        memory_size: int = 6,
        linear_pop_reduction: bool = False,
        patience: Optional[int] = None,
        min_delta: float = 1e-9,
        mode: str = 'maximize',
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        on_generation: Optional[Callable] = None,
        use_multiprocessing: bool = False,
        workers: Optional[int] = None,
    ):
        if mode not in ('maximize', 'minimize'):
            raise ValueError("mode must be 'maximize' or 'minimize'")
        if strategy not in ('current_to_best', 'rand1'):
            raise ValueError("strategy must be 'current_to_best' or 'rand1'")
        for name, spec in gene_builder.specs.items():
            if not isinstance(spec, FloatRange):
                raise ValueError(
                    f"DEOptimizer only supports FloatRange genes. "
                    f"Gene '{name}' is {type(spec).__name__}. "
                    f"Use GeneticAlgorithm for mixed gene types."
                )
        n = len(gene_builder.order)
        if n < 2:
            raise ValueError(
                "DEOptimizer requires at least 2 genes. "
                "For 1-dimensional problems use GeneticAlgorithm."
            )

        self.genes = gene_builder
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.strategy = strategy
        self.memory_size = memory_size
        self.linear_pop_reduction = linear_pop_reduction
        self.patience = patience
        self.min_delta = min_delta
        self._mode = mode
        self._sign = 1.0 if mode == 'maximize' else -1.0
        self._seed = seed
        self.log_path = log_path
        self._on_generation = on_generation
        self.use_multiprocessing = use_multiprocessing
        self._workers = workers
        self._pool = None
        self._n = n
        self._min_population = 4

    def _diagnose_generation(self, F_mean, CR_mean, improved, gens_without_improvement):
        if F_mean < 0.1:
            return 'F_collapsed', 'Step size too small; increase population_size or try rand1 strategy'
        if CR_mean > 0.95:
            return 'CR_saturated', 'Nearly full crossover; problem may be separable — try rand1'
        if improved:
            return 'improving', 'No changes needed'
        if gens_without_improvement > 10:
            return 'stagnating', 'Consider increasing population_size'
        return 'stable', 'No changes needed'

    def _to_individual(self, x: list[float]) -> dict:
        """Convert normalized [0,1]^n vector to gene dict, clamped to bounds."""
        ind = {}
        for i, name in enumerate(self.genes.order):
            spec = self.genes.specs[name]
            v = max(0.0, min(1.0, x[i]))
            ind[name] = spec.low + v * (spec.high - spec.low)
        return ind

    def _evaluate(self, x: list[float]) -> float:
        """Evaluate a normalized vector. Returns internal score (always-maximize sign)."""
        return self._sign * self.fitness_function(self._to_individual(x))

    @staticmethod
    def _lehmer_mean(values: list[float]) -> float:
        """Lehmer mean (power-2 / power-1) — recommended for F in SHADE."""
        if not values:
            return 0.5
        s1 = sum(v * v for v in values)
        s0 = sum(v for v in values)
        return s1 / s0 if s0 > 0 else 0.5

    def run(self) -> tuple[dict, float, list]:
        """
        Run the SHADE differential evolution optimizer.

        Returns:
            (best_individual, best_score, history)
        """
        _seed_all(self._seed)
        t_start = time.time()
        n = self._n

        pop_size = self.population_size
        pop = [[random.random() for _ in range(n)] for _ in range(pop_size)]

        n_workers = _resolve_workers(self._workers, self.use_multiprocessing)
        self._pool = mp.Pool(n_workers) if n_workers is not None else None

        if self._pool is not None:
            individuals = [self._to_individual(x) for x in pop]
            raw_fits = list(self._pool.map(self.fitness_function, individuals))
            fit = [self._sign * f for f in raw_fits]
        else:
            fit = [self._evaluate(x) for x in pop]

        # SHADE memory
        M_F  = [0.5] * self.memory_size
        M_CR = [0.5] * self.memory_size
        mem_ptr = 0

        best_score_internal = max(fit)
        best_idx = fit.index(best_score_internal)
        best_individual = self._to_individual(pop[best_idx])

        history: list[dict] = []
        gens_without_improvement = 0

        for gen in range(1, self.generations + 1):
            # L-SHADE linear population reduction
            if self.linear_pop_reduction:
                effective_size = max(
                    self._min_population,
                    round(
                        self.population_size
                        - (self.population_size - self._min_population)
                        * (gen / self.generations)
                    ),
                )
                if effective_size < len(pop):
                    order = sorted(range(len(pop)), key=lambda i: -fit[i])
                    pop      = [pop[i] for i in order[:effective_size]]
                    fit      = [fit[i] for i in order[:effective_size]]
                    pop_size = effective_size
            else:
                pop_size = len(pop)

            S_F:  list[float] = []
            S_CR: list[float] = []
            new_pop = list(pop)
            new_fit = list(fit)

            best_i = fit.index(max(fit))

            trials = []
            trial_meta = []

            for i in range(pop_size):
                r_mem = random.randint(0, self.memory_size - 1)
                F  = min(1.0, max(0.0, random.gauss(M_F[r_mem], 0.1)))
                CR = min(1.0, max(0.0, random.gauss(M_CR[r_mem], 0.1)))

                candidates = [j for j in range(pop_size) if j != i]

                if self.strategy == 'current_to_best':
                    non_best = [j for j in candidates if j != best_i]
                    if len(non_best) >= 2:
                        r1, r2 = random.sample(non_best, 2)
                    elif non_best:
                        r1 = r2 = non_best[0]
                    else:
                        r1 = r2 = candidates[0] if candidates else i
                    v = [
                        pop[i][d] + F * (pop[best_i][d] - pop[i][d]) + F * (pop[r1][d] - pop[r2][d])
                        for d in range(n)
                    ]
                else:  # rand1
                    if len(candidates) >= 3:
                        r1, r2, r3 = random.sample(candidates, 3)
                    else:
                        r1, r2, r3 = candidates[0], candidates[0], candidates[0]
                    v = [
                        pop[r1][d] + F * (pop[r2][d] - pop[r3][d])
                        for d in range(n)
                    ]

                v = [max(0.0, min(1.0, vd)) for vd in v]

                j_rand = random.randint(0, n - 1)
                trial = [
                    v[d] if (random.random() < CR or d == j_rand) else pop[i][d]
                    for d in range(n)
                ]

                trials.append(trial)
                trial_meta.append((i, F, CR))

            if self._pool is not None:
                trial_individuals = [self._to_individual(t) for t in trials]
                raw_fits = list(self._pool.map(self.fitness_function, trial_individuals))
                trial_fits = [self._sign * f for f in raw_fits]
            else:
                trial_fits = [self._evaluate(t) for t in trials]

            for (i, F, CR), trial, trial_fit in zip(trial_meta, trials, trial_fits):
                if trial_fit >= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    if trial_fit > fit[i]:
                        S_F.append(F)
                        S_CR.append(CR)

            pop = new_pop
            fit = new_fit

            if S_F:
                M_F[mem_ptr]  = self._lehmer_mean(S_F)
                M_CR[mem_ptr] = sum(S_CR) / len(S_CR)
                mem_ptr = (mem_ptr + 1) % self.memory_size

            gen_best_internal = max(fit)
            gen_avg_real      = (sum(fit) / len(fit)) * self._sign

            improved = gen_best_internal > best_score_internal + self.min_delta
            if improved:
                best_score_internal = gen_best_internal
                best_idx = fit.index(gen_best_internal)
                best_individual = self._to_individual(pop[best_idx])
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1

            running_best_real = best_score_internal * self._sign
            F_mean  = sum(M_F)  / len(M_F)
            CR_mean = sum(M_CR) / len(M_CR)

            diagnosis, recommendation = self._diagnose_generation(
                F_mean, CR_mean, improved, gens_without_improvement,
            )

            history.append({
                'gen':                      gen,
                'best_score':               running_best_real,
                'avg_score':                gen_avg_real,
                'F_mean':                   round(F_mean, 6),
                'CR_mean':                  round(CR_mean, 6),
                'improved':                 improved,
                'gens_without_improvement': gens_without_improvement,
                'stop_reason':              None,
                'pop_size':                 pop_size,
                'diagnosis':                diagnosis,
                'recommendation':           recommendation,
            })

            print(
                f"[GEN {gen:05}] Best: {running_best_real:.11f} | "
                f"Avg: {gen_avg_real:.11f} | F={F_mean:.4f} CR={CR_mean:.4f}"
            )

            if self._on_generation is not None:
                result = self._on_generation(gen, running_best_real, gen_avg_real, best_individual)
                self._apply_steering(result)

            if self.patience is not None and gens_without_improvement >= self.patience:
                history[-1]['stop_reason'] = 'patience'
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

        elapsed = time.time() - t_start
        best_score_real = best_score_internal * self._sign

        if self.log_path:
            self._write_log(best_individual, best_score_real, history, elapsed)

        return best_individual, best_score_real, history

    def _write_log(
        self,
        best_individual: dict,
        best_score: float,
        history: list,
        elapsed: float,
    ) -> None:
        convergence_gen = next(
            (h['gen'] for h in reversed(history) if h['improved']),
            history[-1]['gen'] if history else 0,
        )
        log = {
            'run': {
                'timestamp':       datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'type':            'de',
            },
            'config': {
                'population_size':      self.population_size,
                'generations_max':      self.generations,
                'generations_run':      len(history),
                'strategy':             self.strategy,
                'memory_size':          self.memory_size,
                'linear_pop_reduction': self.linear_pop_reduction,
                'patience':             self.patience,
                'min_delta':            self.min_delta,
                'mode':                 self._mode,
                'use_multiprocessing':  self.use_multiprocessing,
                'workers':              self._workers,
            },
            'genes': self.genes.describe(),
            'result': {
                'best_score':      best_score,
                'best_individual': best_individual,
                'convergence_gen': convergence_gen,
                'stop_reason':     history[-1]['stop_reason'] if history else None,
            },
            'history': history,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2)
        print(f"[LOG] Written to {self.log_path}")
