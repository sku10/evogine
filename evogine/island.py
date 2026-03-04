import random
import multiprocessing as mp
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all, _resolve_workers
from .genes import GeneBuilder
from .operators import SelectionStrategy, CrossoverStrategy, RouletteSelection, UniformCrossover


class IslandModel:
    """
    Multiple independent GA sub-populations (islands) with periodic migration.

    The on_generation callback may return a dict of parameter overrides.
    Steerable keys: mutation_rate, crossover_rate, elitism, migration_interval, migration_size.

    Each island evolves independently. Every migration_interval generations,
    the top migration_size individuals from each island migrate to neighbouring
    islands according to the chosen topology.

    Args:
        n_islands:          Number of sub-populations (default 4).
        island_population:  Individuals per island (default 50).
        migration_interval: Migrate every N generations (default 10).
        migration_size:     Top K individuals to copy per migration (default 2).
        topology:           'ring' (default), 'fully_connected', or 'star'.

    Returns from run():
        best_individual: Best solution found across all islands.
        best_score:      Its fitness score.
        history:         Per-generation dicts with island_bests list.
    """

    _STEERABLE_KEYS = {'mutation_rate', 'crossover_rate', 'elitism', 'migration_interval', 'migration_size'}

    def _apply_steering(self, result):
        if not isinstance(result, dict):
            return
        for key, value in result.items():
            if key in self._STEERABLE_KEYS:
                setattr(self, key, value)

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
        workers: Optional[int] = None,
        seed: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 1e-6,
        selection: Optional[SelectionStrategy] = None,
        crossover: Optional[CrossoverStrategy] = None,
        mode: str = 'maximize',
        log_path: Optional[str] = None,
        on_generation: Optional[Callable] = None,
        topology: str = 'ring',
    ):
        if mode not in ('maximize', 'minimize'):
            raise ValueError(f"mode must be 'maximize' or 'minimize', got {mode!r}")
        if topology not in ('ring', 'fully_connected', 'star'):
            raise ValueError(
                f"topology must be 'ring', 'fully_connected', or 'star', got {topology!r}"
            )
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
        self._workers = workers
        self._seed = seed
        self.patience = patience
        self.min_delta = min_delta
        self._selection = selection or RouletteSelection()
        self._crossover = crossover or UniformCrossover()
        self._mode = mode
        self._sign = -1 if mode == 'minimize' else 1
        self.log_path = log_path
        self._on_generation = on_generation
        self._topology = topology
        self._pool = None

    def _diagnose_generation(self, island_bests, improved, gens_without_improvement):
        if len(island_bests) > 1:
            mean_best = sum(island_bests) / len(island_bests)
            if mean_best != 0 and all(
                abs(b - mean_best) / max(abs(mean_best), 1e-300) < 0.01 for b in island_bests
            ):
                return 'islands_converged', 'Increase migration_interval or try star topology'
        if improved:
            return 'improving', 'No changes needed'
        if gens_without_improvement > 10:
            return 'stagnating', 'Increase migration_size or mutation_rate'
        return 'stable', 'No changes needed'

    def _get_migration_pairs(self) -> list[tuple[int, int]]:
        """Return (source, destination) island index pairs for the current topology."""
        n = self.n_islands
        pairs = []
        if self._topology == 'ring':
            for i in range(n):
                pairs.append((i, (i + 1) % n))
        elif self._topology == 'fully_connected':
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pairs.append((i, j))
        elif self._topology == 'star':
            for i in range(1, n):
                pairs.append((i, 0))   # spoke → hub
            for i in range(1, n):
                pairs.append((0, i))   # hub → spoke
        return pairs

    def _evaluate_population(self, population: list[dict]) -> list[tuple[dict, float]]:
        if self._pool is not None:
            fitnesses = self._pool.map(self.fitness_function, population)
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

        n_workers = _resolve_workers(self._workers, self.use_multiprocessing)
        self._pool = mp.Pool(n_workers) if n_workers is not None else None
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

            running_best_real = best_score * self._sign
            real_avg = gen_avg_sc * self._sign

            diagnosis, recommendation = self._diagnose_generation(
                island_bests, improved, gens_without_improvement,
            )

            history.append({
                'gen': gen,
                'best_score': running_best_real,
                'avg_score': real_avg,
                'island_bests': island_bests,
                'improved': improved,
                'gens_without_improvement': gens_without_improvement,
                'diagnosis': diagnosis,
                'recommendation': recommendation,
            })

            print(
                f"[GEN {gen:05}] Best: {running_best_real:.8f} | "
                f"Avg: {real_avg:.8f} | Islands: {island_bests}"
            )

            if self._on_generation is not None:
                result = self._on_generation(gen, running_best_real, real_avg, best_overall)
                self._apply_steering(result)

            if self.patience is not None and gens_without_improvement >= self.patience:
                print(f"[EARLY STOP] No improvement for {self.patience} generations.")
                break

            if gen % self.migration_interval == 0 and self.n_islands > 1:
                pairs = self._get_migration_pairs()
                sorted_islands = [
                    sorted(all_island_scored[i], key=lambda x: -x[1])
                    for i in range(self.n_islands)
                ]
                pending: dict[int, list[dict]] = {i: [] for i in range(self.n_islands)}
                for src, dst in pairs:
                    migrants = [ind.copy() for ind, _ in sorted_islands[src][:self.migration_size]]
                    pending[dst].extend(migrants)
                for dst, migrants in pending.items():
                    if not migrants:
                        continue
                    target_scored = list(zip(
                        populations[dst],
                        [sc for _, sc in all_island_scored[dst]]
                    ))
                    target_scored.sort(key=lambda x: -x[1])
                    n_replace = min(len(migrants), max(0, len(target_scored) - self.elitism))
                    if n_replace > 0:
                        populations[dst] = (
                            [ind for ind, _ in target_scored[:-n_replace]]
                            + migrants[:n_replace]
                        )
                print(
                    f"[MIGRATION] Gen {gen}: "
                    f"{self.migration_size} migrants, topology={self._topology}"
                )

        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

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
                'topology': self._topology,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism': self.elitism,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'mode': self._mode,
                'use_multiprocessing': self.use_multiprocessing,
                'workers': self._workers,
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
