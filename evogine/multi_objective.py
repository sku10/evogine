import random
import multiprocessing as mp
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all
from .genes import GeneBuilder
from .operators import CrossoverStrategy, UniformCrossover


class MultiObjectiveGA:
    """
    Multi-objective genetic algorithm supporting NSGA-II and NSGA-III.

    Fitness function must return a list/tuple of floats — one per objective.
    Each objective can be independently maximized or minimized.

    Returns a Pareto front: a list of non-dominated solutions.

    Args:
        fitness_function: fn(dict) -> list[float]  (one value per objective)
        n_objectives:     Number of objectives (must match length of fitness output).
        objectives:       List of 'maximize'/'minimize' per objective.
                          Default: all 'maximize'.
        algorithm:        'nsga2' (default) or 'nsga3'.
        reference_point_divisions: For NSGA-III, number of divisions on each
                          objective axis for Das-Dennis reference points.
                          Defaults to max(1, 12 // n_objectives).
        reference_points: For NSGA-III, user-supplied reference points
                          (list of lists). Overrides reference_point_divisions.

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
        algorithm: str = 'nsga2',
        reference_point_divisions: Optional[int] = None,
        reference_points: Optional[list] = None,
    ):
        if algorithm not in ('nsga2', 'nsga3'):
            raise ValueError(f"algorithm must be 'nsga2' or 'nsga3', got {algorithm!r}")
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
        self._algorithm = algorithm

        if algorithm == 'nsga3':
            if reference_points is not None:
                self._ref_points = [list(rp) for rp in reference_points]
            else:
                divs = reference_point_divisions if reference_point_divisions is not None else max(1, 12 // n_objectives)
                self._ref_points = self._generate_reference_points(n_objectives, divs)
        else:
            self._ref_points = []

    # ------------------------------------------------------------------
    # NSGA-III reference point utilities
    # ------------------------------------------------------------------

    def _generate_reference_points(self, n_obj: int, divisions: int) -> list[list[float]]:
        """Das-Dennis simplex lattice reference points."""
        def _recursive(n, left, depth, current):
            if depth == n - 1:
                result.append(current + [left / divisions])
                return
            for i in range(left + 1):
                _recursive(n, left - i, depth + 1, current + [i / divisions])
        result = []
        _recursive(n_obj, divisions, 0, [])
        return result

    def _associate_ref_points(
        self,
        front_indices: list[int],
        scored: list[tuple[dict, list[float]]],
        already_selected_indices: list[int],
    ) -> dict[int, list[int]]:
        """Associate each individual in front_indices to nearest reference point."""
        all_indices = list(already_selected_indices) + list(front_indices)
        all_scores = [scored[i][1] for i in all_indices]
        n_obj = self.n_objectives
        ideal = [max(s[j] for s in all_scores) for j in range(n_obj)]

        def translate(s):
            return [ideal[j] - s[j] for j in range(n_obj)]

        def perp_distance(point, ref):
            ref_norm_sq = sum(r * r for r in ref)
            if ref_norm_sq < 1e-300:
                return sum(p * p for p in point) ** 0.5
            dot = sum(point[j] * ref[j] for j in range(n_obj))
            proj_sq = dot * dot / ref_norm_sq
            dist_sq = max(0.0, sum(p * p for p in point) - proj_sq)
            return dist_sq ** 0.5

        assoc: dict[int, list[int]] = {r: [] for r in range(len(self._ref_points))}
        for idx in front_indices:
            t = translate(scored[idx][1])
            best_ref = min(
                range(len(self._ref_points)),
                key=lambda r: perp_distance(t, self._ref_points[r])
            )
            assoc[best_ref].append(idx)
        return assoc

    def _nsga3_niching_select(
        self,
        last_front: list[int],
        already_selected: list[int],
        needed: int,
        scored: list[tuple[dict, list[float]]],
    ) -> list[int]:
        """Select `needed` individuals from last_front using NSGA-III niching."""
        all_assoc = self._associate_ref_points(last_front, scored, already_selected)
        sel_assoc = self._associate_ref_points([], scored, already_selected)
        niche_count: dict[int, int] = {
            r: len(sel_assoc.get(r, [])) for r in range(len(self._ref_points))
        }
        remaining = list(last_front)
        selected: list[int] = []
        while len(selected) < needed and remaining:
            min_count = min(niche_count.values())
            rho_min = [r for r, c in niche_count.items() if c == min_count]
            chosen_ref = random.choice(rho_min)
            candidates = [i for i in remaining if i in all_assoc.get(chosen_ref, [])]
            if candidates:
                pick = random.choice(candidates)
                selected.append(pick)
                remaining.remove(pick)
                niche_count[chosen_ref] = niche_count.get(chosen_ref, 0) + 1
            else:
                niche_count[chosen_ref] = len(already_selected) + len(selected) + 1
        return selected

    # ------------------------------------------------------------------
    # Core NSGA-II utilities (also used by NSGA-III for fronts)
    # ------------------------------------------------------------------

    def _evaluate(self, population: list[dict]) -> list[list[float]]:
        """Evaluate population. Returns sign-adjusted score vectors."""
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
        """NSGA-II non-dominated sorting. Returns fronts of indices."""
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

    def _survival_select(
        self, scored: list[tuple[dict, list[float]]]
    ) -> list[dict]:
        """Select next generation using NSGA-II or NSGA-III depending on algorithm."""
        fronts = self._non_dominated_sort(scored)
        next_gen_indices: list[int] = []

        for front in fronts:
            if len(next_gen_indices) + len(front) <= self.population_size:
                next_gen_indices.extend(front)
            else:
                needed = self.population_size - len(next_gen_indices)
                if self._algorithm == 'nsga3':
                    chosen = self._nsga3_niching_select(front, next_gen_indices, needed, scored)
                else:
                    distances = self._crowding_distance(front, scored)
                    sorted_front = sorted(front, key=lambda i: -distances[i])
                    chosen = sorted_front[:needed]
                next_gen_indices.extend(chosen)
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

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> tuple[list[dict], list[dict]]:
        """
        Run the multi-objective genetic algorithm.

        Returns:
            pareto_front: Non-dominated solutions. Each entry:
                          {'individual': dict, 'scores': list[float]}
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

            # Generate offspring via crowding-tournament selection
            offspring = []
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

            combined = [ind for ind, _ in scored] + offspring
            combined_vecs = self._evaluate(combined)
            combined_scored = list(zip(combined, combined_vecs))
            population = self._survival_select(combined_scored)

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
                'algorithm': self._algorithm,
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
