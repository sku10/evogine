import random
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all
from .genes import GeneBuilder


class MAPElites:
    """
    MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) optimizer.

    Maintains a quality-diversity archive: a discretized behavior grid where
    each cell stores the best individual found for that region of behavior space.
    Unlike single-objective optimizers, MAP-Elites simultaneously discovers
    high-quality solutions across the full range of user-defined behaviors.

    Args:
        gene_builder:       GeneBuilder (any gene types).
        fitness_function:   Callable (dict -> float).
        behavior_fn:        Callable (dict -> tuple). Maps an individual to behavior
                            coordinates. Values should be in [0, 1] for each dimension.
                            Example: lambda ind: (ind['x'] / 10, ind['y'] / 10)
        grid_shape:         Tuple of ints defining bins per behavior dimension.
                            Example: (20, 20) for a 2-D 20×20 grid.
        initial_population: Number of random individuals used to seed the archive
                            (default 200).
        generations:        Evaluation budget after the seeding phase (default 1000).
        mutation_rate:      Mutation rate for archive mutations (default 0.1).
        mode:               'maximize' (default) or 'minimize'.
        seed:               Random seed.
        log_path:           Path for JSON log.
        on_generation:      Callback fn(gen, archive_size, best_score, coverage).

    Returns from run():
        (archive, history)
        archive: dict keyed by behavior grid cell tuple, each value:
            {'individual': dict, 'score': float, 'behavior': tuple}
        history: list of dicts with keys: gen, archive_size, best_score, coverage
    """

    def __init__(
        self,
        gene_builder: GeneBuilder,
        fitness_function: Callable[[dict], float],
        behavior_fn: Callable[[dict], tuple],
        grid_shape: tuple,
        initial_population: int = 200,
        generations: int = 1000,
        mutation_rate: float = 0.1,
        mode: str = 'maximize',
        seed: Optional[int] = None,
        log_path: Optional[str] = None,
        on_generation: Optional[Callable] = None,
    ):
        if mode not in ('maximize', 'minimize'):
            raise ValueError("mode must be 'maximize' or 'minimize'")
        if not grid_shape or any(d < 1 for d in grid_shape):
            raise ValueError("grid_shape must be a tuple of positive integers")

        self.genes = gene_builder
        self.fitness_function = fitness_function
        self.behavior_fn = behavior_fn
        self.grid_shape = tuple(grid_shape)
        self.initial_population = initial_population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._mode = mode
        self._sign = 1.0 if mode == 'maximize' else -1.0
        self._seed = seed
        self.log_path = log_path
        self._on_generation = on_generation
        self._total_cells = 1
        for d in self.grid_shape:
            self._total_cells *= d

    def _discretize(self, behavior: tuple) -> Optional[tuple]:
        """
        Map raw behavior coordinates to a grid cell index tuple.
        Returns None if behavior has wrong length.
        Coordinates are clamped to [0,1] then mapped to [0, dim-1].
        """
        if len(behavior) != len(self.grid_shape):
            return None
        cell = []
        for b, dim in zip(behavior, self.grid_shape):
            b_clamped = max(0.0, min(1.0, float(b)))
            idx = min(dim - 1, int(b_clamped * dim))
            cell.append(idx)
        return tuple(cell)

    def _evaluate(self, ind: dict) -> float:
        """Evaluate individual. Returns sign-adjusted score (internal always-maximize)."""
        return self._sign * self.fitness_function(ind)

    def _try_add(self, archive: dict, ind: dict, score_internal: float) -> bool:
        """
        Compute behavior cell and attempt to insert into archive.
        Returns True if inserted (cell was empty or score improved).
        """
        behavior_raw = self.behavior_fn(ind)
        cell = self._discretize(behavior_raw)
        if cell is None:
            return False
        if cell not in archive or score_internal > archive[cell]['_internal']:
            archive[cell] = {
                'individual': ind,
                'score': score_internal * self._sign,
                'behavior': behavior_raw,
                '_internal': score_internal,
            }
            return True
        return False

    def run(self) -> tuple[dict, list]:
        """
        Run MAP-Elites.

        Returns:
            archive: dict of grid_cell -> {'individual', 'score', 'behavior'}
            history: list of per-generation dicts
        """
        _seed_all(self._seed)
        t_start = time.time()

        archive: dict = {}
        history: list[dict] = []

        # Seeding phase
        for _ in range(self.initial_population):
            ind = self.genes.sample()
            score = self._evaluate(ind)
            self._try_add(archive, ind, score)

        best_internal = max((v['_internal'] for v in archive.values()), default=float('-inf'))
        history.append({
            'gen': 0,
            'archive_size': len(archive),
            'best_score': best_internal * self._sign,
            'coverage': round(len(archive) / self._total_cells, 6),
        })

        # Main loop
        for gen in range(1, self.generations + 1):
            if archive:
                cells = list(archive.keys())
                parent_cell = random.choice(cells)
                parent = archive[parent_cell]['individual'].copy()
                child = self.genes.mutate(parent, self.mutation_rate)
            else:
                child = self.genes.sample()

            score = self._evaluate(child)
            self._try_add(archive, child, score)

            best_internal = max(v['_internal'] for v in archive.values())
            coverage = len(archive) / self._total_cells

            history.append({
                'gen': gen,
                'archive_size': len(archive),
                'best_score': best_internal * self._sign,
                'coverage': round(coverage, 6),
            })

            if self._on_generation is not None:
                self._on_generation(gen, len(archive), best_internal * self._sign, coverage)

            if gen % max(1, self.generations // 10) == 0:
                print(
                    f"[GEN {gen:05}] Archive: {len(archive)}/{self._total_cells} cells "
                    f"({coverage*100:.1f}%) | Best: {best_internal * self._sign:.8f}"
                )

        elapsed = time.time() - t_start

        # Strip internal key before returning
        public_archive = {
            cell: {k: v for k, v in entry.items() if k != '_internal'}
            for cell, entry in archive.items()
        }

        if self.log_path:
            self._write_log(public_archive, history, elapsed)

        return public_archive, history

    def _write_log(self, archive: dict, history: list, elapsed: float) -> None:
        serializable_archive = {str(cell): entry for cell, entry in archive.items()}
        log = {
            'run': {
                'timestamp':       datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'type':            'map_elites',
            },
            'config': {
                'grid_shape':         list(self.grid_shape),
                'total_cells':        self._total_cells,
                'initial_population': self.initial_population,
                'generations':        self.generations,
                'mutation_rate':      self.mutation_rate,
                'mode':               self._mode,
            },
            'genes': self.genes.describe(),
            'result': {
                'archive_size': len(archive),
                'coverage':     round(len(archive) / self._total_cells, 6),
                'best_score':   max(
                    (e['score'] * self._sign for e in archive.values()), default=None
                ),
                'archive': serializable_archive,
            },
            'history': history,
        }
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, default=str)
        print(f"[LOG] Written to {self.log_path}")
