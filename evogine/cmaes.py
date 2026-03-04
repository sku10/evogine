import math
import json
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from ._utils import _seed_all
from .genes import FloatRange, GeneBuilder


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
        fitness_function:  Callable (dict -> float).
        sigma0:            Initial step size, relative to gene range (default 0.3).
        generations:       Maximum number of generations (default 200).
        popsize:           Population size lambda. Default: 4 + floor(3*ln(n)).
        patience:          Stop after this many generations without improvement.
        min_delta:         Minimum improvement to reset the patience counter.
        mode:              'maximize' (default) or 'minimize'.
        seed:              Random seed for reproducibility.
        log_path:          Path to write a JSON log.
        tolx:              Stop when sigma * max(eigenvalues) < tolx.
        tolfun:            Stop when best score has not changed meaningfully.
        on_generation:     Callback fn(gen, best_score, avg_score, best_individual).

    Returns from run():
        (best_individual, best_score, history)
        history entries: gen, best_score, avg_score, sigma, improved,
                         gens_without_improvement, stop_reason
    """

    def __init__(
        self,
        gene_builder: GeneBuilder,
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

        weights_raw = np.array(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
        )
        weights = weights_raw / weights_raw.sum()
        mueff   = 1.0 / float((weights ** 2).sum())

        cc    = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs    = (mueff + 2) / (n + mueff + 5)
        c1    = 2 / ((n + 1.3) ** 2 + mueff)
        cmu   = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chiN  = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        m         = np.full(n, 0.5)
        sigma     = self.sigma0
        pc        = np.zeros(n)
        ps        = np.zeros(n)
        B         = np.eye(n)
        D         = np.ones(n)
        C         = np.eye(n)
        invsqrtC  = np.eye(n)
        eigeneval = 0

        history                  = []
        best_score_internal      = -math.inf
        best_individual          = None
        gen_best_ind             = None
        gen_best_internal        = -math.inf
        gens_without_improvement = 0
        t_start                  = time.time()

        for gen in range(1, self.generations + 1):

            arz = np.random.randn(lam, n)
            arx = m + sigma * (B @ (D * arz).T).T

            fitvals    = np.array([self._evaluate(arx[k]) for k in range(lam)])
            sorted_idx = np.argsort(fitvals)[::-1]

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

            m_old = m.copy()
            m = sum(weights[i] * arx[sorted_idx[i]] for i in range(mu))

            ps = (
                (1 - cs) * ps
                + math.sqrt(cs * (2 - cs) * mueff)
                * (invsqrtC @ (m - m_old)) / sigma
            )

            hsig = (
                1.0
                if (np.linalg.norm(ps)
                    / math.sqrt(max(1e-300, 1 - (1 - cs) ** (2 * gen)))
                    / chiN) < (1.4 + 2 / (n + 1))
                else 0.0
            )

            pc = (
                (1 - cc) * pc
                + hsig * math.sqrt(cc * (2 - cc) * mueff)
                * (m - m_old) / sigma
            )

            artmp = (arx[sorted_idx[:mu]] - m_old) / sigma

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

            sigma *= math.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))

            if gen - eigeneval > lam / (c1 + cmu) / n / 10:
                eigeneval = gen
                C         = np.triu(C) + np.triu(C, 1).T
                D2, B     = np.linalg.eigh(C)
                D         = np.sqrt(np.maximum(D2, 1e-20))
                invsqrtC  = B @ np.diag(1.0 / D) @ B.T

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
