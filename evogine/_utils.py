import json
import multiprocessing as mp
from typing import Optional


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars."""

    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def _seed_all(seed: Optional[int]) -> None:
    """Seed random and numpy.random (if numpy is available)."""
    if seed is not None:
        import random
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass


def _resolve_workers(workers: Optional[int], use_multiprocessing: bool) -> Optional[int]:
    """Resolve workers param to a concrete pool size or None (no pool).

    Returns None (no pool) or a positive int.
    """
    if workers is not None:
        cpus = mp.cpu_count()
        if workers == 0:
            return cpus
        if workers > 0:
            return max(1, min(workers, cpus))
        # negative: cpus - abs(workers)
        return max(1, cpus + workers)
    if use_multiprocessing:
        return mp.cpu_count()
    return None
