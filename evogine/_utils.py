from typing import Optional


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
