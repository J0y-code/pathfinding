import math
from functools import lru_cache

class AiUtils:
    @staticmethod
    def fast_dist(a, b):
        """Distance euclidienne entre deux itérables (list, tuple, np.array)."""
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    @staticmethod
    @lru_cache(maxsize=200000)
    def fast_dist_cached(p1_tuple, p2_tuple):
        """
        Distance euclidienne avec cache LRU.
        Attention : les arguments doivent être des tuples (x,y,z) hashables.
        """
        dx = p1_tuple[0] - p2_tuple[0]
        dy = p1_tuple[1] - p2_tuple[1]
        dz = p1_tuple[2] - p2_tuple[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    @staticmethod
    def fast_dist_cached_from_arrays(a, b):
        """
        Wrapper pratique pour np.array ou listes → tuples pour le cache.
        """
        return AiUtils.fast_dist_cached(
            (float(a[0]), float(a[1]), float(a[2])),
            (float(b[0]), float(b[1]), float(b[2]))
        )
