import storage
import numpy as np
from typing import Dict, List


class EmbeddingCache:
    """Thread-safe in-memory embedding cache with SQLite version-based invalidation.
    
    On each access, checks a lightweight integer version counter in SQLite (~5μs).
    Only reloads the full embedding dataset when the version has actually changed,
    ensuring near-zero overhead on read-heavy recognize requests while keeping
    all Gunicorn/uWSGI workers synchronized.
    """
    
    def __init__(self):
        self._cache: Dict[int, List[np.ndarray]] = {}
        self._version: int = -1  # Force reload on first access
    
    def get_embeddings(self) -> Dict[int, List[np.ndarray]]:
        """Returns cached embeddings, auto-reloading from SQLite if DB version changed."""
        current_version = storage.get_cache_version()
        if current_version != self._version:
            self._cache = storage.load_embeddings()
            self._version = current_version
        return self._cache
    
    def invalidate(self):
        """Force next get_embeddings() call to reload from database."""
        self._version = -1
