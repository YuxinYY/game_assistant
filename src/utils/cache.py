"""
Simple in-memory LRU cache for LLM responses and retrieval results.
Keyed by (query, filters) hash. TTL configurable in config.yaml.
"""

import hashlib
import json
import time
from typing import Any, Optional


class Cache:
    def __init__(self, ttl_seconds: int = 3600, enabled: bool = True):
        self.ttl = ttl_seconds
        self.enabled = enabled
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any):
        if self.enabled:
            self._store[key] = (value, time.time())

    def make_key(self, *args, **kwargs) -> str:
        raw = json.dumps({"args": args, "kwargs": kwargs}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def clear(self):
        self._store.clear()


_default_cache: Optional[Cache] = None


def get_cache(config: dict | None = None) -> Cache:
    global _default_cache
    if _default_cache is None:
        cfg = (config or {}).get("cache", {})
        _default_cache = Cache(
            ttl_seconds=cfg.get("ttl_seconds", 3600),
            enabled=cfg.get("enable", True),
        )
    return _default_cache
