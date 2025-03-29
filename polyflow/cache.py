import os
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from functools import wraps
from typing import Any, Callable

import polyflow


def require_cache_enabled(func: Callable) -> Callable:
    """Decorator to check if caching is enabled before calling the function."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not polyflow.settings.enable_cache:
            return None
        return func(self, *args, **kwargs)

    return wrapper


class CacheType(Enum):
    IN_MEMORY = "in_memory"
    SQLITE = "sqlite"


class CacheConfig:
    def __init__(self, cache_type: CacheType, max_size: int, **kwargs):
        self.cache_type = cache_type
        self.max_size = max_size
        self.kwargs = kwargs


class Cache(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size

    @abstractmethod
    def get(self, key: str) -> Any | None:
        pass

    @abstractmethod
    def insert(self, key: str, value: Any):
        pass

    @abstractmethod
    def reset(self, max_size: int | None = None):
        pass


class CacheFactory:
    @staticmethod
    def create_cache(config: CacheConfig) -> Cache:
        if config.cache_type == CacheType.IN_MEMORY:
            return InMemoryCache(max_size=config.max_size)
        elif config.cache_type == CacheType.SQLITE:
            cache_dir = config.kwargs.get("cache_dir", "~/.polyflow/cache")
            if not isinstance(cache_dir, str):
                raise ValueError("cache_dir must be a string")
            return SQLiteCache(max_size=config.max_size, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported cache type: {config.cache_type}")

    @staticmethod
    def create_default_cache(max_size: int = 1024) -> Cache:
        return CacheFactory.create_cache(CacheConfig(CacheType.IN_MEMORY, max_size))


class SQLiteCache(Cache):
    def __init__(self, max_size: int, cache_dir=os.path.expanduser("~/.polyflow/cache")):
        super().__init__(max_size)
        self.db_path = os.path.join(cache_dir, "lotus_cache.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    last_accessed INTEGER
                )
            """)

    def _get_time(self):
        return int(time.time())

    @require_cache_enabled
    def get(self, key: str) -> Any | None:
        with self.conn:
            cursor = self.conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
            result = cursor.fetchone()
            if result:
                polyflow.logger.debug(f"Cache hit for {key}")
                value = pickle.loads(result[0])
                self.conn.execute(
                    "UPDATE cache SET last_accessed = ? WHERE key = ?",
                    (
                        self._get_time(),
                        key,
                    ),
                )
                return value
        return None

    @require_cache_enabled
    def insert(self, key: str, value: Any):
        pickled_value = pickle.dumps(value)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, last_accessed) 
                VALUES (?, ?, ?)
            """,
                (key, pickled_value, self._get_time()),
            )
            self._enforce_size_limit()

    def _enforce_size_limit(self):
        with self.conn:
            count = self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count > self.max_size:
                num_to_delete = count - self.max_size
                self.conn.execute(
                    """
                    DELETE FROM cache WHERE key IN (
                        SELECT key FROM cache
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                """,
                    (num_to_delete,),
                )

    def reset(self, max_size: int | None = None):
        with self.conn:
            self.conn.execute("DELETE FROM cache")
        if max_size is not None:
            self.max_size = max_size

    def __del__(self):
        self.conn.close()


class InMemoryCache(Cache):
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.cache: OrderedDict[str, Any] = OrderedDict()

    @require_cache_enabled
    def get(self, key: str) -> Any | None:
        if key in self.cache:
            polyflow.logger.debug(f"Cache hit for {key}")

        return self.cache.get(key)

    @require_cache_enabled
    def insert(self, key: str, value: Any):
        self.cache[key] = value

        # LRU eviction
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def reset(self, max_size: int | None = None):
        self.cache.clear()
        if max_size is not None:
            self.max_size = max_size
