"""
缓存模块
"""

import time
import json
import hashlib
import asyncio
from typing import Any, Optional, Dict, List
from abc import ABC, abstractmethod
from collections import OrderedDict
import threading
from datetime import datetime, timedelta


class CacheItem:
    """缓存项"""

    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl if ttl else None
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def access(self) -> Any:
        """访问缓存项"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value


class BaseCache(ABC):
    """缓存基础接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        pass


class SimpleCache(BaseCache):
    """简单内存缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key in self._cache:
                item = self._cache[key]
                if item.is_expired():
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return None

                # 移到最后 (LRU)
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                return item.access()
            else:
                self._stats['misses'] += 1
                return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            ttl = ttl or self.default_ttl
            item = CacheItem(key, value, ttl)

            # 如果缓存已满，删除最旧的项
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)
                self._stats['evictions'] += 1

            self._cache[key] = item
            self._stats['sets'] += 1
            return True

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['deletes'] += 1
                return True
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            return True

    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'evictions': self._stats['evictions']
            }

    def _cleanup_expired(self):
        """清理过期项"""
        with self._lock:
            expired_keys = [
                key for key, item in self._cache.items()
                if item.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]


class RedisCache(BaseCache):
    """Redis缓存（需要redis-py）"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0",
                 key_prefix: str = "agent_cache:",
                 default_ttl: int = 3600):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self._client = None
        self._connect()

    def _connect(self):
        """连接Redis"""
        try:
            import redis
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        except ImportError:
            raise ImportError("redis-py is required for RedisCache")

    def _make_key(self, key: str) -> str:
        """生成完整的键名"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = await self._client.get(self._make_key(key))
            if value is None:
                return None
            return json.loads(value)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            ttl = ttl or self.default_ttl
            json_value = json.dumps(value, ensure_ascii=False)
            return await self._client.setex(
                self._make_key(key), ttl, json_value
            )
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            return bool(await self._client.delete(self._make_key(key)))
        except Exception:
            return False

    async def clear(self) -> bool:
        """清空缓存"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self._client.keys(pattern)
            if keys:
                return bool(await self._client.delete(*keys))
            return True
        except Exception:
            return False

    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        try:
            info = await self._client.info()
            return {
                'used_memory': info.get('used_memory', 0),
                'connected_clients': info.get('connected_clients', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'key_prefix': self.key_prefix
            }
        except Exception:
            return {}


class MultiLevelCache(BaseCache):
    """多级缓存"""

    def __init__(self, l1_cache: BaseCache, l2_cache: BaseCache):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值（先查L1，再查L2）"""
        # 先查L1缓存
        value = await self.l1_cache.get(key)
        if value is not None:
            return value

        # 再查L2缓存
        value = await self.l2_cache.get(key)
        if value is not None:
            # 回填到L1缓存
            await self.l1_cache.set(key, value)
            return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值（同时设置到L1和L2）"""
        l1_success = await self.l1_cache.set(key, value, ttl)
        l2_success = await self.l2_cache.set(key, value, ttl)
        return l1_success and l2_success

    async def delete(self, key: str) -> bool:
        """删除缓存（同时从L1和L2删除）"""
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)
        return l1_success or l2_success

    async def clear(self) -> bool:
        """清空缓存（同时清空L1和L2）"""
        l1_success = await self.l1_cache.clear()
        l2_success = await self.l2_cache.clear()
        return l1_success and l2_success

    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        l1_stats = await self.l1_cache.stats()
        l2_stats = await self.l2_cache.stats()

        return {
            'l1_cache': l1_stats,
            'l2_cache': l2_stats
        }


# 全局缓存实例
_default_cache = None


def get_default_cache() -> BaseCache:
    """获取默认缓存实例"""
    global _default_cache
    if _default_cache is None:
        cache_type = os.getenv('CACHE_TYPE', 'simple').lower()
        max_size = int(os.getenv('CACHE_MAX_SIZE', '1000'))
        default_ttl = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))

        if cache_type == 'redis':
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            _default_cache = RedisCache(redis_url=redis_url, default_ttl=default_ttl)
        else:
            _default_cache = SimpleCache(max_size=max_size, default_ttl=default_ttl)

    return _default_cache


def cache_key(*args, **kwargs) -> str:
    """生成缓存键"""
    key_parts = []
    for arg in args:
        if isinstance(arg, (list, dict)):
            key_parts.append(json.dumps(arg, sort_keys=True))
        else:
            key_parts.append(str(arg))

    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")

    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()
