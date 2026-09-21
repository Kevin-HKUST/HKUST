import pickle
import time
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class Cache:
    """缓存管理类 - 使用内存缓存，避免Redis依赖"""

    def __init__(self):
        self.memory_cache = {}
        self.redis_available = False
        logger.info("Using in-memory cache (Redis not required)")

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if key in self.memory_cache:
                data, expiry = self.memory_cache[key]
                if expiry is None or time.time() < expiry:
                    return data
                else:
                    # 过期删除
                    del self.memory_cache[key]
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值"""
        try:
            expiry = time.time() + ttl if ttl > 0 else None
            self.memory_cache[key] = (value, expiry)

            # 简单的内存清理：如果缓存太大，删除一些过期项目
            if len(self.memory_cache) > 1000:
                self._cleanup_expired()

            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            if key in self.memory_cache:
                del self.memory_cache[key]
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if key in self.memory_cache:
            data, expiry = self.memory_cache[key]
            if expiry is None or time.time() < expiry:
                return True
            else:
                del self.memory_cache[key]
        return False

    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for key, (_, expiry) in self.memory_cache.items():
            if expiry is not None and current_time >= expiry:
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")


# 全局缓存实例
cache = Cache()