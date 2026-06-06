import json
import hashlib
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps, lru_cache
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CacheBackend:
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._access_order: list = []
    
    def _generate_key(self, *args, **kwargs) -> str:
        key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _evict_if_needed(self):
        while len(self._cache) > self._max_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)
            else:
                break
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.get('expires_at') and time.time() > entry['expires_at']:
            self.delete(key)
            return None
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._evict_if_needed()
        
        expires_at = None
        if ttl is not None or self._default_ttl:
            expires_at = time.time() + (ttl or self._default_ttl)
        
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.utcnow().isoformat()
        }
        
        if key not in self._access_order:
            self._access_order.append(key)
    
    def delete(self, key: str):
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear(self):
        self._cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        valid_count = 0
        expired_count = 0
        now = time.time()
        
        for entry in self._cache.values():
            if entry.get('expires_at') and now > entry['expires_at']:
                expired_count += 1
            else:
                valid_count += 1
        
        return {
            'total_items': len(self._cache),
            'valid_items': valid_count,
            'expired_items': expired_count,
            'max_size': self._max_size
        }


_cache = CacheBackend()


def get_cache() -> CacheBackend:
    return _cache


def cached(ttl: int = 3600, key_prefix: str = ''):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{_cache._generate_key(*args, **kwargs)}"
            
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            result = func(*args, **kwargs)
            
            if result is not None:
                _cache.set(cache_key, result, ttl=ttl)
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator



