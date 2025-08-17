# PY Files/infrastructure/caching.py
"""
Phase 6: Caching Layer
Provides Redis integration with performance monitoring, cache invalidation strategies, and enterprise features.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps
import threading
import pickle
from pathlib import Path
from typing import List, Callable, Optional

# Try to import optional dependencies
try:
    import redis
    from redis.exceptions import RedisError, ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Local imports
try:
    from constants import PROJECT_ROOT
    from path_utils import get_project_paths
except ImportError:
    PROJECT_ROOT = "/Project_Root"

class CacheManager:
    """Enterprise cache manager with Redis integration and performance monitoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        self.redis_client = None
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "total_operations": 0
        }
        
        # Performance monitoring
        self.operation_times = []
        self.slow_operation_threshold = self.config.get("slow_operation_threshold", 0.1)
        self.max_operation_history = self.config.get("max_operation_history", 1000)
        
        # Cache invalidation patterns
        self.invalidation_patterns = {}
        self.pattern_locks = threading.Lock()
        
        # Initialize Redis client
        if REDIS_AVAILABLE:
            self._init_redis_client()
        else:
            logging.warning("Redis not available. Caching will be simulated in memory.")
            self._init_memory_cache()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load cache configuration."""
        try:
            config_path = Path(PROJECT_ROOT) / "configs" / "cache.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Could not load cache config: {e}")
        
        # Default configuration
        return {
            "host": os.environ.get("REDIS_HOST", "localhost"),
            "port": int(os.environ.get("REDIS_PORT", "6379")),
            "db": int(os.environ.get("REDIS_DB", "0")),
            "password": os.environ.get("REDIS_PASSWORD", ""),
            "ssl": os.environ.get("REDIS_SSL", "false").lower() == "true",
            "connection_pool_size": int(os.environ.get("REDIS_POOL_SIZE", "10")),
            "socket_timeout": int(os.environ.get("REDIS_TIMEOUT", "5")),
            "socket_connect_timeout": int(os.environ.get("REDIS_CONNECT_TIMEOUT", "2")),
            "retry_on_timeout": True,
            "health_check_interval": int(os.environ.get("REDIS_HEALTH_CHECK_INTERVAL", "30")),
            "slow_operation_threshold": float(os.environ.get("REDIS_SLOW_OP_THRESHOLD", "0.1")),
            "max_operation_history": int(os.environ.get("REDIS_MAX_OP_HISTORY", "1000")),
            "default_ttl": int(os.environ.get("REDIS_DEFAULT_TTL", "3600")),
            "compression_threshold": int(os.environ.get("REDIS_COMPRESSION_THRESHOLD", "1024"))
        }
    
    def _init_redis_client(self):
        """Initialize Redis client with connection pooling."""
        try:
            # Create connection pool
            pool = redis.ConnectionPool(
                host=self.config["host"],
                port=self.config["port"],
                db=self.config["db"],
                password=self.config["password"],
                ssl=self.config["ssl"],
                max_connections=self.config["connection_pool_size"],
                socket_timeout=self.config["socket_timeout"],
                socket_connect_timeout=self.config["socket_connect_timeout"],
                retry_on_timeout=self.config["retry_on_timeout"]
            )
            
            self.redis_client = redis.Redis(connection_pool=pool)
            
            # Test connection
            self.redis_client.ping()
            logging.info(f"Connected to Redis at {self.config['host']}:{self.config['port']}")
            
            # Start health check thread
            self._start_health_check()
            
        except Exception as e:
            logging.error(f"Failed to initialize Redis client: {e}")
            self.redis_client = None
    
    def _init_memory_cache(self):
        """Initialize in-memory cache for development/testing."""
        self._local = {}  # dict[str, tuple[Any, float|None]]
        self._tags = {}   # dict[str, set[str]]
        self.memory_cache_lock = threading.Lock()
        logging.info("Initialized in-memory cache with tag support")
    
    def _start_health_check(self):
        """Start background health check thread."""
        def health_check_worker():
            while True:
                try:
                    if self.redis_client:
                        self.redis_client.ping()
                    time.sleep(self.config["health_check_interval"])
                except Exception as e:
                    logging.warning(f"Redis health check failed: {e}")
                    time.sleep(5)  # Shorter interval on failure
        
        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()
    
    def _log_operation_performance(self, operation: str, execution_time: float, 
                                  key: str = None, size: int = 0):
        """Log operation performance metrics."""
        operation_info = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "key": key,
            "execution_time": execution_time,
            "size": size,
            "slow_operation": execution_time > self.slow_operation_threshold
        }
        
        self.operation_times.append(operation_info)
        
        # Maintain operation history size
        if len(self.operation_times) > self.max_operation_history:
            self.operation_times.pop(0)
        
        # Log slow operations
        if execution_time > self.slow_operation_threshold:
            logging.warning(f"Slow cache operation detected: {execution_time:.3f}s - {operation}")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage with compression if needed."""
        try:
            serialized = pickle.dumps(value)
            
            # Compress if above threshold
            if len(serialized) > self.config["compression_threshold"]:
                import gzip
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized):
                    return b"gzip:" + compressed
            
            return serialized
            
        except Exception as e:
            logging.error(f"Serialization failed: {e}")
            # Fallback to JSON
            return json.dumps(value).encode('utf-8')
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from storage with decompression if needed."""
        try:
            if value.startswith(b"gzip:"):
                import gzip
                compressed = value[5:]  # Remove "gzip:" prefix
                decompressed = gzip.decompress(compressed)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(value)
                
        except Exception as e:
            logging.error(f"Deserialization failed: {e}")
            # Fallback to JSON
            try:
                return json.loads(value.decode('utf-8'))
            except:
                return value.decode('utf-8', errors='ignore')
    
    def set(self, key: str, value: Any, ttl: int = None, 
            pattern: str = None, tags: List[str] = None) -> bool:
        """Set a value in cache with TTL and pattern invalidation."""
        start_time = time.time()
        ttl = ttl or self.config["default_ttl"]
        
        try:
            if self.redis_client:
                # Serialize value
                serialized_value = self._serialize_value(value)
                
                # Store in Redis
                result = self.redis_client.setex(key, ttl, serialized_value)
                
                # Store metadata for pattern invalidation
                if pattern or tags:
                    metadata = {
                        "pattern": pattern,
                        "tags": tags or [],
                        "created_at": datetime.now().isoformat(),
                        "ttl": ttl
                    }
                    meta_key = f"meta:{key}"
                    self.redis_client.setex(meta_key, ttl, json.dumps(metadata))
                
                # Update invalidation patterns
                if pattern:
                    with self.pattern_locks:
                        if pattern not in self.invalidation_patterns:
                            self.invalidation_patterns[pattern] = set()
                        self.invalidation_patterns[pattern].add(key)
                
                self.cache_stats["sets"] += 1
                self.cache_stats["total_operations"] += 1
                
                # Log performance
                execution_time = time.time() - start_time
                self._log_operation_performance("set", execution_time, key, len(serialized_value))
                
                return result
                
            else:
                # Fallback to memory cache
                with self.memory_cache_lock:
                    expiry_time = time.time() + ttl if ttl else None
                    self._local[key] = (value, expiry_time)
                    
                    # Add tags if provided
                    if tags:
                        for tag in tags:
                            if tag not in self._tags:
                                self._tags[tag] = set()
                            self._tags[tag].add(key)
                
                self.cache_stats["sets"] += 1
                self.cache_stats["total_operations"] += 1
                return True
                
        except Exception as e:
            logging.error(f"Cache set failed for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        start_time = time.time()
        
        try:
            if self.redis_client:
                # Get from Redis
                value = self.redis_client.get(key)
                
                if value is not None:
                    # Deserialize value
                    deserialized_value = self._deserialize_value(value)
                    
                    self.cache_stats["hits"] += 1
                    self.cache_stats["total_operations"] += 1
                    
                    # Log performance
                    execution_time = time.time() - start_time
                    self._log_operation_performance("get", execution_time, key, len(value))
                    
                    return deserialized_value
                else:
                    self.cache_stats["misses"] += 1
                    self.cache_stats["total_operations"] += 1
                    return default
                    
            else:
                # Fallback to memory cache
                with self.memory_cache_lock:
                    if key in self._local:
                        value, expiry_time = self._local[key]
                        # Check TTL
                        if expiry_time is None or time.time() < expiry_time:
                            self.cache_stats["hits"] += 1
                            self.cache_stats["total_operations"] += 1
                            return value
                        else:
                            # Expired, remove
                            del self._local[key]
                    
                    self.cache_stats["misses"] += 1
                    self.cache_stats["total_operations"] += 1
                    return default
                    
        except Exception as e:
            logging.error(f"Cache get failed for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        start_time = time.time()
        
        try:
            if self.redis_client:
                # Delete from Redis
                result = self.redis_client.delete(key)
                
                # Delete metadata
                meta_key = f"meta:{key}"
                self.redis_client.delete(meta_key)
                
                # Remove from invalidation patterns
                with self.pattern_locks:
                    for pattern, keys in self.invalidation_patterns.items():
                        if key in keys:
                            keys.remove(key)
                
                self.cache_stats["deletes"] += 1
                self.cache_stats["total_operations"] += 1
                
                # Log performance
                execution_time = time.time() - start_time
                self._log_operation_performance("delete", execution_time, key)
                
                return result > 0
                
            else:
                # Fallback to memory cache
                with self.memory_cache_lock:
                    if key in self._local:
                        del self._local[key]
                        # Remove from all tag sets
                        for tag_set in self._tags.values():
                            tag_set.discard(key)
                        self.cache_stats["deletes"] += 1
                        self.cache_stats["total_operations"] += 1
                        return True
                    return False
                    
        except Exception as e:
            logging.error(f"Cache delete failed for key {key}: {e}")
            self.cache_stats["errors"] += 1
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        start_time = time.time()
        invalidated_count = 0
        
        try:
            if self.redis_client:
                # Get all keys matching pattern
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    # Delete all matching keys
                    invalidated_count = self.redis_client.delete(*keys)
                    
                    # Delete metadata
                    meta_keys = [f"meta:{key.decode()}" for key in keys]
                    if meta_keys:
                        self.redis_client.delete(*meta_keys)
                
                # Clear pattern from invalidation patterns
                with self.pattern_locks:
                    if pattern in self.invalidation_patterns:
                        del self.invalidation_patterns[pattern]
            
            else:
                # Fallback to memory cache
                with self.memory_cache_lock:
                    keys_to_delete = [key for key in self._local.keys()
                                    if self._pattern_match(key, pattern)]
                    
                    for key in keys_to_delete:
                        del self._local[key]
                        # Remove from all tag sets
                        for tag_set in self._tags.values():
                            tag_set.discard(key)
                        invalidated_count += 1
            
            # Log performance
            execution_time = time.time() - start_time
            self._log_operation_performance("invalidate_pattern", execution_time, 
                                          pattern, invalidated_count)
            
            logging.info(f"Invalidated {invalidated_count} keys matching pattern: {pattern}")
            return invalidated_count
            
        except Exception as e:
            logging.error(f"Pattern invalidation failed for {pattern}: {e}")
            return 0
    
    def invalidate_tags(self, tags: List[str]) -> int:
        """Invalidate all keys with matching tags."""
        start_time = time.time()
        invalidated_count = 0
        
        try:
            if self.redis_client:
                # Scan for keys with matching tags
                for key in self.redis_client.scan_iter():
                    meta_key = f"meta:{key.decode()}"
                    metadata = self.redis_client.get(meta_key)
                    
                    if metadata:
                        try:
                            meta_data = json.loads(metadata)
                            if any(tag in meta_data.get("tags", []) for tag in tags):
                                # Delete key and metadata
                                self.redis_client.delete(key)
                                self.redis_client.delete(meta_key)
                                invalidated_count += 1
                        except:
                            pass
            
            else:
                # Fallback to memory cache
                with self.memory_cache_lock:
                    for tag in tags:
                        if tag in self._tags:
                            # Delete all keys with this tag
                            keys_to_delete = list(self._tags[tag])
                            for key in keys_to_delete:
                                if key in self._local:
                                    del self._local[key]
                                    invalidated_count += 1
                            # Remove the tag
                            del self._tags[tag]
            
            # Log performance
            execution_time = time.time() - start_time
            self._log_operation_performance("invalidate_tags", execution_time, 
                                          str(tags), invalidated_count)
            
            logging.info(f"Invalidated {invalidated_count} keys with tags: {tags}")
            return invalidated_count
            
        except Exception as e:
            logging.error(f"Tag invalidation failed for {tags}: {e}")
            return 0
    
    def _pattern_match(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for memory cache."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = {
            "cache_stats": self.cache_stats.copy(),
            "performance_metrics": {
                "total_operations": len(self.operation_times),
                "slow_operations": len([op for op in self.operation_times 
                                     if op["execution_time"] > self.slow_operation_threshold]),
                "average_execution_time": 0.0,
                "max_execution_time": 0.0,
                "min_execution_time": float('inf')
            },
            "invalidation_patterns": {
                "total_patterns": len(self.invalidation_patterns),
                "patterns": list(self.invalidation_patterns.keys())
            }
        }
        
        if self.operation_times:
            execution_times = [op["execution_time"] for op in self.operation_times]
            stats["performance_metrics"]["average_execution_time"] = sum(execution_times) / len(execution_times)
            stats["performance_metrics"]["max_execution_time"] = max(execution_times)
            stats["performance_metrics"]["min_execution_time"] = min(execution_times)
        
        # Add hit rate
        total_requests = stats["cache_stats"]["hits"] + stats["cache_stats"]["misses"]
        if total_requests > 0:
            stats["cache_stats"]["hit_rate"] = stats["cache_stats"]["hits"] / total_requests
        else:
            stats["cache_stats"]["hit_rate"] = 0.0
        
        return stats
    
    def clear_statistics(self):
        """Clear performance statistics."""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "total_operations": 0
        }
        self.operation_times = []
        logging.info("Cache statistics cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            if self.redis_client:
                # Test Redis connection
                self.redis_client.ping()
                health_status["checks"]["redis"] = "healthy"
                
                # Check memory usage
                info = self.redis_client.info("memory")
                used_memory = info.get("used_memory_human", "Unknown")
                health_status["checks"]["memory"] = f"healthy ({used_memory})"
                
            else:
                health_status["checks"]["redis"] = "not_available"
                health_status["checks"]["memory"] = "simulated"
            
            # Check performance
            stats = self.get_statistics()
            if stats["performance_metrics"]["slow_operations"] > 10:
                health_status["status"] = "degraded"
                health_status["checks"]["performance"] = "degraded"
            else:
                health_status["checks"]["performance"] = "healthy"
                
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["checks"]["error"] = str(e)
        
        return health_status
    
    def close(self):
        """Close the cache manager."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logging.info("Redis connection closed")
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def cache_result(ttl: Optional[int] = None, key_prefix: str = "", 
                pattern: str = None, tags: List[str] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add args and kwargs to key
            if args:
                key_parts.append(str(hash(str(args))))
            if kwargs:
                key_parts.append(str(hash(str(sorted(kwargs.items())))))
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache = CacheManager()
            try:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl, pattern, tags)
                return result
                
            finally:
                cache.close()
        
        return wrapper
    return decorator


# Factory function for easy instantiation
def create_cache_manager(config: Dict[str, Any] = None) -> CacheManager:
    """Create a cache manager instance."""
    return CacheManager(config)


# Example usage
if __name__ == "__main__":
    # Test cache manager
    cache = CacheManager()
    
    try:
        # Test basic operations
        cache.set("test:key", {"data": "test_value"}, ttl=60)
        value = cache.get("test:key")
        print(f"Cached value: {value}")
        
        # Test pattern invalidation
        cache.set("user:123:profile", {"name": "John"}, ttl=60, pattern="user:*:profile")
        cache.set("user:456:profile", {"name": "Jane"}, ttl=60, pattern="user:*:profile")
        
        invalidated = cache.invalidate_pattern("user:*:profile")
        print(f"Invalidated {invalidated} keys")
        
        # Get statistics
        stats = cache.get_statistics()
        print(f"Cache statistics: {stats}")
        
        # Health check
        health = cache.health_check()
        print(f"Cache health: {health['status']}")
        
    finally:
        cache.close()
