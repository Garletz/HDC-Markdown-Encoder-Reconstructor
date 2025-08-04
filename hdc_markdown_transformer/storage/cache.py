"""Document hypervector caching system for HDC Markdown Transformer."""

import os
import time
import hashlib
import pickle
import threading
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import json
import logging

from ..core.models import PreprocessedDocument


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    vector: np.ndarray
    timestamp: float
    access_count: int
    last_accessed: float
    document_hash: str
    metadata: Dict[str, Any]


@dataclass
class CacheStats:
    """Cache statistics for monitoring and optimization."""
    total_entries: int
    cache_hits: int
    cache_misses: int
    total_size_bytes: int
    hit_rate: float
    average_access_time: float


class DocumentVectorCache:
    """
    High-performance caching system for document hypervectors.
    
    Features:
    - LRU eviction policy
    - Thread-safe operations
    - Persistent storage with compression
    - Cache invalidation and cleanup
    - Performance monitoring
    """
    
    def __init__(self, 
                 cache_dir: str = ".cache/hdc_vectors",
                 max_entries: int = 1000,
                 max_size_mb: int = 500,
                 ttl_hours: int = 24):
        """
        Initialize the document vector cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_entries: Maximum number of cached vectors
            max_size_mb: Maximum cache size in MB
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_entries = max_entries
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_hours * 3600.0
        
        # In-memory cache for fast access
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU tracking
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStats(0, 0, 0, 0, 0.0, 0.0)
        self._access_times: List[float] = []
        
        # Load existing cache from disk
        self._load_cache_index()
    
    def _generate_cache_key(self, document: PreprocessedDocument) -> str:
        """Generate a unique cache key for a document."""
        # Create hash from document content and preprocessing parameters
        content_hash = hashlib.sha256(document.original_content.encode()).hexdigest()
        
        # Include preprocessing parameters in hash
        params = {
            'tokens_count': len(document.tokens),
            'structure_headers': len(document.structure.headers),
            'structure_lists': len(document.structure.lists),
            'tfidf_keys': sorted(document.tf_idf_weights.keys())[:10]  # Sample for uniqueness
        }
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"{content_hash[:16]}_{params_hash}"
    
    def get(self, document: PreprocessedDocument) -> Optional[np.ndarray]:
        """
        Retrieve cached vector for a document.
        
        Args:
            document: Preprocessed document to look up
            
        Returns:
            Cached hypervector if found, None otherwise
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(document)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check TTL
                current_time = time.time()
                if current_time - entry.timestamp > self.ttl_seconds:
                    self._remove_entry(cache_key)
                    self._stats.cache_misses += 1
                    logger.debug(f"Cache entry {cache_key[:8]}... expired (age: {current_time - entry.timestamp:.3f}s, TTL: {self.ttl_seconds:.3f}s)")
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Update LRU order
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                
                self._stats.cache_hits += 1
                access_time = time.time() - start_time
                self._access_times.append(access_time)
                
                logger.debug(f"Cache hit for key {cache_key[:8]}...")
                return entry.vector.copy()
            else:
                self._stats.cache_misses += 1
                logger.debug(f"Cache miss for key {cache_key[:8]}...")
                return None
    
    def put(self, document: PreprocessedDocument, vector: np.ndarray, 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a vector in the cache.
        
        Args:
            document: Preprocessed document
            vector: Hypervector to cache
            metadata: Optional metadata to store with the vector
        """
        cache_key = self._generate_cache_key(document)
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                vector=vector.copy(),
                timestamp=time.time(),
                access_count=1,
                last_accessed=time.time(),
                document_hash=cache_key,
                metadata=metadata or {}
            )
            
            # Check if we need to evict entries
            self._ensure_capacity()
            
            # Store entry
            self._cache[cache_key] = entry
            
            # Update LRU order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            
            # Update statistics
            self._stats.total_entries = len(self._cache)
            self._update_cache_size()
            
            # Persist to disk
            self._persist_entry(cache_key, entry)
            
            logger.debug(f"Cached vector for key {cache_key[:8]}...")
    
    def invalidate(self, document: PreprocessedDocument) -> bool:
        """
        Invalidate cached vector for a document.
        
        Args:
            document: Document to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_key = self._generate_cache_key(document)
        
        with self._lock:
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                logger.debug(f"Invalidated cache entry {cache_key[:8]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats(0, 0, 0, 0, 0.0, 0.0)
            self._access_times.clear()
            
            # Remove cache files
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                age = current_time - entry.timestamp
                if age > self.ttl_seconds:
                    expired_keys.append(key)
                    logger.debug(f"Found expired entry {key[:8]}... (age: {age:.3f}s, TTL: {self.ttl_seconds:.3f}s)")
            
            for key in expired_keys:
                self._remove_entry(key)
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self._stats.total_entries = len(self._cache)
            self._update_cache_size()
            
            total_requests = self._stats.cache_hits + self._stats.cache_misses
            self._stats.hit_rate = (
                self._stats.cache_hits / total_requests if total_requests > 0 else 0.0
            )
            
            self._stats.average_access_time = (
                sum(self._access_times) / len(self._access_times) 
                if self._access_times else 0.0
            )
            
            return self._stats
    
    def _ensure_capacity(self) -> None:
        """Ensure cache doesn't exceed capacity limits."""
        # Remove expired entries first
        self.cleanup_expired()
        
        # Check entry count limit
        while len(self._cache) >= self.max_entries:
            self._evict_lru()
        
        # Check size limit
        while self._get_cache_size() > self.max_size_bytes:
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_order:
            return
        
        lru_key = self._access_order[0]
        self._remove_entry(lru_key)
        logger.debug(f"Evicted LRU entry {lru_key[:8]}...")
    
    def _remove_entry(self, cache_key: str) -> None:
        """Remove an entry from cache and disk."""
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        
        # Remove from disk
        cache_file = self.cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            cache_file.unlink()
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size in bytes."""
        total_size = 0
        for entry in self._cache.values():
            total_size += entry.vector.nbytes
            total_size += len(pickle.dumps(entry.metadata))
        return total_size
    
    def _update_cache_size(self) -> None:
        """Update cache size statistics."""
        self._stats.total_size_bytes = self._get_cache_size()
    
    def _persist_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            # Prepare data for serialization
            data = {
                'vector': entry.vector,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'document_hash': entry.document_hash,
                'metadata': entry.metadata
            }
            
            # Use numpy's compressed format for vectors
            np.savez_compressed(cache_file, **data)
            
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {cache_key}: {e}")
    
    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        try:
            for cache_file in self.cache_dir.glob("*.cache.npz"):
                cache_key = cache_file.stem.replace('.cache', '')
                
                try:
                    data = np.load(cache_file, allow_pickle=True)
                    
                    entry = CacheEntry(
                        vector=data['vector'],
                        timestamp=float(data['timestamp']),
                        access_count=int(data['access_count']),
                        last_accessed=float(data['last_accessed']),
                        document_hash=str(data['document_hash']),
                        metadata=data['metadata'].item() if 'metadata' in data else {}
                    )
                    
                    # Check if entry is still valid
                    current_time = time.time()
                    age = current_time - entry.timestamp
                    if age <= self.ttl_seconds:
                        self._cache[cache_key] = entry
                        self._access_order.append(cache_key)
                    else:
                        # Remove expired file
                        cache_file.unlink()
                        
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {cache_key}: {e}")
                    # Remove corrupted file
                    cache_file.unlink()
            
            logger.info(f"Loaded {len(self._cache)} cache entries from disk")
            
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")


class CacheManager:
    """
    High-level cache manager for coordinating multiple cache instances.
    """
    
    def __init__(self, base_cache_dir: str = ".cache"):
        """Initialize cache manager."""
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._caches: Dict[str, DocumentVectorCache] = {}
        self._lock = threading.RLock()
    
    def get_cache(self, cache_name: str, **kwargs) -> DocumentVectorCache:
        """
        Get or create a named cache instance.
        
        Args:
            cache_name: Name of the cache
            **kwargs: Arguments for cache initialization
            
        Returns:
            DocumentVectorCache instance
        """
        with self._lock:
            if cache_name not in self._caches:
                cache_dir = self.base_cache_dir / cache_name
                self._caches[cache_name] = DocumentVectorCache(
                    cache_dir=str(cache_dir),
                    **kwargs
                )
            return self._caches[cache_name]
    
    def clear_all_caches(self) -> None:
        """Clear all managed caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
            self._caches.clear()
    
    def get_global_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all managed caches."""
        with self._lock:
            return {
                name: cache.get_stats() 
                for name, cache in self._caches.items()
            }
    
    def cleanup_all_expired(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches."""
        with self._lock:
            return {
                name: cache.cleanup_expired()
                for name, cache in self._caches.items()
            }


# Global cache manager instance
cache_manager = CacheManager()