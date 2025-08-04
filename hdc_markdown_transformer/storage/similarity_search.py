"""Enhanced similarity search functionality with ranking, filtering, and batch operations."""

import logging
from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np
from dataclasses import dataclass

from ..core.interfaces import VectorDatabaseInterface
from ..core.models import SimilarityResult


logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filter configuration for similarity search."""
    min_similarity: Optional[float] = None
    max_similarity: Optional[float] = None
    token_patterns: Optional[List[str]] = None  # Regex patterns to match tokens
    metadata_filters: Optional[Dict[str, Any]] = None  # Key-value filters for metadata
    exclude_tokens: Optional[List[str]] = None  # Tokens to exclude from results


@dataclass
class SearchConfig:
    """Configuration for similarity search operations."""
    k: int = 10
    similarity_threshold: float = 0.0
    enable_reranking: bool = True
    rerank_factor: float = 2.0  # Fetch k * rerank_factor, then rerank to k
    batch_size: int = 100  # For batch operations
    normalize_query: bool = True
    return_vectors: bool = True


class SimilaritySearchEngine:
    """Enhanced similarity search engine with ranking, filtering, and batch operations."""
    
    def __init__(self, vector_database: VectorDatabaseInterface):
        """
        Initialize similarity search engine.
        
        Args:
            vector_database: Vector database instance to search
        """
        self.vector_database = vector_database
        self.logger = logging.getLogger(__name__)
    
    def search(self, 
               query_vector: np.ndarray, 
               config: SearchConfig = None,
               search_filter: SearchFilter = None) -> List[SimilarityResult]:
        """
        Perform enhanced similarity search with ranking and filtering.
        
        Args:
            query_vector: Query vector
            config: Search configuration
            search_filter: Filter configuration
            
        Returns:
            List of filtered and ranked similarity results
        """
        config = config or SearchConfig()
        search_filter = search_filter or SearchFilter()
        
        try:
            # Determine how many results to fetch for reranking
            fetch_k = config.k
            if config.enable_reranking:
                fetch_k = int(config.k * config.rerank_factor)
            
            # Perform initial search
            raw_results = self.vector_database.similarity_search(query_vector, fetch_k)
            self.logger.info(f"[SIMSEARCH] Résultats bruts: {len(raw_results)} candidats (top 10: {[r.token for r in raw_results[:10]]})")
            if raw_results:
                scores = [r.similarity_score for r in raw_results]
                self.logger.info(f"[SIMSEARCH] Score min/max/moyen (brut): {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")
            
            # Apply filters
            filtered_results = self._apply_filters(raw_results, search_filter)
            self.logger.info(f"[SIMSEARCH] Après filtrage: {len(filtered_results)} candidats (top 10: {[r.token for r in filtered_results[:10]]})")
            if filtered_results:
                scores = [r.similarity_score for r in filtered_results]
                self.logger.info(f"[SIMSEARCH] Score min/max/moyen (filtré): {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")
            
            # Apply reranking if enabled
            if config.enable_reranking and len(filtered_results) > config.k:
                reranked_results = self._rerank_results(query_vector, filtered_results, config)
                self.logger.info(f"[SIMSEARCH] Après reranking: {len(reranked_results)} candidats (top 10: {[r.token for r in reranked_results[:10]]})")
                if reranked_results:
                    scores = [r.similarity_score for r in reranked_results]
                    self.logger.info(f"[SIMSEARCH] Score min/max/moyen (rerank): {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")
                final_results = reranked_results[:config.k]
            else:
                final_results = filtered_results[:config.k]
            
            # Apply similarity threshold
            before_thresh = len(final_results)
            final_results = [r for r in final_results if r.similarity_score >= config.similarity_threshold]
            self.logger.info(f"[SIMSEARCH] Après seuil (threshold={config.similarity_threshold}): {len(final_results)}/{before_thresh} candidats gardés (top 10: {[r.token for r in final_results[:10]]})")
            if final_results:
                scores = [r.similarity_score for r in final_results]
                self.logger.info(f"[SIMSEARCH] Score min/max/moyen (final): {min(scores):.4f}/{max(scores):.4f}/{sum(scores)/len(scores):.4f}")
            else:
                self.logger.warning(f"[SIMSEARCH] Aucun candidat final après application du seuil de similarité {config.similarity_threshold}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            raise RuntimeError(f"Similarity search failed: {e}")
    
    def batch_search(self, 
                     query_vectors: List[np.ndarray], 
                     config: SearchConfig = None,
                     search_filter: SearchFilter = None) -> List[List[SimilarityResult]]:
        """
        Perform batch similarity search for multiple query vectors.
        
        Args:
            query_vectors: List of query vectors
            config: Search configuration
            search_filter: Filter configuration
            
        Returns:
            List of result lists, one for each query vector
        """
        config = config or SearchConfig()
        
        try:
            results = []
            
            # Process in batches to manage memory
            for i in range(0, len(query_vectors), config.batch_size):
                batch = query_vectors[i:i + config.batch_size]
                
                batch_results = []
                for query_vector in batch:
                    search_results = self.search(query_vector, config, search_filter)
                    batch_results.append(search_results)
                
                results.extend(batch_results)
                
                self.logger.debug(f"Processed batch {i//config.batch_size + 1}/{(len(query_vectors) + config.batch_size - 1)//config.batch_size}")
            
            self.logger.info(f"Completed batch search for {len(query_vectors)} queries")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch similarity search: {e}")
            raise RuntimeError(f"Batch similarity search failed: {e}")
    
    def search_with_expansion(self, 
                              query_vector: np.ndarray, 
                              expansion_terms: List[str],
                              config: SearchConfig = None) -> List[SimilarityResult]:
        """
        Perform similarity search with query expansion using additional terms.
        
        Args:
            query_vector: Original query vector
            expansion_terms: Additional terms to expand the query
            config: Search configuration
            
        Returns:
            List of similarity results from expanded query
        """
        config = config or SearchConfig()
        
        try:
            # Get vectors for expansion terms
            expansion_vectors = []
            for term in expansion_terms:
                term_vector = self.vector_database.get_vector(term)
                if term_vector is not None:
                    expansion_vectors.append(term_vector)
            
            if not expansion_vectors:
                # No expansion vectors found, use original query
                return self.search(query_vector, config)
            
            # Combine query vector with expansion vectors (weighted average)
            all_vectors = [query_vector] + expansion_vectors
            weights = [0.7] + [0.3 / len(expansion_vectors)] * len(expansion_vectors)  # Give more weight to original query
            
            expanded_query = np.average(all_vectors, axis=0, weights=weights)
            
            # Normalize if required
            if config.normalize_query and np.linalg.norm(expanded_query) > 0:
                expanded_query = expanded_query / np.linalg.norm(expanded_query)
            
            return self.search(expanded_query, config)
            
        except Exception as e:
            self.logger.error(f"Error in query expansion search: {e}")
            raise RuntimeError(f"Query expansion search failed: {e}")
    
    def find_similar_clusters(self, 
                              query_vector: np.ndarray, 
                              cluster_size: int = 5,
                              num_clusters: int = 3,
                              config: SearchConfig = None) -> List[List[SimilarityResult]]:
        """
        Find clusters of similar results around the query.
        
        Args:
            query_vector: Query vector
            cluster_size: Number of results per cluster
            num_clusters: Number of clusters to find
            config: Search configuration
            
        Returns:
            List of clusters, each containing similar results
        """
        config = config or SearchConfig()
        
        try:
            # Get more results than needed for clustering
            extended_config = SearchConfig(
                k=cluster_size * num_clusters * 2,
                similarity_threshold=config.similarity_threshold,
                enable_reranking=config.enable_reranking
            )
            
            results = self.search(query_vector, extended_config)
            
            if len(results) < cluster_size:
                return [results]  # Not enough results for clustering
            
            # Simple clustering based on similarity scores
            clusters = []
            used_indices = set()
            
            for _ in range(num_clusters):
                if len(used_indices) >= len(results):
                    break
                
                # Find the next best unused result as cluster center
                cluster_center_idx = None
                for i, result in enumerate(results):
                    if i not in used_indices:
                        cluster_center_idx = i
                        break
                
                if cluster_center_idx is None:
                    break
                
                # Build cluster around this center
                cluster = [results[cluster_center_idx]]
                used_indices.add(cluster_center_idx)
                
                # Add similar results to cluster
                center_vector = results[cluster_center_idx].vector
                for i, result in enumerate(results):
                    if i in used_indices or len(cluster) >= cluster_size:
                        continue
                    
                    # Calculate similarity to cluster center
                    if result.vector is not None and center_vector is not None:
                        similarity = np.dot(center_vector, result.vector)
                        if similarity > 0.8:  # High similarity threshold for clustering
                            cluster.append(result)
                            used_indices.add(i)
                
                clusters.append(cluster)
            
            self.logger.debug(f"Found {len(clusters)} clusters with sizes: {[len(c) for c in clusters]}")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error in cluster search: {e}")
            raise RuntimeError(f"Cluster search failed: {e}")
    
    def _apply_filters(self, results: List[SimilarityResult], search_filter: SearchFilter) -> List[SimilarityResult]:
        """Apply filters to search results."""
        if not search_filter:
            return results
        
        filtered_results = []
        
        for result in results:
            # Apply similarity filters
            if search_filter.min_similarity is not None and result.similarity_score < search_filter.min_similarity:
                continue
            if search_filter.max_similarity is not None and result.similarity_score > search_filter.max_similarity:
                continue
            
            # Apply token exclusion filter
            if search_filter.exclude_tokens and result.token in search_filter.exclude_tokens:
                continue
            
            # Apply token pattern filters
            if search_filter.token_patterns:
                import re
                pattern_match = False
                for pattern in search_filter.token_patterns:
                    if re.search(pattern, result.token):
                        pattern_match = True
                        break
                if not pattern_match:
                    continue
            
            # Apply metadata filters
            if search_filter.metadata_filters:
                metadata_match = True
                for key, value in search_filter.metadata_filters.items():
                    if key not in result.metadata or result.metadata[key] != value:
                        metadata_match = False
                        break
                if not metadata_match:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_results(self, 
                        query_vector: np.ndarray, 
                        results: List[SimilarityResult], 
                        config: SearchConfig) -> List[SimilarityResult]:
        """
        Rerank results using additional scoring methods.
        
        This implementation uses a simple weighted combination of:
        - Original similarity score
        - Vector magnitude (as a quality indicator)
        - Token length (prefer meaningful tokens)
        """
        try:
            reranked_results = []
            
            for result in results:
                # Original similarity score (weight: 0.7)
                similarity_score = result.similarity_score * 0.7
                
                # Vector magnitude score (weight: 0.2)
                magnitude_score = 0.0
                if result.vector is not None:
                    magnitude = np.linalg.norm(result.vector)
                    magnitude_score = min(magnitude, 1.0) * 0.2  # Normalize to [0, 0.2]
                
                # Token length score (weight: 0.1)
                # Prefer tokens that are not too short or too long
                token_length = len(result.token)
                if 3 <= token_length <= 15:
                    length_score = 0.1
                elif token_length < 3:
                    length_score = 0.05
                else:
                    length_score = 0.1 * (20 - min(token_length, 20)) / 20
                
                # Combined reranking score
                rerank_score = similarity_score + magnitude_score + length_score
                
                # Create new result with reranking score
                reranked_result = SimilarityResult(
                    token=result.token,
                    similarity_score=rerank_score,
                    vector=result.vector,
                    metadata={**result.metadata, 'original_similarity': result.similarity_score}
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by reranking score
            reranked_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error in reranking: {e}")
            return results  # Return original results if reranking fails
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search engine and underlying database."""
        try:
            stats = {
                "database_type": type(self.vector_database).__name__,
                "database_size": getattr(self.vector_database, 'size', lambda: 0)()
            }
            
            # Get database-specific statistics if available
            if hasattr(self.vector_database, 'get_statistics'):
                db_stats = self.vector_database.get_statistics()
                stats.update(db_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting search statistics: {e}")
            return {"error": str(e)}


class BatchSimilaritySearchEngine:
    """Specialized engine for high-performance batch similarity search operations."""
    
    def __init__(self, vector_database: VectorDatabaseInterface):
        """Initialize batch search engine."""
        self.vector_database = vector_database
        self.logger = logging.getLogger(__name__)
    
    def parallel_batch_search(self, 
                              query_vectors: List[np.ndarray], 
                              k: int = 10,
                              batch_size: int = 50) -> List[List[SimilarityResult]]:
        """
        Perform parallel batch search for optimal performance.
        
        Args:
            query_vectors: List of query vectors
            k: Number of results per query
            batch_size: Size of processing batches
            
        Returns:
            List of result lists for each query
        """
        try:
            all_results = []
            
            # Process in batches
            for i in range(0, len(query_vectors), batch_size):
                batch = query_vectors[i:i + batch_size]
                
                # Process batch
                batch_results = []
                for query_vector in batch:
                    results = self.vector_database.similarity_search(query_vector, k)
                    batch_results.append(results)
                
                all_results.extend(batch_results)
                
                self.logger.debug(f"Processed batch {i//batch_size + 1}")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in parallel batch search: {e}")
            raise RuntimeError(f"Parallel batch search failed: {e}")
    
    def streaming_search(self, 
                         query_vectors: List[np.ndarray], 
                         k: int = 10) -> List[SimilarityResult]:
        """
        Perform streaming search that yields results as they're computed.
        
        Args:
            query_vectors: List of query vectors
            k: Number of results per query
            
        Yields:
            Individual similarity results as they're computed
        """
        try:
            for i, query_vector in enumerate(query_vectors):
                results = self.vector_database.similarity_search(query_vector, k)
                for result in results:
                    # Add query index to metadata
                    result.metadata['query_index'] = i
                    yield result
                    
        except Exception as e:
            self.logger.error(f"Error in streaming search: {e}")
            raise RuntimeError(f"Streaming search failed: {e}")