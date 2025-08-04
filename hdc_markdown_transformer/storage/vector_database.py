"""Vector database implementations for HDC dictionary storage."""

import logging
from typing import Dict, List, Optional
import numpy as np

from ..core.interfaces import VectorDatabaseInterface
from ..core.models import SimilarityResult


logger = logging.getLogger(__name__)


class InMemoryVectorDatabase(VectorDatabaseInterface):
    """
    Simple in-memory vector database implementation for development and testing.
    
    This implementation stores vectors in memory and performs brute-force
    similarity search. It's suitable for development and small-scale testing.
    """
    
    def __init__(self):
        """Initialize in-memory vector database."""
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        logger.info("Initialized InMemoryVectorDatabase")
    
    def store_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        """
        Store vectors with their identifiers.
        
        Args:
            vectors: Dictionary mapping identifiers to vectors
        """
        logger.info(f"Storing {len(vectors)} vectors")
        
        for identifier, vector in vectors.items():
            # Ensure vector is float32 and normalized
            vector = np.array(vector, dtype=np.float32)
            if np.linalg.norm(vector) > 0:
                vector = vector / np.linalg.norm(vector)
            
            self.vectors[identifier] = vector
            self.metadata[identifier] = {
                "dimension": len(vector),
                "norm": np.linalg.norm(vector)
            }
        
        logger.info(f"Total vectors stored: {len(self.vectors)}")
        # Diagnostic: afficher les 10 premiers identifiants stockés
        preview = list(self.vectors.keys())[:10]
        logger.info(f"[DIAG] Premiers identifiants dans vector database : {preview}")
    
    def similarity_search(self, query_vector: np.ndarray, k: int) -> List[SimilarityResult]:
        """
        Perform K-NN similarity search.
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of similarity results sorted by similarity score (descending)
        """
        if not self.vectors:
            logger.warning("No vectors stored in database")
            return []
        
        # Normalize query vector
        query_vector = np.array(query_vector, dtype=np.float32)
        if np.linalg.norm(query_vector) > 0:
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Calculate similarities
        similarities = []
        for identifier, vector in self.vectors.items():
            # Use cosine similarity
            similarity = np.dot(query_vector, vector)
            similarities.append((identifier, similarity, vector))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:k]
        
        # Create SimilarityResult objects
        results = []
        for identifier, similarity, vector in similarities:
            result = SimilarityResult(
                token=identifier,
                similarity_score=float(similarity),
                vector=vector.copy(),
                metadata=self.metadata.get(identifier, {})
            )
            results.append(result)
        
        logger.debug(f"Found {len(results)} similar vectors for query")
        return results
    
    def update_vector(self, identifier: str, vector: np.ndarray) -> None:
        """
        Update an existing vector.
        
        Args:
            identifier: Vector identifier
            vector: New vector
        """
        if identifier not in self.vectors:
            logger.warning(f"Vector {identifier} not found for update")
            return
        
        # Normalize vector
        vector = np.array(vector, dtype=np.float32)
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        self.vectors[identifier] = vector
        self.metadata[identifier] = {
            "dimension": len(vector),
            "norm": np.linalg.norm(vector)
        }
        
        logger.debug(f"Updated vector for {identifier}")
    
    def delete_vector(self, identifier: str) -> None:
        """
        Delete a vector by identifier.
        
        Args:
            identifier: Vector identifier to delete
        """
        if identifier in self.vectors:
            del self.vectors[identifier]
            if identifier in self.metadata:
                del self.metadata[identifier]
            logger.debug(f"Deleted vector {identifier}")
        else:
            logger.warning(f"Vector {identifier} not found for deletion")
    
    def get_vector(self, identifier: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by identifier.
        
        Args:
            identifier: Vector identifier
            
        Returns:
            Vector if found, None otherwise
        """
        return self.vectors.get(identifier)
    
    def clear(self) -> None:
        """Clear all vectors from database."""
        self.vectors.clear()
        self.metadata.clear()
        logger.info("Cleared all vectors from database")
    
    def size(self) -> int:
        """Get number of vectors in database."""
        return len(self.vectors)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.vectors:
            return {"size": 0}
        
        dimensions = [len(v) for v in self.vectors.values()]
        norms = [np.linalg.norm(v) for v in self.vectors.values()]
        
        return {
            "size": len(self.vectors),
            "avg_dimension": np.mean(dimensions),
            "min_dimension": min(dimensions),
            "max_dimension": max(dimensions),
            "avg_norm": np.mean(norms),
            "min_norm": min(norms),
            "max_norm": max(norms)
        }


class VectorDatabaseFactory:
    """Factory for creating vector database instances."""
    
    @staticmethod
    def create(db_type: str, config: Dict = None) -> VectorDatabaseInterface:
        """
        Create vector database instance.
        
        Args:
            db_type: Type of database (only "memory" supported)
            config: Database configuration (unused)
            
        Returns:
            Vector database instance
            
        Raises:
            ValueError: If database type is not supported
        """
        if db_type == "memory":
            return InMemoryVectorDatabase()
        else:
            raise ValueError(f"Unsupported database type: {db_type}. Only 'memory' is supported.")
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported vector database types."""
        return ["memory"]
    
    @staticmethod
    def validate_config(db_type: str, config: Dict) -> bool:
        """
        Validate configuration for a specific database type.
        
        Args:
            db_type: Type of database
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if db_type == "memory":
            return True  # No configuration needed
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
