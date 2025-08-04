"""HDC Document Encoder for converting preprocessed documents to hypervectors."""

import logging
from typing import List, Optional, Dict, Any
import numpy as np

from ..core.models import PreprocessedDocument
from .item_memory import ItemMemory
from .positional_encoder import PositionalEncoder

logger = logging.getLogger(__name__)


class HDCEncoder:
    """
    HDC Document Encoder for converting preprocessed documents to hypervectors.
    
    This class implements the core HDC encoding operations including bundling
    (sum + thresholding), batch processing, and weighted bundling using TF-IDF scores.
    """
    
    def __init__(self, 
                 dimension: int,
                 item_memory: ItemMemory,
                 positional_encoder: Optional[PositionalEncoder] = None,
                 use_positional_encoding: bool = True,
                 bundling_threshold: float = 0.0,
                 random_seed: int = 42):
        """
        Initialize HDCEncoder.
        
        Args:
            dimension: Dimension of hypervectors
            item_memory: ItemMemory instance with word-to-vector mappings
            positional_encoder: Optional positional encoder for sequence-aware encoding
            use_positional_encoding: Whether to apply positional encoding
            bundling_threshold: Threshold for bundling operation (default 0.0 for majority vote)
            random_seed: Random seed for deterministic operations
        """
        self.dimension = dimension
        self.item_memory = item_memory
        self.use_positional_encoding = use_positional_encoding
        self.bundling_threshold = bundling_threshold
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize positional encoder if needed
        if use_positional_encoding and positional_encoder is None:
            self.positional_encoder = PositionalEncoder(dimension, random_seed=random_seed)
        else:
            self.positional_encoder = positional_encoder
        
        # Validate dimensions
        if item_memory.dimension != dimension:
            raise ValueError(f"ItemMemory dimension {item_memory.dimension} doesn't match "
                           f"encoder dimension {dimension}")
        
        logger.info(f"Initialized HDCEncoder with dimension={dimension}, "
                   f"positional_encoding={use_positional_encoding}, "
                   f"threshold={bundling_threshold}")
    
    def encode_document(self, 
                       preprocessed_doc: PreprocessedDocument,
                       use_tf_idf_weights: bool = True,
                       normalize_weights: bool = True,
                       apply_positional_encoding: bool = None) -> np.ndarray:
        """
        Encode a preprocessed document into a single hypervector.
        
        Args:
            preprocessed_doc: Preprocessed document to encode
            use_tf_idf_weights: Whether to use TF-IDF weights in bundling
            normalize_weights: Whether to normalize TF-IDF weights
            apply_positional_encoding: Whether to apply positional encoding (overrides instance setting)
            
        Returns:
            Document hypervector of shape (dimension,)
            
        Raises:
            ValueError: If document contains no valid tokens
        """
        if not preprocessed_doc.normalized_tokens:
            raise ValueError("Document contains no tokens to encode")
        
        logger.debug(f"Encoding document with {len(preprocessed_doc.normalized_tokens)} tokens")
        
        # Log OOV tokens (tokens not in ItemMemory)
        oov_tokens = [token for token in preprocessed_doc.normalized_tokens if not self.item_memory.has_word(token)]
        if oov_tokens:
            logger.info(f"OOV tokens ({len(oov_tokens)}): {oov_tokens[:20]}{'...' if len(oov_tokens) > 20 else ''}")

        # Get token vectors
        token_vectors = self._get_token_vectors(preprocessed_doc.normalized_tokens)
        
        if len(token_vectors) == 0:
            raise ValueError("No valid token vectors found in ItemMemory")
        
        # Apply positional encoding if enabled (can be forced off by apply_positional_encoding)
        use_position = self.use_positional_encoding if apply_positional_encoding is None else apply_positional_encoding
        if use_position and self.positional_encoder is not None:
            # Use positional info from preprocessing, or sequential positions
            positions = preprocessed_doc.positional_info[:len(token_vectors)]
            if len(positions) != len(token_vectors):
                positions = list(range(len(token_vectors)))
            
            token_vectors = self.positional_encoder.apply_positional_encoding(
                token_vectors, positions
            )
        
        # Apply TF-IDF weighting if requested
        if use_tf_idf_weights:
            # Get only the tokens that have vectors
            valid_tokens = [token for token in preprocessed_doc.normalized_tokens 
                           if self.item_memory.has_word(token)][:len(token_vectors)]
            weights = self._get_tf_idf_weights(
                valid_tokens,
                preprocessed_doc.tf_idf_weights,
                normalize_weights
            )
            token_vectors = self._apply_weights(token_vectors, weights)
        
        # Bundle vectors using sum + thresholding
        document_vector = self._bundle_vectors(token_vectors)
        
        logger.debug(f"Encoded document to hypervector with {np.sum(document_vector > 0)} positive components")
        
        return document_vector
    
    def encode_document_dual(self, 
                       preprocessed_doc: PreprocessedDocument) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode a preprocessed document into separate content and position hypervectors.
        
        Args:
            preprocessed_doc: Preprocessed document to encode
            
        Returns:
            Tuple of (content_vector, position_vector)
            
        Raises:
            ValueError: If document contains no valid tokens
        """
        if not preprocessed_doc.normalized_tokens:
            raise ValueError("Document contains no tokens to encode")
        
        logger.debug(f"Encoding document with {len(preprocessed_doc.normalized_tokens)} tokens")
        
        # Get token vectors for content
        token_vectors = self._get_token_vectors(preprocessed_doc.normalized_tokens)
        
        if len(token_vectors) == 0:
            raise ValueError("No valid token vectors found in ItemMemory")
        
        # Create position vectors
        positions = list(range(len(preprocessed_doc.normalized_tokens)))
        position_vectors = []
        for pos in positions:
            if pos < self.positional_encoder.max_positions:
                pos_vector = self.positional_encoder.get_position_vector(pos)
                position_vectors.append(pos_vector)
            else:
                # Fallback: create random vector for out-of-range positions
                pos_vector = np.random.choice([-1, 1], size=self.dimension)
                position_vectors.append(pos_vector)
        
        position_vectors = np.array(position_vectors)
        
        # Encode content vector (words only) - for finding which words are present
        content_vector = self._bundle_vectors(token_vectors)
        
        # Encode position vector (positions only) - for finding which positions are present
        position_vector = self._bundle_vectors(position_vectors)
        
        # Store original tokens and positions for reconstruction
        self._last_tokens = preprocessed_doc.normalized_tokens
        self._last_positions = positions
        
        # Also store individual pair vectors for order deduction
        self._last_pair_vectors = []
        for i, (word_vector, pos_vector) in enumerate(zip(token_vectors, position_vectors)):
            # HDC binding: element-wise multiplication (XOR for bipolar vectors)
            pair_vector = word_vector * pos_vector
            self._last_pair_vectors.append(pair_vector)
        
        logger.debug(f"Encoded content vector with {np.sum(content_vector > 0)} positive components")
        logger.debug(f"Encoded position vector with {np.sum(position_vector > 0)} positive components")
        logger.debug(f"Created {len(self._last_pair_vectors)} (word, position) pairs for order deduction")
        
        return content_vector, position_vector

    def encode_batch(self, 
                    preprocessed_docs: List[PreprocessedDocument],
                    use_tf_idf_weights: bool = True,
                    normalize_weights: bool = True) -> np.ndarray:
        """
        Encode multiple documents in batch.
        
        Args:
            preprocessed_docs: List of preprocessed documents
            use_tf_idf_weights: Whether to use TF-IDF weights
            normalize_weights: Whether to normalize TF-IDF weights
            
        Returns:
            Array of shape (num_docs, dimension) with document hypervectors
        """
        if not preprocessed_docs:
            raise ValueError("No documents provided for batch encoding")
        
        logger.info(f"Batch encoding {len(preprocessed_docs)} documents")
        
        document_vectors = []
        for i, doc in enumerate(preprocessed_docs):
            try:
                doc_vector = self.encode_document(doc, use_tf_idf_weights, normalize_weights)
                document_vectors.append(doc_vector)
            except ValueError as e:
                logger.warning(f"Failed to encode document {i}: {e}")
                # Create zero vector for failed documents
                document_vectors.append(np.zeros(self.dimension, dtype=np.int8))
        
        return np.array(document_vectors)
    
    def encode_tokens(self, 
                     tokens: List[str],
                     apply_positional_encoding: bool = None) -> np.ndarray:
        """
        Encode individual tokens to hypervectors.
        
        Args:
            tokens: List of tokens to encode
            apply_positional_encoding: Whether to apply positional encoding
                                     (defaults to instance setting)
            
        Returns:
            Array of shape (num_tokens, dimension) with token hypervectors
        """
        if not tokens:
            return np.empty((0, self.dimension), dtype=np.int8)
        
        if apply_positional_encoding is None:
            apply_positional_encoding = self.use_positional_encoding
        
        # Get token vectors
        token_vectors = self._get_token_vectors(tokens)
        
        if len(token_vectors) == 0:
            return np.empty((0, self.dimension), dtype=np.int8)
        
        # Apply positional encoding if requested
        if apply_positional_encoding and self.positional_encoder is not None:
            positions = list(range(len(token_vectors)))
            token_vectors = self.positional_encoder.apply_positional_encoding(
                token_vectors, positions
            )
        
        return token_vectors
    
    def _get_token_vectors(self, tokens: List[str]) -> np.ndarray:
        """
        Get hypervectors for tokens from ItemMemory.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Array of token vectors, excluding tokens not found in ItemMemory
        """
        vectors = []
        for token in tokens:
            vector = self.item_memory.get_vector(token)
            if vector is not None:
                vectors.append(vector)
        
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.int8)
        
        return np.array(vectors)
    
    def _get_tf_idf_weights(self, 
                           tokens: List[str], 
                           tf_idf_dict: Dict[str, float],
                           normalize: bool = True) -> np.ndarray:
        """
        Get TF-IDF weights for tokens.
        
        Args:
            tokens: List of tokens (should already be filtered for valid tokens)
            tf_idf_dict: TF-IDF weights dictionary
            normalize: Whether to normalize weights
            
        Returns:
            Array of weights for each token
        """
        weights = []
        for token in tokens:
            weight = tf_idf_dict.get(token, 1.0)
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float32)
        
        if normalize and len(weights) > 0:
            # Normalize to [0, 1] range
            min_weight = np.min(weights)
            max_weight = np.max(weights)
            if max_weight > min_weight:
                weights = (weights - min_weight) / (max_weight - min_weight)
            
            # Scale to [0.1, 1.0] to avoid zero weights
            weights = 0.1 + 0.9 * weights
        
        return weights
    
    def _apply_weights(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Apply TF-IDF weights to token vectors.
        
        Args:
            vectors: Token vectors of shape (num_tokens, dimension)
            weights: Weights array of shape (num_tokens,)
            
        Returns:
            Weighted vectors
        """
        if len(weights) != len(vectors):
            raise ValueError(f"Number of weights {len(weights)} doesn't match "
                           f"number of vectors {len(vectors)}")
        
        # Apply weights by scaling vectors
        # Convert to float for multiplication, then back to int8
        weighted_vectors = vectors.astype(np.float32) * weights.reshape(-1, 1)
        
        # Round and convert back to int8, preserving sign
        weighted_vectors = np.round(weighted_vectors).astype(np.int8)
        
        return weighted_vectors
    
    def _bundle_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Bundle vectors using sum + thresholding.
        
        Args:
            vectors: Array of vectors to bundle, shape (num_vectors, dimension)
            
        Returns:
            Bundled hypervector of shape (dimension,)
        """
        if len(vectors) == 0:
            return np.zeros(self.dimension, dtype=np.int8)
        
        if len(vectors) == 1:
            return vectors[0].copy()
        
        # Sum all vectors
        vector_sum = np.sum(vectors, axis=0, dtype=np.int32)
        
        # Apply thresholding
        if self.bundling_threshold == 0.0:
            # Majority vote: positive if sum > 0, negative otherwise
            bundled = np.where(vector_sum > 0, 1, -1).astype(np.int8)
        else:
            # Custom threshold
            bundled = np.where(vector_sum > self.bundling_threshold, 1, -1).astype(np.int8)
        
        return bundled
    
    def _encode_position_vectors(self, positions: List[int]) -> np.ndarray:
        """
        Encode position vectors separately.
        """
        if not self.positional_encoder:
            raise ValueError("PositionalEncoder is required for position encoding")
        
        # Get position vectors for each position
        position_vectors = []
        for pos in positions:
            # Use the position_vectors attribute directly
            if pos < 0 or pos >= self.positional_encoder.max_positions:
                raise IndexError(f"Position {pos} is out of range [0, {self.positional_encoder.max_positions})")
            pos_vector = self.positional_encoder.position_vectors[pos]
            position_vectors.append(pos_vector)
        
        if not position_vectors:
            raise ValueError("No position vectors generated")
        
        # Bundle position vectors
        position_vector = self._bundle_vectors(np.array(position_vectors))
        
        return position_vector
    
    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two hypervectors.
        
        Args:
            vector1: First hypervector
            vector2: Second hypervector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector shapes don't match: {vector1.shape} vs {vector2.shape}")
        
        # For bipolar vectors, cosine similarity is just dot product / (norm1 * norm2)
        # Since all elements are ±1, norms are sqrt(dimension)
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def hamming_distance(self, vector1: np.ndarray, vector2: np.ndarray) -> int:
        """
        Calculate Hamming distance between two hypervectors.
        
        Args:
            vector1: First hypervector
            vector2: Second hypervector
            
        Returns:
            Hamming distance (number of differing positions)
        """
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector shapes don't match: {vector1.shape} vs {vector2.shape}")
        
        return int(np.sum(vector1 != vector2))
    
    def get_encoding_statistics(self, preprocessed_doc: PreprocessedDocument) -> Dict[str, Any]:
        """
        Get statistics about document encoding.
        
        Args:
            preprocessed_doc: Preprocessed document
            
        Returns:
            Dictionary with encoding statistics
        """
        total_tokens = len(preprocessed_doc.normalized_tokens)
        valid_tokens = sum(1 for token in preprocessed_doc.normalized_tokens 
                          if self.item_memory.has_word(token))
        
        coverage = valid_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Get TF-IDF statistics
        tf_idf_weights = list(preprocessed_doc.tf_idf_weights.values())
        
        return {
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "coverage": coverage,
            "avg_tf_idf_weight": np.mean(tf_idf_weights) if tf_idf_weights else 0.0,
            "max_tf_idf_weight": np.max(tf_idf_weights) if tf_idf_weights else 0.0,
            "min_tf_idf_weight": np.min(tf_idf_weights) if tf_idf_weights else 0.0,
            "use_positional_encoding": self.use_positional_encoding,
            "bundling_threshold": self.bundling_threshold
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get general encoder statistics.
        
        Returns:
            Dictionary with encoder statistics
        """
        return {
            "dimension": self.dimension,
            "use_positional_encoding": self.use_positional_encoding,
            "bundling_threshold": self.bundling_threshold,
            "random_seed": self.random_seed,
            "item_memory_size": self.item_memory.size(),
            "positional_encoder_available": self.positional_encoder is not None
        }