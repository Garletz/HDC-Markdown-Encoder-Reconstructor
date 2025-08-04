"""Positional encoding for sequence-aware HDC hypervector encoding."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PositionalEncoder:
    """
    HDC Positional Encoder for sequence-aware hypervector encoding.
    
    This class implements position-based vector transformations to encode
    positional information into hypervectors. It supports permutation and
    rotation operations to create position-dependent representations.
    """
    
    def __init__(self, dimension: int, max_positions: int = 10000, random_seed: int = 42):
        """
        Initialize PositionalEncoder.
        
        Args:
            dimension: Dimension of hypervectors
            max_positions: Maximum number of positions to support
            random_seed: Random seed for deterministic position vector generation
        """
        self.dimension = dimension
        self.max_positions = max_positions
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Pre-generate position vectors for efficiency
        self.position_vectors = self._generate_position_vectors()
        
        logger.info(f"Initialized PositionalEncoder with dimension={dimension}, "
                   f"max_positions={max_positions}, seed={random_seed}")
    
    def _generate_position_vectors(self) -> np.ndarray:
        """
        Generate deterministic position vectors using permutation patterns.
        
        Returns:
            Array of shape (max_positions, dimension) containing position vectors
        """
        logger.info(f"Generating {self.max_positions} position vectors")
        
        # Reset RNG for deterministic generation
        self.rng = np.random.RandomState(self.random_seed)
        
        position_vectors = np.zeros((self.max_positions, self.dimension), dtype=np.int8)
        
        # Base identity permutation
        base_permutation = np.arange(self.dimension)
        
        for pos in range(self.max_positions):
            if pos == 0:
                # Position 0 is identity (no transformation)
                position_vectors[pos] = np.ones(self.dimension, dtype=np.int8)
            else:
                # Create position-specific permutation
                # Use a combination of rotation and random permutation
                permutation = self._create_position_permutation(pos, base_permutation.copy())
                
                # Generate bipolar vector and apply permutation
                base_vector = self.rng.choice([-1, 1], size=self.dimension)
                position_vectors[pos] = base_vector[permutation].astype(np.int8)
        
        return position_vectors
    
    def _create_position_permutation(self, position: int, base_perm: np.ndarray) -> np.ndarray:
        """
        Create position-specific permutation pattern.
        
        Args:
            position: Position index
            base_perm: Base permutation array
            
        Returns:
            Position-specific permutation
        """
        # Use position as seed modifier for deterministic but varied permutations
        pos_rng = np.random.RandomState(self.random_seed + position)
        
        # Apply rotation based on position
        rotation_amount = position % self.dimension
        rotated_perm = np.roll(base_perm, rotation_amount)
        
        # Add some controlled randomness for better distribution
        if position > 1:
            # Shuffle small segments to create more variation
            segment_size = max(2, self.dimension // 100)
            for i in range(0, self.dimension, segment_size):
                end_idx = min(i + segment_size, self.dimension)
                segment = rotated_perm[i:end_idx].copy()
                pos_rng.shuffle(segment)
                rotated_perm[i:end_idx] = segment
        
        return rotated_perm
    
    def encode_position(self, position: int) -> np.ndarray:
        """
        Get position encoding vector for a specific position.
        
        Args:
            position: Position index (0-based)
            
        Returns:
            Position encoding hypervector
            
        Raises:
            ValueError: If position exceeds max_positions
        """
        if position < 0:
            raise ValueError(f"Position must be non-negative, got {position}")
        
        if position >= self.max_positions:
            raise ValueError(f"Position {position} exceeds max_positions {self.max_positions}")
        
        return self.position_vectors[position].copy()
    
    def encode_sequence_positions(self, sequence_length: int, start_position: int = 0) -> np.ndarray:
        """
        Get position encoding vectors for a sequence.
        
        Args:
            sequence_length: Length of the sequence
            start_position: Starting position index
            
        Returns:
            Array of shape (sequence_length, dimension) with position vectors
            
        Raises:
            ValueError: If sequence extends beyond max_positions
        """
        end_position = start_position + sequence_length
        
        if end_position > self.max_positions:
            raise ValueError(f"Sequence extends beyond max_positions: "
                           f"{start_position}+{sequence_length} > {self.max_positions}")
        
        return self.position_vectors[start_position:end_position].copy()
    
    def apply_positional_encoding(self, 
                                 token_vectors: np.ndarray, 
                                 positions: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply positional encoding to token vectors.
        
        This method combines token vectors with their positional encodings
        using element-wise multiplication (binding operation in HDC).
        
        Args:
            token_vectors: Array of shape (sequence_length, dimension) with token vectors
            positions: Optional list of position indices. If None, uses sequential positions.
            
        Returns:
            Position-encoded vectors of same shape as input
            
        Raises:
            ValueError: If dimensions don't match or positions are invalid
        """
        if token_vectors.ndim != 2:
            raise ValueError(f"Expected 2D token_vectors, got shape {token_vectors.shape}")
        
        sequence_length, vector_dim = token_vectors.shape
        
        if vector_dim != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, "
                           f"got {vector_dim}")
        
        # Use sequential positions if not provided
        if positions is None:
            positions = list(range(sequence_length))
        
        if len(positions) != sequence_length:
            raise ValueError(f"Position list length {len(positions)} doesn't match "
                           f"sequence length {sequence_length}")
        
        # Get position vectors
        position_vectors = np.array([self.encode_position(pos) for pos in positions])
        
        # Apply positional encoding using element-wise multiplication (binding)
        # This is the standard HDC operation for combining vectors
        encoded_vectors = token_vectors * position_vectors
        
        return encoded_vectors.astype(np.int8)
    
    def permute_vector(self, vector: np.ndarray, position: int) -> np.ndarray:
        """
        Apply position-specific permutation to a vector.
        
        This is an alternative encoding method that uses permutation
        instead of multiplication for positional encoding.
        
        Args:
            vector: Input hypervector
            position: Position index
            
        Returns:
            Permuted vector
            
        Raises:
            ValueError: If position is invalid or vector dimension doesn't match
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, "
                           f"got {vector.shape[0]}")
        
        if position < 0 or position >= self.max_positions:
            raise ValueError(f"Invalid position {position}, must be in [0, {self.max_positions})")
        
        # Create position-specific permutation
        base_perm = np.arange(self.dimension)
        permutation = self._create_position_permutation(position, base_perm)
        
        # Apply permutation
        return vector[permutation]
    
    def rotate_vector(self, vector: np.ndarray, position: int) -> np.ndarray:
        """
        Apply position-specific rotation to a vector.
        
        This method rotates the vector elements based on the position,
        providing a simpler form of positional encoding.
        
        Args:
            vector: Input hypervector
            position: Position index
            
        Returns:
            Rotated vector
            
        Raises:
            ValueError: If vector dimension doesn't match
        """
        if vector.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, "
                           f"got {vector.shape[0]}")
        
        # Calculate rotation amount based on position
        rotation_amount = position % self.dimension
        
        # Apply rotation
        return np.roll(vector, rotation_amount)
    
    def get_relative_position_encoding(self, pos1: int, pos2: int) -> np.ndarray:
        """
        Get relative position encoding between two positions.
        
        This method computes the relative positional relationship
        between two positions using HDC operations.
        
        Args:
            pos1: First position
            pos2: Second position
            
        Returns:
            Relative position encoding vector
        """
        if pos1 < 0 or pos1 >= self.max_positions:
            raise ValueError(f"Invalid pos1 {pos1}")
        if pos2 < 0 or pos2 >= self.max_positions:
            raise ValueError(f"Invalid pos2 {pos2}")
        
        # Get position vectors
        vec1 = self.encode_position(pos1)
        vec2 = self.encode_position(pos2)
        
        # Compute relative encoding using XOR (unbinding operation)
        # This gives us the "difference" between positions
        relative_encoding = np.logical_xor(vec1 > 0, vec2 > 0).astype(np.int8)
        relative_encoding[relative_encoding == 0] = -1  # Convert to bipolar
        
        return relative_encoding
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the positional encoder.
        
        Returns:
            Dictionary with encoder statistics
        """
        return {
            "dimension": self.dimension,
            "max_positions": self.max_positions,
            "random_seed": self.random_seed,
            "memory_usage_mb": self._estimate_memory_usage(),
            "position_vectors_shape": self.position_vectors.shape
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Position vectors: max_positions * dimension * 1 byte (int8)
        position_vectors_size = self.max_positions * self.dimension * 1
        
        # Convert to MB
        return position_vectors_size / (1024 * 1024)

    def get_position_vector(self, position: int) -> np.ndarray:
        """
        Get the position vector for a specific position.
        
        Args:
            position: Position index
            
        Returns:
            Position vector for the given position
            
        Raises:
            IndexError: If position is out of range
        """
        if position < 0 or position >= self.max_positions:
            raise IndexError(f"Position {position} is out of range [0, {self.max_positions})")
        
        return self.position_vectors[position].copy()