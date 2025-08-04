import os
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

def save_hypervector_with_token_count(hypervector: np.ndarray, token_count: int, save_dir: str = "encoded_vectors", prefix: str = "encoded") -> str:
    """
    Save a hypervector to a .npy file with the token count in the filename.

    Args:
        hypervector: The numpy array to save
        token_count: Number of valid tokens used to create this hypervector
        save_dir: Directory to save the file
        prefix: Prefix for the filename

    Returns:
        The path to the saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}_{token_count}.npy"
    filepath = os.path.join(save_dir, filename)
    np.save(filepath, hypervector)
    return filepath


def save_dual_hypervectors(content_vector: np.ndarray, 
                          position_vector: np.ndarray,
                          token_count: int,
                          output_dir: str = "encoded_vectors") -> tuple[str, str]:
    """
    Save dual hypervectors (content + position) to separate files.
    
    Args:
        content_vector: Content hypervector
        position_vector: Position hypervector
        token_count: Number of tokens in the document
        output_dir: Directory to save files
        
    Returns:
        Tuple of (content_file_path, position_file_path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    content_file_path = os.path.join(output_dir, f"encoded_{token_count}_content.npy")
    position_file_path = os.path.join(output_dir, f"encoded_{token_count}_position.npy")
    
    np.save(content_file_path, content_vector)
    np.save(position_file_path, position_vector)
    
    logger.info(f"Saved dual hypervectors: content={content_file_path}, position={position_file_path} (valid_tokens={token_count})")
    
    return content_file_path, position_file_path

def save_dual_hypervectors_with_pairs(content_vector: np.ndarray, 
                                     position_vector: np.ndarray,
                                     pair_vectors: List[np.ndarray],
                                     token_count: int,
                                     output_dir: str = "encoded_vectors") -> tuple[str, str, str]:
    """
    Save dual hypervectors with pair vectors for order deduction.
    
    Args:
        content_vector: Content hypervector
        position_vector: Position hypervector
        pair_vectors: List of (word, position) pair vectors
        token_count: Number of tokens in the document
        output_dir: Directory to save files
        
    Returns:
        Tuple of (content_file_path, position_file_path, pairs_file_path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    content_file_path = os.path.join(output_dir, f"encoded_{token_count}_content.npy")
    position_file_path = os.path.join(output_dir, f"encoded_{token_count}_position.npy")
    pairs_file_path = os.path.join(output_dir, f"encoded_{token_count}_pairs.npy")
    
    np.save(content_file_path, content_vector)
    np.save(position_file_path, position_vector)
    np.save(pairs_file_path, np.array(pair_vectors))
    
    logger.info(f"Saved dual hypervectors with pairs: content={content_file_path}, position={position_file_path}, pairs={pairs_file_path} (valid_tokens={token_count})")
    
    return content_file_path, position_file_path, pairs_file_path


def parse_token_count_from_filename(filename: str, prefix: str = "encoded") -> int:
    """
    Robustly parse the token count from any filename or path of the form 'encoded_<token_count>.npy' or 'encoded_<token_count>_content.npy'.
    Args:
        filename: The filename or path to parse
        prefix: The expected prefix (default: 'encoded')
    Returns:
        Extracted token count as int
    Raises:
        ValueError if the filename does not match the expected pattern
    """
    import re, os
    base = os.path.basename(filename)
    # Accept any case for prefix and extension
    # Pattern for both: encoded_<token_count>.npy and encoded_<token_count>_content.npy
    pattern = rf"{prefix}_(\d+)(?:_content|_position)?\.npy$"
    match = re.search(pattern, base, re.IGNORECASE)
    if not match:
        raise ValueError(f"Filename {filename} does not match pattern {prefix}_<token_count>.npy or {prefix}_<token_count>_content.npy")
    return int(match.group(1))
