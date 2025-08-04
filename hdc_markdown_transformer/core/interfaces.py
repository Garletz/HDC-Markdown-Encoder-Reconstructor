"""Abstract interfaces for extensibility and dependency injection."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .models import SimilarityResult, PreprocessedDocument


class TokenizerInterface(ABC):
    """Abstract interface for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into list of tokens."""
        pass
    
    @abstractmethod
    def normalize(self, tokens: List[str]) -> List[str]:
        """Normalize tokens (lowercase, punctuation removal, etc.)."""
        pass


class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def store_vectors(self, vectors: Dict[str, np.ndarray]) -> None:
        """Store vectors with their identifiers."""
        pass
    
    @abstractmethod
    def similarity_search(self, query_vector: np.ndarray, k: int) -> List[SimilarityResult]:
        """Perform K-NN similarity search."""
        pass
    
    @abstractmethod
    def update_vector(self, identifier: str, vector: np.ndarray) -> None:
        """Update an existing vector."""
        pass
    
    @abstractmethod
    def delete_vector(self, identifier: str) -> None:
        """Delete a vector by identifier."""
        pass
    
    @abstractmethod
    def get_vector(self, identifier: str) -> Optional[np.ndarray]:
        """Retrieve a vector by identifier."""
        pass


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def reconstruct_markdown(self, 
                           candidates: List[str], 
                           structure_info: Dict[str, Any],
                           context: str = "") -> str:
        """Reconstruct markdown from candidates and structure information."""
        pass


class ReconstructorInterface(ABC):
    """Abstract interface for markdown reconstructors."""
    
    @abstractmethod
    def reconstruct_markdown(self, 
                           candidates: List[SimilarityResult], 
                           structure: Any,  # MarkdownStructure
                           context: str = "") -> Any:  # ReconstructionResult
        """Reconstruct markdown from similarity candidates and structure."""
        pass


class HDCEncoderInterface(ABC):
    """Abstract interface for HDC encoders."""
    
    @abstractmethod
    def encode_document(self, preprocessed_doc: PreprocessedDocument) -> np.ndarray:
        """Encode a preprocessed document into HDC hypervector."""
        pass
    
    @abstractmethod
    def encode_tokens(self, tokens: List[str]) -> List[np.ndarray]:
        """Encode individual tokens into HDC hypervectors."""
        pass