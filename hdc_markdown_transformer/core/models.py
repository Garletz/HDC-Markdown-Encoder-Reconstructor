"""Core data models for HDC Markdown Transformer."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Header:
    """Represents a markdown header."""
    level: int
    text: str
    position: int


@dataclass
class ListItem:
    """Represents a markdown list item."""
    text: str
    level: int
    ordered: bool
    position: int


@dataclass
class Link:
    """Represents a markdown link."""
    text: str
    url: str
    position: int


@dataclass
class CodeBlock:
    """Represents a markdown code block."""
    content: str
    language: Optional[str]
    position: int


@dataclass
class Emphasis:
    """Represents markdown emphasis (bold, italic)."""
    text: str
    type: str  # 'bold', 'italic', 'bold_italic'
    position: int


@dataclass
class MarkdownStructure:
    """Represents the structural elements of a markdown document."""
    headers: List[Header]
    lists: List[ListItem]
    links: List[Link]
    code_blocks: List[CodeBlock]
    emphasis: List[Emphasis]


@dataclass
class PreprocessedDocument:
    """Represents a preprocessed markdown document ready for HDC encoding."""
    tokens: List[str]
    normalized_tokens: List[str]
    structure: MarkdownStructure
    positional_info: List[int]
    tf_idf_weights: Dict[str, float]
    original_content: str


@dataclass
class SimilarityResult:
    """Represents a similarity search result from vector database."""
    token: str
    similarity_score: float
    vector: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class EvaluationMetrics:
    """Represents evaluation metrics for reconstruction quality."""
    vector_similarity: float
    bleu_score: float
    rouge_score: float
    structure_preservation: float
    semantic_coherence: float
    
    def overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        return (
            self.vector_similarity * 0.3 +
            self.bleu_score * 0.2 +
            self.rouge_score * 0.2 +
            self.structure_preservation * 0.2 +
            self.semantic_coherence * 0.1
        )