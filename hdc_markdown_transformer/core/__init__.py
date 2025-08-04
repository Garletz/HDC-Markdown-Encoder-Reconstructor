"""Core components and data models for HDC Markdown Transformer."""

from .models import (
    PreprocessedDocument,
    MarkdownStructure,
    SimilarityResult,
    EvaluationMetrics,
    Header,
    ListItem,
    Link,
    CodeBlock,
    Emphasis
)

from .interfaces import (
    TokenizerInterface,
    VectorDatabaseInterface,
    LLMInterface
)

__all__ = [
    "PreprocessedDocument",
    "MarkdownStructure",
    "SimilarityResult", 
    "EvaluationMetrics",
    "Header",
    "ListItem",
    "Link",
    "CodeBlock",
    "Emphasis",
    "TokenizerInterface",
    "VectorDatabaseInterface",
    "LLMInterface"
]