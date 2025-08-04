"""
HDC Markdown Transformer

A system that encodes Markdown files into HDC hypervectors and uses LLM 
to reconstruct coherent documents from vectorial representations.
"""

__version__ = "0.1.0"
__author__ = "HDC Markdown Transformer Team"

from .core.models import (
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

__all__ = [
    "PreprocessedDocument",
    "MarkdownStructure", 
    "SimilarityResult",
    "EvaluationMetrics",
    "Header",
    "ListItem",
    "Link", 
    "CodeBlock",
    "Emphasis"
]