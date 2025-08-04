"""Main markdown preprocessing functionality."""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import math

from ..core.models import PreprocessedDocument, MarkdownStructure
from ..core.interfaces import TokenizerInterface
from .tokenizers import create_tokenizer
from .tfidf import SingleDocumentTFIDF


class MarkdownPreprocessor:
    """Main preprocessor for markdown documents with tokenization and normalization."""
    
    def __init__(self, 
                 tokenizer_type: str = "simple",
                 tokenizer_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer_type: Type of tokenizer to use ('simple', 'spacy', 'nltk', 'sentencepiece')
            tokenizer_kwargs: Additional arguments for tokenizer initialization
        """
        self.tokenizer_type = tokenizer_type
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = create_tokenizer(tokenizer_type, **self.tokenizer_kwargs)
        self.structure_extractor = MarkdownStructureExtractor()
    
    def preprocess(self, markdown_content: str) -> PreprocessedDocument:
        """
        Preprocess a markdown document.
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            PreprocessedDocument with tokens, structure, and metadata
        """
        # Extract structure first (before tokenization to preserve positions)
        structure = self.structure_extractor.extract_structure(markdown_content)
        
        # Remove markdown syntax for tokenization (keep content only)
        clean_text = self._remove_markdown_syntax(markdown_content)
        
        # Tokenize the clean text
        tokens = self.tokenizer.tokenize(clean_text)
        
        # Normalize tokens
        normalized_tokens = self.tokenizer.normalize(tokens)
        
        # Calculate positional information
        positional_info = self._calculate_positions(tokens, markdown_content)
        
        # Calculate TF-IDF weights
        tf_idf_weights = self._calculate_tf_idf_weights(normalized_tokens)
        
        return PreprocessedDocument(
            tokens=tokens,
            normalized_tokens=normalized_tokens,
            structure=structure,
            positional_info=positional_info,
            tf_idf_weights=tf_idf_weights,
            original_content=markdown_content
        )
    
    def _remove_markdown_syntax(self, content: str) -> str:
        """Remove markdown syntax while preserving text content."""
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Remove emphasis (bold/italic)
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
        content = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', content)
        
        # Remove links but keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Remove code blocks
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Remove list markers
        content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _calculate_positions(self, tokens: List[str], original_content: str) -> List[int]:
        """Calculate approximate positions of tokens in original content."""
        positions = []
        content_lower = original_content.lower()
        current_pos = 0
        
        for token in tokens:
            # Find token position (approximate)
            token_pos = content_lower.find(token.lower(), current_pos)
            if token_pos != -1:
                positions.append(token_pos)
                current_pos = token_pos + len(token)
            else:
                # If exact match not found, use current position
                positions.append(current_pos)
        
        return positions
    
    def _calculate_tf_idf_weights(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF weights for tokens using single-document approach.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            Dictionary mapping tokens to their TF-IDF weights
        """
        return SingleDocumentTFIDF.calculate_tf_idf_normalized(tokens)


class MarkdownStructureExtractor:
    """Extracts structural elements from markdown documents."""
    
    def extract_structure(self, content: str) -> MarkdownStructure:
        """Extract all structural elements from markdown content."""
        return MarkdownStructure(
            headers=self._extract_headers(content),
            lists=self._extract_lists(content),
            links=self._extract_links(content),
            code_blocks=self._extract_code_blocks(content),
            emphasis=self._extract_emphasis(content)
        )
    
    def _extract_headers(self, content: str) -> List:
        """Extract markdown headers."""
        from ..core.models import Header
        
        headers = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # ATX headers (# ## ###)
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                position = sum(len(l) + 1 for l in lines[:i])  # Approximate position
                headers.append(Header(level=level, text=text, position=position))
        
        return headers
    
    def _extract_lists(self, content: str) -> List:
        """Extract markdown list items."""
        from ..core.models import ListItem
        
        lists = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Unordered lists
            match = re.match(r'^(\s*)([-*+])\s+(.+)$', line)
            if match:
                indent = len(match.group(1))
                level = indent // 2  # Approximate nesting level
                text = match.group(3).strip()
                position = sum(len(l) + 1 for l in lines[:i])
                lists.append(ListItem(text=text, level=level, ordered=False, position=position))
            
            # Ordered lists
            match = re.match(r'^(\s*)(\d+)\.\s+(.+)$', line)
            if match:
                indent = len(match.group(1))
                level = indent // 2
                text = match.group(3).strip()
                position = sum(len(l) + 1 for l in lines[:i])
                lists.append(ListItem(text=text, level=level, ordered=True, position=position))
        
        return lists
    
    def _extract_links(self, content: str) -> List:
        """Extract markdown links."""
        from ..core.models import Link
        
        links = []
        # Find all markdown links [text](url)
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
            text = match.group(1)
            url = match.group(2)
            position = match.start()
            links.append(Link(text=text, url=url, position=position))
        
        return links
    
    def _extract_code_blocks(self, content: str) -> List:
        """Extract markdown code blocks."""
        from ..core.models import CodeBlock
        
        code_blocks = []
        
        # Fenced code blocks ```
        for match in re.finditer(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL):
            language = match.group(1)
            code_content = match.group(2)
            position = match.start()
            code_blocks.append(CodeBlock(content=code_content, language=language, position=position))
        
        # Inline code `code`
        for match in re.finditer(r'`([^`]+)`', content):
            code_content = match.group(1)
            position = match.start()
            code_blocks.append(CodeBlock(content=code_content, language=None, position=position))
        
        return code_blocks
    
    def _extract_emphasis(self, content: str) -> List:
        """Extract markdown emphasis (bold, italic)."""
        from ..core.models import Emphasis
        
        emphasis = []
        
        # Bold **text** or __text__
        for match in re.finditer(r'\*\*([^*]+)\*\*', content):
            text = match.group(1)
            position = match.start()
            emphasis.append(Emphasis(text=text, type='bold', position=position))
        
        for match in re.finditer(r'__([^_]+)__', content):
            text = match.group(1)
            position = match.start()
            emphasis.append(Emphasis(text=text, type='bold', position=position))
        
        # Italic *text* or _text_ (but not bold)
        # Use negative lookbehind/lookahead to avoid matching bold patterns
        for match in re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', content):
            text = match.group(1)
            position = match.start()
            emphasis.append(Emphasis(text=text, type='italic', position=position))
        
        for match in re.finditer(r'(?<!_)_([^_]+)_(?!_)', content):
            text = match.group(1)
            position = match.start()
            emphasis.append(Emphasis(text=text, type='italic', position=position))
        
        return emphasis