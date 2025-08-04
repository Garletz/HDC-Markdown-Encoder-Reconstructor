"""Tokenizer implementations for different backends."""

import re
import string
from typing import List, Optional
from abc import ABC, abstractmethod

from ..core.interfaces import TokenizerInterface


class BaseTokenizer(TokenizerInterface):
    """Base tokenizer with common normalization functionality."""
    
    def __init__(self, remove_punctuation: bool = True, lowercase: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def normalize(self, tokens: List[str]) -> List[str]:
        """Normalize tokens with common preprocessing steps."""
        normalized = []
        
        for token in tokens:
            # Convert to lowercase
            if self.lowercase:
                token = token.lower()
            
            # Remove punctuation
            if self.remove_punctuation:
                token = self.punctuation_pattern.sub('', token)
            
            # Skip empty tokens
            if token.strip():
                normalized.append(token.strip())
        
        return normalized


class SimpleTokenizer(BaseTokenizer):
    """Simple whitespace-based tokenizer."""
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text by splitting on whitespace."""
        # Normalize whitespace first
        text = self.whitespace_pattern.sub(' ', text.strip())
        return text.split()


class SpacyTokenizer(BaseTokenizer):
    """SpaCy-based tokenizer."""
    
    def __init__(self, model: str = "en_core_web_sm", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self._nlp = None
    
    @property
    def nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.model_name)
            except ImportError:
                raise ImportError("spaCy is not installed. Install with: pip install spacy")
            except OSError:
                raise OSError(f"spaCy model '{self.model_name}' not found. Install with: python -m spacy download {self.model_name}")
        return self._nlp
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy."""
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]


class NLTKTokenizer(BaseTokenizer):
    """NLTK-based tokenizer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            import nltk
            # Try to use punkt tokenizer, download if not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except ImportError:
            raise ImportError("NLTK is not installed. Install with: pip install nltk")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK word tokenizer."""
        import nltk
        return nltk.word_tokenize(text)


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece-based tokenizer."""
    
    def __init__(self, model_path: Optional[str] = None, vocab_size: int = 8000, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.vocab_size = vocab_size
        self._sp = None
    
    @property
    def sp(self):
        """Lazy load SentencePiece model."""
        if self._sp is None:
            try:
                import sentencepiece as spm
                self._sp = spm.SentencePieceProcessor()
                
                if self.model_path:
                    self._sp.load(self.model_path)
                else:
                    # Use a simple character-based model for demo
                    # In practice, you'd train a proper model
                    raise ValueError("SentencePiece model path is required")
                    
            except ImportError:
                raise ImportError("SentencePiece is not installed. Install with: pip install sentencepiece")
        return self._sp
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        return self.sp.encode_as_pieces(text)


def create_tokenizer(tokenizer_type: str, **kwargs) -> TokenizerInterface:
    """Factory function to create tokenizer instances."""
    tokenizer_map = {
        'simple': SimpleTokenizer,
        'spacy': SpacyTokenizer,
        'nltk': NLTKTokenizer,
        'sentencepiece': SentencePieceTokenizer,
    }
    
    if tokenizer_type not in tokenizer_map:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available: {list(tokenizer_map.keys())}")
    
    return tokenizer_map[tokenizer_type](**kwargs)