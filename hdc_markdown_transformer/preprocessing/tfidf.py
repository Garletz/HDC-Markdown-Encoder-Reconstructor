"""TF-IDF calculation utilities for document weighting."""

import math
from typing import List, Dict, Set
from collections import Counter, defaultdict


class TFIDFCalculator:
    """Calculator for TF-IDF weights across multiple documents."""
    
    def __init__(self):
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.total_documents: int = 0
        self.vocabulary: Set[str] = set()
    
    def fit(self, documents: List[List[str]]) -> None:
        """
        Fit the TF-IDF calculator on a corpus of documents.
        
        Args:
            documents: List of documents, where each document is a list of tokens
        """
        self.document_frequencies.clear()
        self.vocabulary.clear()
        self.total_documents = len(documents)
        
        # Count document frequencies for each term
        for doc_tokens in documents:
            unique_tokens = set(doc_tokens)
            self.vocabulary.update(unique_tokens)
            
            for token in unique_tokens:
                self.document_frequencies[token] += 1
    
    def calculate_tf_idf(self, document_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF weights for a single document.
        
        Args:
            document_tokens: List of tokens in the document
            
        Returns:
            Dictionary mapping tokens to their TF-IDF weights
        """
        if not document_tokens or self.total_documents == 0:
            return {}
        
        # Calculate term frequencies (TF)
        token_counts = Counter(document_tokens)
        total_tokens = len(document_tokens)
        
        tf_idf_weights = {}
        
        for token, count in token_counts.items():
            # Term Frequency (TF)
            tf = count / total_tokens
            
            # Inverse Document Frequency (IDF)
            doc_freq = self.document_frequencies.get(token, 0)
            if doc_freq > 0:
                # Standard IDF formula: log(N / df)
                idf = math.log(self.total_documents / doc_freq)
            else:
                # Token not seen in training corpus, assign high IDF
                idf = math.log(self.total_documents + 1)
            
            # Ensure IDF is not negative (can happen with single document corpus)
            idf = max(idf, 0.1)
            
            # TF-IDF score
            tf_idf_weights[token] = tf * idf
        
        return tf_idf_weights
    
    def calculate_tf_idf_normalized(self, document_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate normalized TF-IDF weights (L2 normalization).
        
        Args:
            document_tokens: List of tokens in the document
            
        Returns:
            Dictionary mapping tokens to their normalized TF-IDF weights
        """
        tf_idf_weights = self.calculate_tf_idf(document_tokens)
        
        if not tf_idf_weights:
            return {}
        
        # L2 normalization
        norm = math.sqrt(sum(weight ** 2 for weight in tf_idf_weights.values()))
        
        if norm == 0:
            return tf_idf_weights
        
        normalized_weights = {
            token: weight / norm 
            for token, weight in tf_idf_weights.items()
        }
        
        return normalized_weights
    
    def get_top_terms(self, document_tokens: List[str], k: int = 10) -> List[tuple]:
        """
        Get top-k terms by TF-IDF weight for a document.
        
        Args:
            document_tokens: List of tokens in the document
            k: Number of top terms to return
            
        Returns:
            List of (token, tf_idf_weight) tuples, sorted by weight descending
        """
        tf_idf_weights = self.calculate_tf_idf(document_tokens)
        
        # Sort by TF-IDF weight descending
        sorted_terms = sorted(
            tf_idf_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_terms[:k]
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save the vocabulary and document frequencies to a file."""
        import json
        
        data = {
            'vocabulary': list(self.vocabulary),
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load vocabulary and document frequencies from a file."""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocabulary = set(data['vocabulary'])
        self.document_frequencies = defaultdict(int, data['document_frequencies'])
        self.total_documents = data['total_documents']


class SingleDocumentTFIDF:
    """Simple TF-IDF calculator for single documents (no corpus)."""
    
    @staticmethod
    def calculate_tf_idf(tokens: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF weights for a single document using document-internal statistics.
        
        This uses a simplified approach where IDF is based on term rarity within the document.
        
        Args:
            tokens: List of tokens in the document
            
        Returns:
            Dictionary mapping tokens to their TF-IDF weights
        """
        if not tokens:
            return {}
        
        # Calculate term frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        unique_tokens = len(token_counts)
        
        tf_idf_weights = {}
        
        for token, count in token_counts.items():
            # Term Frequency (TF)
            tf = count / total_tokens
            
            # Simplified IDF for single document: log(total_tokens / term_frequency)
            # This gives higher weights to rarer terms within the document
            if count == total_tokens:
                # Special case: if token appears in all positions, give it a base weight
                idf = 1.0
            else:
                idf = math.log(total_tokens / count)
            
            # TF-IDF score
            tf_idf_weights[token] = tf * idf
        
        return tf_idf_weights
    
    @staticmethod
    def calculate_tf_idf_normalized(tokens: List[str]) -> Dict[str, float]:
        """Calculate normalized TF-IDF weights for a single document."""
        tf_idf_weights = SingleDocumentTFIDF.calculate_tf_idf(tokens)
        
        if not tf_idf_weights:
            return {}
        
        # L2 normalization
        norm = math.sqrt(sum(weight ** 2 for weight in tf_idf_weights.values()))
        
        if norm == 0:
            return tf_idf_weights
        
        normalized_weights = {
            token: weight / norm 
            for token, weight in tf_idf_weights.items()
        }
        
        return normalized_weights