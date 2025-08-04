"""Item Memory implementation for HDC dictionary management."""

import os
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
import numpy as np
import requests
from urllib.parse import urljoin

from ..core.interfaces import VectorDatabaseInterface
from ..config.configuration_manager import Configuration


logger = logging.getLogger(__name__)


class ItemMemory:
    """
    HDC Item Memory for storing word-to-hypervector mappings.
    
    This class manages a comprehensive linguistic dictionary where each word
    is mapped to a deterministic hypervector. It supports loading from various
    sources, persistence, and integration with vector databases.
    """
    
    def __init__(self, dimension: int, random_seed: int = 42):
        """
        Initialize ItemMemory.
        
        Args:
            dimension: Dimension of hypervectors
            random_seed: Random seed for deterministic vector generation
        """
        self.dimension = dimension
        self.random_seed = random_seed
        self.dictionary: Dict[str, np.ndarray] = {}
        self.rng = np.random.RandomState(random_seed)
        
        logger.info(f"Initialized ItemMemory with dimension={dimension}, seed={random_seed}")
    
    def create_from_word_list(self, word_list: List[str]) -> None:
        """
        Create hypervectors for a list of words.
        
        Args:
            word_list: List of words to create hypervectors for
        """
        logger.info(f"Creating hypervectors for {len(word_list)} words")
        
        # Reset RNG to ensure deterministic generation
        self.rng = np.random.RandomState(self.random_seed)
        
        for word in word_list:
            if word not in self.dictionary:
                # Generate random bipolar hypervector (-1 or +1)
                vector = self.rng.choice([-1, 1], size=self.dimension)
                self.dictionary[word] = vector.astype(np.int8)
        
        logger.info(f"Created {len(self.dictionary)} hypervectors")
        # Diagnostic : afficher les 10 premiers mots du dictionnaire
        preview = list(self.dictionary.keys())[:10]
        logger.info(f"[DIAG] Premiers mots dans le dictionnaire HDC : {preview}")
    
    def create_from_linguistic_dictionary(self, size: int = 100000) -> None:
        """
        Create comprehensive HDC dictionary from linguistic sources.
        
        This method loads a large linguistic dictionary and creates hypervectors
        for each word. It attempts to load from multiple sources to reach the
        target size.
        
        Args:
            size: Target dictionary size (default 100k words)
        """
        logger.info(f"Creating linguistic dictionary with target size: {size}")
        
        word_set: Set[str] = set()
        
        # Try to load from multiple sources
        sources = [
            self._load_english_words_alpha,
            self._load_nltk_words,
            self._load_wordnet_words,
            self._load_common_words
        ]
        
        for source_func in sources:
            try:
                words = source_func()
                word_set.update(words)
                logger.info(f"Loaded {len(words)} words from {source_func.__name__}")
                
                if len(word_set) >= size:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load from {source_func.__name__}: {e}")
                continue
        
        # Convert to sorted list for deterministic ordering
        word_list = sorted(list(word_set))[:size]
        
        logger.info(f"Final word list size: {len(word_list)}")
        self.create_from_word_list(word_list)
    
    def _load_english_words_alpha(self) -> List[str]:
        """Load words from english-words-alpha dataset."""
        try:
            # Try to download from GitHub
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            words = [line.strip().lower() for line in response.text.split('\n') if line.strip()]
            return [word for word in words if word.isalpha() and len(word) > 1]
            
        except Exception as e:
            logger.warning(f"Failed to download english-words-alpha: {e}")
            return []
    
    def _load_nltk_words(self) -> List[str]:
        """Load words from NLTK corpus."""
        try:
            import nltk
            from nltk.corpus import words
            
            # Download words corpus if not available
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words', quiet=True)
            
            word_list = words.words()
            return [word.lower() for word in word_list if word.isalpha() and len(word) > 1]
            
        except ImportError:
            logger.warning("NLTK not available for word loading")
            return []
        except Exception as e:
            logger.warning(f"Failed to load NLTK words: {e}")
            return []
    
    def _load_wordnet_words(self) -> List[str]:
        """Load words from WordNet."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download wordnet if not available
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            
            words = set()
            for synset in wordnet.all_synsets():
                for lemma in synset.lemmas():
                    word = lemma.name().lower().replace('_', ' ')
                    if word.isalpha() and len(word) > 1:
                        words.add(word)
            
            return list(words)
            
        except ImportError:
            logger.warning("NLTK not available for WordNet loading")
            return []
        except Exception as e:
            logger.warning(f"Failed to load WordNet words: {e}")
            return []
    
    def _load_common_words(self) -> List[str]:
        """Load common English words as fallback."""
        # Basic fallback word list
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "water", "long", "find", "here", "thing", "great", "man",
            "world", "life", "still", "hand", "high", "year", "government", "person", "day",
            "part", "child", "eye", "woman", "place", "work", "week", "case", "point",
            "company", "right", "program", "question", "fact", "group", "problem", "area"
        ]
        
        # Extend with more words
        extended_words = []
        for word in common_words:
            extended_words.append(word)
            # Add some variations
            if len(word) > 3:
                extended_words.append(word + "s")  # plural
                extended_words.append(word + "ed")  # past tense
                extended_words.append(word + "ing")  # present participle
        
        return extended_words
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get hypervector for a word.
        
        Args:
            word: Word to get vector for
            
        Returns:
            Hypervector for the word, or None if not found
        """
        return self.dictionary.get(word.lower())
    
    def has_word(self, word: str) -> bool:
        """
        Check if word exists in dictionary.
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists, False otherwise
        """
        return word.lower() in self.dictionary
    
    def get_words(self) -> List[str]:
        """
        Get all words in dictionary.
        
        Returns:
            List of all words
        """
        return list(self.dictionary.keys())
    
    def size(self) -> int:
        """
        Get dictionary size.
        
        Returns:
            Number of words in dictionary
        """
        return len(self.dictionary)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save dictionary to .npz file for persistence.
        
        Args:
            filepath: Path to save file
        """
        logger.info(f"Saving ItemMemory to {filepath}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare data for saving
        words = list(self.dictionary.keys())
        vectors = np.array([self.dictionary[word] for word in words])
        
        # Save with metadata
        np.savez_compressed(
            filepath,
            words=words,
            vectors=vectors,
            dimension=self.dimension,
            random_seed=self.random_seed,
            size=len(words)
        )
        
        logger.info(f"Saved {len(words)} word vectors to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load dictionary from .npz file.
        
        Args:
            filepath: Path to load file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ItemMemory file not found: {filepath}")
        
        logger.info(f"Loading ItemMemory from {filepath}")
        
        try:
            data = np.load(filepath, allow_pickle=True)
            # Validate metadata
            if data['dimension'] != self.dimension:
                raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {data['dimension']}")
            if data['random_seed'] != self.random_seed:
                logger.warning(f"Random seed mismatch: expected {self.random_seed}, got {data['random_seed']}")
            # Load dictionary
            words = data['words']
            vectors = data['vectors']
            self.dictionary = {}
            for word, vector in zip(words, vectors):
                self.dictionary[word] = vector
            logger.info(f"Loaded {len(self.dictionary)} word vectors from {filepath}")
            # Contrôle : afficher les 10 premiers tokens du dictionnaire pour diagnostic
            preview = list(self.dictionary.keys())[:10]
            logger.info(f"[DIAG] Premiers mots du dictionnaire chargé : {preview}")
        except Exception as e:
            raise ValueError(f"Failed to load ItemMemory file: {e}")

    def store_in_vector_db(self, vector_db):
        """
        Store dictionary in vector database for fast K-NN search.
        
        Args:
            vector_db: Vector database instance
        """
        logger.info(f"Storing {len(self.dictionary)} vectors in vector database")
        # Convert int8 vectors to float32 for vector database
        vectors_float = {}
        for word, vector in self.dictionary.items():
            vectors_float[word] = vector.astype(np.float32)
        vector_db.store_vectors(vectors_float)
        logger.info("Successfully stored vectors in vector database")

    def create_backup(self, backup_dir: str = "./backups") -> str:
        """
        Create a backup of the current dictionary.
        
        Args:
            backup_dir: Directory to store backup
        Returns:
            Path to backup file
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"item_memory_backup_{timestamp}.npz"
        backup_path = os.path.join(backup_dir, backup_filename)
        self.save_to_file(backup_path)
        return backup_path

    def get_statistics(self) -> Dict[str, any]:
        """
        Get dictionary statistics.
        Returns:
            Dictionary with statistics
        """
        if not self.dictionary:
            return {"size": 0, "dimension": self.dimension}
        # Calculate some basic statistics
        word_lengths = [len(word) for word in self.dictionary.keys()]
        return {
            "size": len(self.dictionary),
            "dimension": self.dimension,
            "random_seed": self.random_seed,
            "avg_word_length": np.mean(word_lengths),
            "min_word_length": min(word_lengths),
            "max_word_length": max(word_lengths),
            "memory_usage_mb": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.dictionary:
            return 0.0
        # Estimate: each vector is dimension * 1 byte (int8) + word string overhead
        vector_size = self.dimension * 1  # int8
        avg_word_length = np.mean([len(word) for word in self.dictionary.keys()])
        string_overhead = avg_word_length * 1  # approximate
        total_bytes = len(self.dictionary) * (vector_size + string_overhead)
        return total_bytes / (1024 * 1024)  # Convert to MB