"""Configuration management for HDC Markdown Transformer."""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class HDCConfig:
    """HDC-specific configuration."""
    dimension: int = 10000
    random_seed: int = 42


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    type: str = "spacy"  # spacy, nltk, sentencepiece
    language: str = "en_core_web_sm"
    normalize_text: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True


@dataclass
class VectorDatabaseConfig:
    """Vector database configuration."""
    type: str = "pinecone"  # pinecone, weaviate, redis
    index_name: str = "hdc-markdown-dictionary"
    dimension: int = 10000
    metric: str = "cosine"
    api_key: Optional[str] = None
    environment: Optional[str] = None
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # openai, local
    model: str = "gpt-4-turbo"
    max_tokens: int = 2000
    temperature: float = 0.3
    api_key: Optional[str] = None
    local_model_path: Optional[str] = None


@dataclass
class DictionaryConfig:
    """Dictionary configuration."""
    source: str = "linguistic"
    size: int = 100000
    languages: list = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    batch_size: int = 100
    max_document_length: int = 50000
    enable_caching: bool = True
    cache_directory: str = "./cache"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: list = None
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["vector_similarity", "bleu_score", "rouge_score", "structure_preservation"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "hdc_transformer.log"


@dataclass
class ReconstructionConfig:
    max_candidates: int = 20
    min_confidence_threshold: float = 0.1
    include_similarity_scores: bool = False
    preserve_structure: bool = True
    filter_duplicates: bool = True
    rerank_candidates: bool = True
    max_retries: int = 2
    language: str = "en"
    document_type: str = "general"
    target_audience: str = "general"

@dataclass
class Configuration:
    """Main configuration class."""
    hdc: HDCConfig = None
    tokenizer: TokenizerConfig = None
    vector_database: VectorDatabaseConfig = None
    llm: LLMConfig = None
    dictionary: DictionaryConfig = None
    processing: ProcessingConfig = None
    evaluation: EvaluationConfig = None
    logging: LoggingConfig = None
    reconstruction: ReconstructionConfig = None
    
    def __post_init__(self):
        if self.hdc is None:
            self.hdc = HDCConfig()
        if self.reconstruction is None:
            self.reconstruction = ReconstructionConfig()
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig()
        if self.vector_database is None:
            self.vector_database = VectorDatabaseConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.dictionary is None:
            self.dictionary = DictionaryConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigurationManager:
    """Manages configuration loading, validation, and persistence."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config_path = config_path or "config/default.yaml"
        self.config = Configuration()
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration if file exists
        if os.path.exists(self.config_path):
            self.load_from_file(self.config_path)
        
        # Override with environment variables
        self._load_from_environment()
    
    def load_from_file(self, path: str) -> None:
        """Load configuration from YAML or JSON file.
        
        Args:
            path: Path to configuration file.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValueError: If configuration file is invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    data = yaml.safe_load(f)
                elif path.endswith('.json'):
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {path}")
            
            self._update_config_from_dict(data)
            self.config_path = path
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid configuration file format: {e}")
    
    def save_to_file(self, path: str) -> None:
        """Save configuration to YAML or JSON file.
        
        Args:
            path: Path where to save configuration.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config_dict = asdict(self.config)
        
        with open(path, 'w', encoding='utf-8') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {path}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        # OpenAI API
        if os.getenv('OPENAI_API_KEY'):
            self.config.llm.api_key = os.getenv('OPENAI_API_KEY')
        
        # Pinecone
        if os.getenv('PINECONE_API_KEY'):
            self.config.vector_database.api_key = os.getenv('PINECONE_API_KEY')
        if os.getenv('PINECONE_ENVIRONMENT'):
            self.config.vector_database.environment = os.getenv('PINECONE_ENVIRONMENT')
        
        # Weaviate
        if os.getenv('WEAVIATE_URL'):
            self.config.vector_database.url = os.getenv('WEAVIATE_URL')
        if os.getenv('WEAVIATE_API_KEY'):
            self.config.vector_database.api_key = os.getenv('WEAVIATE_API_KEY')
        
        # Redis
        if os.getenv('REDIS_HOST'):
            self.config.vector_database.host = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            self.config.vector_database.port = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            self.config.vector_database.password = os.getenv('REDIS_PASSWORD')
        
        # Local model
        if os.getenv('LOCAL_MODEL_PATH'):
            self.config.llm.local_model_path = os.getenv('LOCAL_MODEL_PATH')
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.config.logging.level = os.getenv('LOG_LEVEL')
    
    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'hdc' in data:
            self.config.hdc = HDCConfig(**data['hdc'])
        
        if 'tokenizer' in data:
            self.config.tokenizer = TokenizerConfig(**data['tokenizer'])
        
        if 'vector_database' in data:
            self.config.vector_database = VectorDatabaseConfig(**data['vector_database'])
        
        if 'llm' in data:
            self.config.llm = LLMConfig(**data['llm'])
        
        if 'dictionary' in data:
            self.config.dictionary = DictionaryConfig(**data['dictionary'])
        
        if 'processing' in data:
            self.config.processing = ProcessingConfig(**data['processing'])
        
        if 'evaluation' in data:
            self.config.evaluation = EvaluationConfig(**data['evaluation'])
        
        if 'logging' in data:
            self.config.logging = LoggingConfig(**data['logging'])
    
    def validate_config(self) -> bool:
        """Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise.
            
        Raises:
            ValueError: If configuration is invalid with detailed error message.
        """
        errors = []
        
        # Validate HDC configuration
        if self.config.hdc.dimension <= 0:
            errors.append("HDC dimension must be positive")
        if self.config.hdc.dimension < 1000:
            errors.append("HDC dimension should be at least 1000 for good performance")
        
        # Validate tokenizer
        valid_tokenizers = ["spacy", "nltk", "sentencepiece", "simple"]
        if self.config.tokenizer.type not in valid_tokenizers:
            errors.append(f"Invalid tokenizer type. Must be one of: {valid_tokenizers}")
        
        # Validate vector database
        valid_db_types = ["pinecone", "weaviate", "redis", "memory"]
        if self.config.vector_database.type not in valid_db_types:
            errors.append(f"Invalid vector database type. Must be one of: {valid_db_types}")
        
        if self.config.vector_database.dimension != self.config.hdc.dimension:
            errors.append("Vector database dimension must match HDC dimension")
        
        # Validate LLM configuration
        valid_providers = ["openai", "local", "gemini"]
        if self.config.llm.provider not in valid_providers:
            errors.append(f"Invalid LLM provider. Must be one of: {valid_providers}")
        
        if self.config.llm.provider == "openai" and not self.config.llm.api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        if self.config.llm.provider == "local" and not self.config.llm.local_model_path:
            errors.append("Local model path is required when using local provider")
        
        # Validate processing configuration
        if self.config.processing.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.config.processing.max_document_length <= 0:
            errors.append("Max document length must be positive")
        
        # Validate evaluation configuration
        if self.config.evaluation.similarity_threshold < 0 or self.config.evaluation.similarity_threshold > 1:
            errors.append("Similarity threshold must be between 0 and 1")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    def get_config(self) -> Configuration:
        """Get current configuration.
        
        Returns:
            Current configuration object.
        """
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values.
        
        Args:
            **kwargs: Configuration values to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = Configuration()
        self._load_from_environment()
    
    def get_summary(self) -> str:
        """Get configuration summary as string.
        
        Returns:
            Human-readable configuration summary.
        """
        summary = []
        summary.append("HDC Markdown Transformer Configuration")
        summary.append("=" * 40)
        summary.append(f"HDC Dimension: {self.config.hdc.dimension}")
        summary.append(f"Random Seed: {self.config.hdc.random_seed}")
        summary.append(f"Tokenizer: {self.config.tokenizer.type}")
        summary.append(f"Vector Database: {self.config.vector_database.type}")
        summary.append(f"LLM Provider: {self.config.llm.provider}")
        summary.append(f"Dictionary Size: {self.config.dictionary.size}")
        summary.append(f"Batch Size: {self.config.processing.batch_size}")
        summary.append(f"Caching: {'Enabled' if self.config.processing.enable_caching else 'Disabled'}")
        
        return "\n".join(summary)