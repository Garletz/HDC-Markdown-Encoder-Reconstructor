"""Main pipeline orchestrator for HDC Markdown Transformer."""

import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
from hdc_markdown_transformer.reconstruction.markdown_reconstructor import ReconstructionResult

from .config.configuration_manager import ConfigurationManager, Configuration
from .core.models import (
    PreprocessedDocument, 
    MarkdownStructure, 
    SimilarityResult, 
    EvaluationMetrics
)
from .core.interfaces import (
    TokenizerInterface,
    VectorDatabaseInterface,
    HDCEncoderInterface
)
from .preprocessing.preprocessor import MarkdownPreprocessor
from .encoding.item_memory import ItemMemory
from .encoding.hdc_encoder import HDCEncoder
from .storage.vector_database import VectorDatabaseFactory
from .storage.similarity_search import SimilaritySearchEngine

# evaluation module removed during purge
from .storage.cache import CacheManager


logger = logging.getLogger(__name__)


class HDCMarkdownTransformer:
    """
    Modular pipeline orchestrator for HDC Markdown Transformer. Supports dynamic insertion of steps (functions/classes) at any point in the pipeline chain. No LLM/MarkdownReconstructor logic.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Configuration] = None):
        """
        Initialize the HDC Markdown Transformer pipeline.
        
        Args:
            config_path: Path to configuration file
            config: Pre-configured Configuration object (overrides config_path)
        """
        # Initialize configuration
        if config is not None:
            self.config_manager = ConfigurationManager()
            self.config_manager.config = config
        else:
            self.config_manager = ConfigurationManager(config_path)
        
        self.config = self.config_manager.get_config()
        
        # Validate configuration
        self.config_manager.validate_config()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self._initialized = False
        self.preprocessor: Optional[MarkdownPreprocessor] = None
        self.item_memory: Optional[ItemMemory] = None
        self.hdc_encoder: Optional[HDCEncoder] = None
        self.vector_database: Optional[VectorDatabaseInterface] = None
        self.similarity_search: Optional[SimilaritySearchEngine] = None

        self.evaluation_engine: Optional[EvaluationEngine] = None
        self.cache_manager: Optional[CacheManager] = None
        
        logger.info("HDCMarkdownTransformer initialized with configuration (modular pipeline)")
        logger.info(self.config_manager.get_summary())
        # Modular pipeline steps (list of callables)
        self.pipeline_steps = []

    def _initialize_pipeline(self):
        """Initialize all pipeline components if not already initialized."""
        if self._initialized:
            return
        import asyncio
        import sys
        from .reconstruction.markdown_reconstructor import MarkdownReconstructor
        from .reconstruction.llm_client import LLMReconstructor
        try:
            # Initialize cache manager
            if self.config.processing.enable_caching:
                self.cache_manager = CacheManager(
                    base_cache_dir=self.config.processing.cache_directory
                )
                logger.info("Cache manager initialized")
            # Initialize preprocessor
            tokenizer_kwargs = {
                'remove_punctuation': self.config.tokenizer.remove_punctuation,
                'lowercase': self.config.tokenizer.lowercase
            }
            if self.config.tokenizer.type == "spacy":
                tokenizer_kwargs['model'] = self.config.tokenizer.language
            self.preprocessor = MarkdownPreprocessor(
                tokenizer_type=self.config.tokenizer.type,
                tokenizer_kwargs=tokenizer_kwargs
            )
            logger.info(f"Preprocessor initialized with {self.config.tokenizer.type} tokenizer")
            # Initialize item memory
            self.item_memory = ItemMemory(
                dimension=self.config.hdc.dimension,
                random_seed=self.config.hdc.random_seed
            )
            # Load or create dictionary
            self._initialize_dictionary()
            # Initialize HDC encoder
            self.hdc_encoder = HDCEncoder(
                dimension=self.config.hdc.dimension,
                item_memory=self.item_memory,
                random_seed=self.config.hdc.random_seed
            )
            logger.info("HDC encoder initialized")
            # Initialize vector database
            self.vector_database = VectorDatabaseFactory.create(
                db_type=self.config.vector_database.type,
                config=self.config.vector_database.__dict__
            )
            logger.info(f"Vector database initialized: {self.config.vector_database.type}")
            # Initialize similarity search engine
            self.similarity_search = SimilaritySearchEngine(
                vector_database=self.vector_database
            )
            logger.info("Similarity search engine initialized")
            # Initialize LLM reconstructor first
            llm_reconstructor = LLMReconstructor(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                local_model_path=self.config.llm.local_model_path
            )
            # Initialize reconstructor
            self.reconstructor = MarkdownReconstructor(
                llm_reconstructor=llm_reconstructor,
                config=None  # Use defaults
            )
            logger.info(f"Reconstructor initialized with {self.config.llm.provider} LLM")
            # evaluation engine removed during purge
            self._initialized = True
            logger.info("Pipeline initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")

    def add_step(self, step, position=None):
        """Add a step (callable) to the pipeline. If position is None, append at end."""
        if position is None:
            self.pipeline_steps.append(step)
        else:
            self.pipeline_steps.insert(position, step)

    def run_pipeline(self, state):
        """Run all steps in the pipeline sequentially. Each step receives and returns the state dict."""
        for step in self.pipeline_steps:
            state = step(state)
        return state

        
        logger.info("Initializing HDC Markdown Transformer pipeline...")
        
        try:
            # Initialize cache manager
            if self.config.processing.enable_caching:
                self.cache_manager = CacheManager(
                    base_cache_dir=self.config.processing.cache_directory
                )
                logger.info("Cache manager initialized")
            
            # Initialize preprocessor
            tokenizer_kwargs = {
                'remove_punctuation': self.config.tokenizer.remove_punctuation,
                'lowercase': self.config.tokenizer.lowercase
            }
            
            # Add model parameter for spacy tokenizer
            if self.config.tokenizer.type == "spacy":
                tokenizer_kwargs['model'] = self.config.tokenizer.language
            
            self.preprocessor = MarkdownPreprocessor(
                tokenizer_type=self.config.tokenizer.type,
                tokenizer_kwargs=tokenizer_kwargs
            )
            logger.info(f"Preprocessor initialized with {self.config.tokenizer.type} tokenizer")
            
            # Initialize item memory
            self.item_memory = ItemMemory(
                dimension=self.config.hdc.dimension,
                random_seed=self.config.hdc.random_seed
            )
            
            # Load or create dictionary
            self._initialize_dictionary()
            
            # Initialize HDC encoder
            self.hdc_encoder = HDCEncoder(
                dimension=self.config.hdc.dimension,
                item_memory=self.item_memory,
                random_seed=self.config.hdc.random_seed
            )
            logger.info("HDC encoder initialized")
            
            # Initialize vector database
            self.vector_database = VectorDatabaseFactory.create(
                db_type=self.config.vector_database.type,
                config=self.config.vector_database.__dict__
            )
            logger.info(f"Vector database initialized: {self.config.vector_database.type}")
            
            # Initialize similarity search engine
            self.similarity_search = SimilaritySearchEngine(
                vector_database=self.vector_database
            )
            logger.info("Similarity search engine initialized")
            
            # Initialize LLM reconstructor first
            from .reconstruction.llm_client import LLMReconstructor
            llm_reconstructor = LLMReconstructor(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                local_model_path=self.config.llm.local_model_path
            )
            
            # Initialize reconstructor
            self.reconstructor = MarkdownReconstructor(
                llm_reconstructor=llm_reconstructor,
                config=None  # Use defaults
            )
            logger.info(f"Reconstructor initialized with {self.config.llm.provider} LLM")
            
            # Initialize evaluation engine
            self.evaluation_engine = EvaluationEngine(
                hdc_encoder=self.hdc_encoder
            )
            logger.info("Evaluation engine initialized")
            
            self._initialized = True
            logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def transform(self, 
                  markdown_content: str,
                  evaluate: bool = True,
                  cache_key: Optional[str] = None,
                  k: Optional[int] = None,
                  similarity_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Transform markdown content through the complete pipeline.
        
        Args:
            markdown_content: Input markdown content
            evaluate: Whether to evaluate reconstruction quality
            cache_key: Optional cache key for caching results
            
        Returns:
            Dictionary containing all pipeline results
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        start_time = time.time()
        logger.info("Starting markdown transformation pipeline")
        
        try:
            # Check cache first
            if self.cache_manager and cache_key:
                cached_result = self.cache_manager.get_cached_result(cache_key)
                if cached_result:
                    logger.info("Retrieved result from cache")
                    return cached_result
            
            # Step 1: Preprocess markdown
            logger.info("Step 1: Preprocessing markdown content")
            preprocessed_doc = self.preprocessor.preprocess(markdown_content)
            logger.info(f"Preprocessed document: {len(preprocessed_doc.tokens)} tokens, "
                       f"{len(preprocessed_doc.structure.headers)} headers")
            
            # Step 2: Encode to HDC hypervector
            logger.info("Step 2: Encoding to HDC hypervector")
            document_vector = self.hdc_encoder.encode_document(preprocessed_doc)
            logger.info(f"Document encoded to {document_vector.shape[0]}-dimensional hypervector")
            
            # Step 3: Similarity search
            logger.info("Step 3: Performing similarity search")
            from .storage.similarity_search import SearchConfig
            # Correction robustesse : si la vector_database est vide, la recréer et repopuler
            if hasattr(self.vector_database, 'vectors') and len(self.vector_database.vectors) == 0:
                logger.warning("Vector database is empty before search. Recreating and repopulating from ItemMemory...")
                from .storage.vector_database import InMemoryVectorDatabase
                self.vector_database = InMemoryVectorDatabase()
                self.item_memory.store_in_vector_db(self.vector_database)
                self.similarity_search.vector_database = self.vector_database
                logger.info(f"After repopulation, self.vector_database contains {len(self.vector_database.vectors)} vectors")
            # Utiliser k/threshold dynamiques ou valeurs de config
            k_val = k if k is not None else self.config.reconstruction.max_candidates
            threshold_val = similarity_threshold if similarity_threshold is not None else getattr(self.config.reconstruction, 'similarity_threshold', 0.0)
            search_config = SearchConfig(k=k_val, similarity_threshold=threshold_val)
            similarity_results = self.similarity_search.search(
                query_vector=document_vector,
                config=search_config
            )
            logger.info(f"Found {len(similarity_results)} similarity candidates (k={k_val}, threshold={threshold_val})")

            # --- Attach token count from .npy filename if available ---
            from .encoding.save_hypervector import parse_token_count_from_filename
            for result in similarity_results:
                # If the result has a filename in its metadata, try to parse token count
                filename = result.metadata.get('filename') if hasattr(result, 'metadata') else None
                if filename:
                    try:
                        token_count = parse_token_count_from_filename(filename)
                        result.metadata['token_count'] = token_count
                        logger.info(f"Parsed token_count={token_count} from {filename} for candidate '{result.token}'")
                    except Exception as e:
                        logger.warning(f"Could not parse token count from filename '{filename}': {e}")
            
            # Step 4: Reconstruct markdown
            logger.info("Step 4: Reconstructing markdown with LLM")
            reconstruction_result = self.reconstructor.reconstruct_markdown(
                candidates=similarity_results,
                structure=preprocessed_doc.structure,
                context=f"Original document had {len(preprocessed_doc.tokens)} tokens"
            )
            import asyncio
            if asyncio.iscoroutine(reconstruction_result):
                reconstruction_result = asyncio.run(reconstruction_result)
            logger.info("Markdown reconstruction completed")
            
            # Step 5: Evaluate (optional)
            evaluation_metrics = None
            if evaluate:
                logger.info("Step 5: Evaluating reconstruction quality")
                evaluation_metrics = self.evaluation_engine.evaluate_reconstruction(
                    original_content=markdown_content,
                    reconstructed_content=reconstruction_result.reconstructed_markdown,
                    original_structure=preprocessed_doc.structure
                )
                logger.info(f"Evaluation completed - Overall score: {evaluation_metrics.overall_score():.3f}")
            
            # Compile results
            processing_time = time.time() - start_time
            results = {
                'original_content': markdown_content,
                'preprocessed_document': preprocessed_doc,
                'document_vector': document_vector,
                'similarity_results': similarity_results,
                'reconstruction_result': reconstruction_result,
                'reconstructed_content': reconstruction_result.reconstructed_markdown,
                'evaluation_metrics': evaluation_metrics,
                'processing_time_seconds': processing_time,
                'pipeline_info': {
                    'tokenizer_type': self.config.tokenizer.type,
                    'hdc_dimension': self.config.hdc.dimension,
                    'vector_db_type': self.config.vector_database.type,
                    'llm_provider': self.config.llm.provider,
                    'dictionary_size': len(self.item_memory.dictionary)
                }
            }
            
            # Cache results if enabled
            if self.cache_manager and cache_key:
                self.cache_manager.cache_result(cache_key, results)
                logger.info("Results cached for future use")
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline transformation failed: {e}")
            raise RuntimeError(f"Transformation failed: {e}")
    
    def encode_only(self, markdown_content: str) -> Dict[str, Any]:
        """
        Encode markdown content to HDC hypervector only (no reconstruction).
        
        Args:
            markdown_content: Input markdown content
            
        Returns:
            HDC hypervector representation
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        logger.info("Encoding markdown to HDC hypervector")
        
        # Preprocess and encode
        preprocessed_doc = self.preprocessor.preprocess(markdown_content)
        normalized_tokens = preprocessed_doc.normalized_tokens
        in_vocab = []
        oov = []
        for token in normalized_tokens:
            if self.item_memory.has_word(token):
                in_vocab.append(token)
            else:
                oov.append(token)
        coverage = len(in_vocab) / len(normalized_tokens) if normalized_tokens else 0
        logger.info(f"[FIDELITE HDC] Couverture dictionnaire : {len(in_vocab)}/{len(normalized_tokens)} tokens (taux={coverage:.2%})")
        if oov:
            logger.info(f"[FIDELITE HDC] Tokens hors-vocabulaire ({len(oov)}): {sorted(set(oov))[:20]}{' ...' if len(set(oov))>20 else ''}")
        else:
            logger.info("[FIDELITE HDC] Tous les tokens sont couverts par le dictionnaire HDC.")
        import sys
        logger.info("[DEBUG COHERENCE HDC] Début bloc test cohérence HDC")
        sys.stdout.flush()
        for h in logger.handlers:
            h.flush() if hasattr(h, 'flush') else None

        # Désactivation temporaire de l'encodage positionnel et de la pondération TF-IDF pour la cohérence HDC
        # On encode le document comme une somme simple des vecteurs mots (comme le dico)
        # Encodage HDC pur : somme brute des vecteurs mots du dico, sans position ni pondération
        document_vector = self.hdc_encoder.encode_document(preprocessed_doc, use_tf_idf_weights=False, normalize_weights=False, apply_positional_encoding=False)
        logger.info(f"[COHERENCE HDC] Document encodé SANS position ni TF-IDF : {document_vector.shape[0]}-dimensional hypervector")
        sys.stdout.flush()
        for h in logger.handlers:
            h.flush() if hasattr(h, 'flush') else None

        # --- Save hypervector with token count in filename ---
        from .encoding.save_hypervector import save_hypervector_with_token_count
        stats = self.hdc_encoder.get_encoding_statistics(preprocessed_doc)
        valid_tokens = stats.get("valid_tokens", 0)
        save_path = save_hypervector_with_token_count(document_vector, valid_tokens)
        logger.info(f"Saved hypervector to {save_path} (valid_tokens={valid_tokens})")


        # --- TEST DE COHÉRENCE HDC ---
        test_word = "the"
        if self.item_memory.has_word(test_word):
            test_vec_doc = self.hdc_encoder.encode_tokens([test_word], apply_positional_encoding=False)[0]
            test_vec_dico = self.item_memory.get_vector(test_word)
            numer = np.dot(test_vec_doc, test_vec_dico)
            denom = np.linalg.norm(test_vec_doc) * np.linalg.norm(test_vec_dico)
            sim = numer / denom if denom != 0 else 0.0
            logger.info(f"[COHERENCE HDC] Similarité cosinus entre 'the' (doc) et 'the' (dico) : {sim:.4f}")
        else:
            logger.warning(f"[COHERENCE HDC] Mot de test 'the' absent du dictionnaire HDC !")
        sys.stdout.flush()
        for h in logger.handlers:
            h.flush() if hasattr(h, 'flush') else None

        unique_tokens = sorted(set(in_vocab))
        sims = []
        for token in unique_tokens:
            vec = self.item_memory.get_vector(token)
            numer = np.dot(document_vector, vec)
            denom = np.linalg.norm(document_vector) * np.linalg.norm(vec)
            sim = numer / denom if denom != 0 else 0.0
            sims.append((token, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"[COHERENCE HDC] Top 10 similarités doc/mot (tokens du markdown) : {sims[:10]}")
        logger.info("[DEBUG COHERENCE HDC] Fin bloc test cohérence HDC")
        sys.stdout.flush()
        for h in logger.handlers:
            h.flush() if hasattr(h, 'flush') else None
        # --- FIN TEST DE COHÉRENCE HDC ---
        return {
            "document_vector": document_vector,
            "save_path": save_path,
            "valid_tokens": valid_tokens
        }
    
    def encode_only_dual(self, markdown_content: str) -> Dict[str, Any]:
        """
        Encode markdown content to dual HDC hypervectors (content + position).
        
        Args:
            markdown_content: Input markdown content
            
        Returns:
            Dictionary with content_vector, position_vector, and save paths
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        logger.info("Encoding markdown to dual HDC hypervectors (content + position)")
        
        # Preprocess and encode
        preprocessed_doc = self.preprocessor.preprocess(markdown_content)
        normalized_tokens = preprocessed_doc.normalized_tokens
        
        # Log vocabulary coverage
        in_vocab = []
        oov = []
        for token in normalized_tokens:
            if self.item_memory.has_word(token):
                in_vocab.append(token)
            else:
                oov.append(token)
        coverage = len(in_vocab) / len(normalized_tokens) if normalized_tokens else 0
        logger.info(f"[DUAL HDC] Couverture dictionnaire : {len(in_vocab)}/{len(normalized_tokens)} tokens (taux={coverage:.2%})")
        if oov:
            logger.info(f"[DUAL HDC] Tokens hors-vocabulaire ({len(oov)}): {sorted(set(oov))[:20]}{' ...' if len(set(oov))>20 else ''}")
        else:
            logger.info("[DUAL HDC] Tous les tokens sont couverts par le dictionnaire HDC.")

        # Encode with dual vectors
        content_vector, position_vector = self.hdc_encoder.encode_document_dual(preprocessed_doc)
        
        logger.info(f"[DUAL HDC] Document encodé en 2 vecteurs : content={content_vector.shape[0]}-dim, position={position_vector.shape[0]}-dim")

        # Save dual hypervectors
        from .encoding.save_hypervector import save_dual_hypervectors_with_pairs
        stats = self.hdc_encoder.get_encoding_statistics(preprocessed_doc)
        valid_tokens = stats.get("valid_tokens", 0)
        
        # Get pair vectors from encoder
        pair_vectors = getattr(self.hdc_encoder, '_last_pair_vectors', [])
        
        content_path, position_path, pairs_path = save_dual_hypervectors_with_pairs(
            content_vector, position_vector, pair_vectors, valid_tokens
        )
        logger.info(f"Saved dual hypervectors with pairs: content={content_path}, position={position_path}, pairs={pairs_path} (valid_tokens={valid_tokens})")

        # Save paths for reconstruction
        self._last_content_path = content_path
        self._last_position_path = position_path
        self._last_pairs_path = pairs_path

        return {
            "content_vector": content_vector,
            "position_vector": position_vector,
            "content_path": content_path,
            "position_path": position_path,
            "pairs_path": pairs_path,
            "valid_tokens": valid_tokens
        }

    # reconstruct_from_vector method removed - replaced by reconstruct_from_dual_vectors
    # batch_transform method removed - not used in dual HDC pipeline
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and component information.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            'initialized': self._initialized,
            'configuration': {
                'hdc_dimension': self.config.hdc.dimension,
                'tokenizer_type': self.config.tokenizer.type,
                'vector_db_type': self.config.vector_database.type,
                'llm_provider': self.config.llm.provider,
                'caching_enabled': self.config.processing.enable_caching
            },
            'components': {
                'preprocessor': self.preprocessor is not None,
                'item_memory': self.item_memory is not None,
                'hdc_encoder': self.hdc_encoder is not None,
                'vector_database': self.vector_database is not None,
                'similarity_search': self.similarity_search is not None,
                'reconstructor': self.reconstructor is not None,
                'evaluation_engine': self.evaluation_engine is not None,
                'cache_manager': self.cache_manager is not None
            },
            'dictionary_info': {
                'size': len(self.item_memory.dictionary) if self.item_memory else 0,
                'dimension': self.config.hdc.dimension
            }
        }
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level.upper()),
            format=self.config.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.logging.file)
            ]
        )
    
    def _initialize_dictionary(self) -> None:
        """Initialize or load the HDC dictionary."""
        logger.info("Initializing HDC dictionary...")
        
        # Try to load from cache first
        cache_path = Path(self.config.processing.cache_directory) / "item_memory.npz"

        # Toujours initialiser la vector_database AVANT de manipuler le dico
        if self.vector_database is None:
            self.vector_database = VectorDatabaseFactory.create(
                db_type=self.config.vector_database.type,
                config=self.config.vector_database.__dict__
            )

        if cache_path.exists():
            try:
                self.item_memory.load_from_file(str(cache_path))
                logger.info(f"Loaded dictionary from cache: {len(self.item_memory.dictionary)} words")

                # Store in vector database of current pipeline (ALWAYS, not temp)
                if len(self.item_memory.dictionary) > 0:
                    self.item_memory.store_in_vector_db(self.vector_database)
                    logger.info("Dictionary stored in vector database (pipeline)")
                    logger.info(f"DEBUG: After pipeline population, self.vector_database contains {len(self.vector_database.vectors) if hasattr(self.vector_database, 'vectors') else 'N/A'} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached dictionary: {e}")

        # Create new dictionary
        logger.info("Creating new HDC dictionary...")

        # Load linguistic dictionary (simplified for now)
        basic_words = self._get_basic_word_list()

        self.item_memory.create_from_word_list(basic_words)
        logger.info(f"Created dictionary with {len(self.item_memory.dictionary)} words")

        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.item_memory.save_to_file(str(cache_path))
        logger.info("Dictionary saved to cache")

        # Store in vector database
        self.item_memory.store_in_vector_db(self.vector_database)
        logger.info("Dictionary stored in vector database (pipeline)")
        logger.info(f"DEBUG: After pipeline population, self.vector_database contains {len(self.vector_database.vectors) if hasattr(self.vector_database, 'vectors') else 'N/A'} vectors")
    
    def _get_basic_word_list(self) -> List[str]:
        """Load a true English vocabulary from english_vocab.txt and check its size."""
        vocab_path = Path("english_vocab.txt")
        if not vocab_path.exists():
            raise RuntimeError("english_vocab.txt introuvable dans le dossier racine du projet. Veuillez fournir un fichier vocabulaire anglais de taille suffisante.")
        with open(vocab_path, "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip() and line.strip().isalpha()]
        if len(words) < self.config.dictionary.size:
            raise RuntimeError(f"Le vocabulaire anglais fourni ({len(words)} mots) est insuffisant pour la taille du dictionnaire demandée ({self.config.dictionary.size}). Veuillez fournir un vocabulaire plus grand.")
        preview = words[:10]
        logger.info(f"[DIAG] Premiers mots du vocabulaire anglais utilisé : {preview}")
        return words[:self.config.dictionary.size]

    def reconstruct_from_dual_vectors(self, 
                                    content_vector: np.ndarray,
                                    position_vector: np.ndarray,
                                    original_structure: Optional[MarkdownStructure] = None,
                                    context: str = "",
                                    k: Optional[int] = None,
                                    similarity_threshold: Optional[float] = None,
                                    content_file_path: Optional[str] = None) -> ReconstructionResult:
        """
        Reconstruct markdown from dual HDC hypervectors (content + position).
        Uses HDC unbinding to recover (word, position) pairs.
        
        Args:
            content_vector: Content hypervector (words)
            position_vector: Position hypervector (positions)
            original_structure: Optional original structure for better reconstruction
            context: Additional context for reconstruction
            k: Number of candidates to retrieve
            similarity_threshold: Similarity threshold for filtering
            content_file_path: Path to content file for extracting token count
            
        Returns:
            Reconstruction result
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        logger.info("Reconstructing markdown from dual HDC hypervectors")
        
        # Extract token count from filename if available
        token_count = None
        if content_file_path:
            try:
                from .encoding.save_hypervector import parse_token_count_from_filename
                token_count = parse_token_count_from_filename(content_file_path)
                logger.info(f"Extracted token count from filename: {token_count}")
            except Exception as e:
                logger.warning(f"Could not parse token count from filename: {e}")
        elif hasattr(self, '_last_content_path') and self._last_content_path:
            try:
                from .encoding.save_hypervector import parse_token_count_from_filename
                token_count = parse_token_count_from_filename(self._last_content_path)
                logger.info(f"Extracted token count from filename: {token_count}")
            except Exception as e:
                logger.warning(f"Could not parse token count from filename: {e}")
        
        # Load pair vectors if available
        pair_vectors = []
        if content_file_path:
            try:
                pairs_file_path = content_file_path.replace('_content.npy', '_pairs.npy')
                pair_vectors = np.load(pairs_file_path)
                logger.info(f"Loaded {len(pair_vectors)} pair vectors from {pairs_file_path}")
                # Store in encoder for order deduction
                self.hdc_encoder._last_pair_vectors = pair_vectors
            except Exception as e:
                logger.warning(f"Could not load pair vectors: {e}")
        
        # Ensure vector database is populated
        if hasattr(self.vector_database, 'vectors') and len(self.vector_database.vectors) == 0:
            logger.warning("Vector database is empty. Recreating and repopulating from ItemMemory...")
            from .storage.vector_database import InMemoryVectorDatabase
            self.vector_database = InMemoryVectorDatabase()
            self.item_memory.store_in_vector_db(self.vector_database)
            self.similarity_search.vector_database = self.vector_database
            logger.info(f"After repopulation, self.vector_database contains {len(self.vector_database.vectors)} vectors")
        
        # Search configuration - use token count if available
        from .storage.similarity_search import SearchConfig
        k_val = token_count if token_count is not None else (k if k is not None else self.config.reconstruction.max_candidates)
        threshold_val = similarity_threshold if similarity_threshold is not None else getattr(self.config.reconstruction, 'similarity_threshold', 0.0)
        search_config = SearchConfig(k=k_val, similarity_threshold=threshold_val)
        
        logger.info(f"Using search configuration: k={k_val}, threshold={threshold_val}")
        
        # Step 1: Find words from content vector
        logger.info("Step 1: Searching for words from content vector")
        content_results = self.similarity_search.search(content_vector, search_config)
        logger.info(f"Found {len(content_results)} content candidates")
        
        # Step 2: Find positions from position vector
        logger.info("Step 2: Searching for positions from position vector")
        from .storage.vector_database import InMemoryVectorDatabase
        position_db = InMemoryVectorDatabase()
        
        # Store position vectors in separate database
        max_positions = min(1000, len(self.hdc_encoder.positional_encoder.position_vectors))
        position_vectors = {}
        for pos in range(max_positions):
            pos_vector = self.hdc_encoder.positional_encoder.get_position_vector(pos)
            position_vectors[str(pos)] = pos_vector
        
        position_db.store_vectors(position_vectors)
        logger.info(f"Created position database with {max_positions} position vectors")
        
        # Search in position database with same k as content
        from .storage.similarity_search import SimilaritySearchEngine
        position_search = SimilaritySearchEngine(vector_database=position_db)
        position_results = position_search.search(position_vector, search_config)
        logger.info(f"Found {len(position_results)} position candidates")
        
        # Step 3: Use HDC unbinding to recover (word, position) pairs
        logger.info("Step 3: Using HDC unbinding to recover word-position pairs")
        ordered_candidates = self._recover_word_position_pairs(content_results, position_results)
        logger.info(f"Recovered {len(ordered_candidates)} word-position pairs")
        
        # Step 4: Reconstruct markdown
        logger.info("Step 4: Reconstructing markdown with LLM")
        logger.info(f"Passing {len(ordered_candidates)} candidates to reconstructor")
        
        # Show all candidates for debugging
        for i, candidate in enumerate(ordered_candidates):
            logger.info(f"Candidate {i}: '{candidate.token}' at position {candidate.metadata.get('position', 'N/A')} (score: {candidate.similarity_score:.4f})")
        
        # Force lower threshold for order deduction results
        if hasattr(self.reconstructor, 'config'):
            self.reconstructor.config.min_confidence_threshold = 0.05
            logger.info("Forced min_confidence_threshold to 0.05 for order deduction results")
        
        import asyncio
        reconstruct_fn = getattr(self.reconstructor, "reconstruct_markdown", None)
        if asyncio.iscoroutinefunction(reconstruct_fn):
            reconstruction_result = asyncio.run(
                reconstruct_fn(
                    candidates=ordered_candidates,
                    structure=original_structure,
                    context=context
                )
            )
        else:
            reconstruction_result = reconstruct_fn(
                candidates=ordered_candidates,
                structure=original_structure,
                context=context
            )
        
        logger.info("Dual vector reconstruction completed")
        return reconstruction_result

    def _recover_word_position_pairs(self, content_results: List[SimilarityResult], position_results: List[SimilarityResult]) -> List[SimilarityResult]:
        """
        Intelligent approach: find words from content vector, then use (word, position) pairs to deduce order.
        
        Args:
            content_results: Similarity results from content vector
            position_results: Similarity results from position vector
            
        Returns:
            Ordered list of similarity results with word-position pairs
        """
        # Extract words and their scores from content results
        word_scores = {result.token: result.similarity_score for result in content_results}
        
        # Extract positions and their scores from position results
        position_scores = {}
        for result in position_results:
            try:
                pos = int(result.token)
                position_scores[pos] = result.similarity_score
            except (ValueError, AttributeError) as e:
                continue
        
        logger.info(f"Extracted {len(position_scores)} valid positions from position results")
        
        # Sort words by score (descending)
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sort positions by score (descending)
        sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Top 10 words: {[word for word, score in sorted_words[:10]]}")
        logger.info(f"Top 10 positions: {[pos for pos, score in sorted_positions[:10]]}")
        
        # Try to deduce order using (word, position) pairs
        ordered_candidates = self._deduce_order_from_pairs(sorted_words, position_scores)
        
        # If deduction failed, fall back to simple association
        if len(ordered_candidates) == 0:
            logger.info("Order deduction failed, using simple association")
            ordered_candidates = self._simple_association(sorted_words, position_scores)
        
        logger.info(f"Created {len(ordered_candidates)} ordered candidates")
        return ordered_candidates

    def _deduce_order_from_pairs(self, sorted_words: List[tuple], position_scores: dict) -> List[SimilarityResult]:
        """
        Deduce order by testing each word at each position using HDC unbinding.
        
        Args:
            sorted_words: List of (word, score) tuples sorted by score
            position_scores: Dictionary of position -> score
            
        Returns:
            List of ordered candidates
        """
        if not hasattr(self.hdc_encoder, '_last_pair_vectors') or len(self.hdc_encoder._last_pair_vectors) == 0:
            logger.warning("No pair vectors available for order deduction")
            return []
        
        # Get the document vector (sum of all pair vectors)
        document_vector = np.sum(self.hdc_encoder._last_pair_vectors, axis=0)
        
        ordered_candidates = []
        found_positions = set()
        
        # For each word, try to find its original position
        for word, word_score in sorted_words:
            best_position = None
            best_score = -1
            
            # Test each position dynamically based on actual token count
            num_positions = len(self.hdc_encoder._last_pair_vectors)
            for pos in range(num_positions):
                if pos in found_positions:
                    continue  # Position already taken
                
                # Get position vector
                pos_vector = self.hdc_encoder.positional_encoder.get_position_vector(pos)
                
                # HDC unbinding: multiply document vector by position vector
                # (sum of word*pos pairs) * pos = word * (pos * pos) = word
                recovered_word_vector = document_vector * pos_vector
                
                # Calculate similarity with the current word
                word_vector = self.item_memory.get_vector(word)
                if word_vector is not None:
                    similarity = self.hdc_encoder.similarity(recovered_word_vector, word_vector)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_position = pos
            
            # If we found a good position for this word
            if best_position is not None and best_score > 0.01:
                pos_score = position_scores.get(best_position, 0.0)
                combined_score = word_score * pos_score * best_score
                
                # Create a new SimilarityResult
                from .core.models import SimilarityResult
                word_vector = self.item_memory.get_vector(word)
                ordered_candidates.append(SimilarityResult(
                    token=word,
                    similarity_score=combined_score,
                    vector=word_vector,
                    metadata={'position': best_position, 'content_score': word_score, 'position_score': pos_score, 'unbinding_score': best_score}
                ))
                found_positions.add(best_position)
                logger.debug(f"Position {best_position}: word '{word}' (unbinding score: {best_score:.4f}, combined: {combined_score:.4f})")
        
        # Sort by position to maintain order
        ordered_candidates.sort(key=lambda x: x.metadata.get('position', 0))
        
        return ordered_candidates

    def _simple_association(self, sorted_words: List[tuple], position_scores: dict) -> List[SimilarityResult]:
        """
        Simple fallback: associate top words with top positions.
        
        Args:
            sorted_words: List of (word, score) tuples sorted by score
            position_scores: Dictionary of position -> score
            
        Returns:
            List of ordered candidates
        """
        ordered_candidates = []
        
        # Sort positions by their actual position number (ascending) to maintain original order
        sorted_positions_by_number = sorted(position_scores.items(), key=lambda x: x[0])
        logger.info(f"Positions sorted by number: {[pos for pos, score in sorted_positions_by_number[:10]]}")
        
        # Create a mapping of position -> word based on similarity scores
        position_to_word = {}
        for i, (word, word_score) in enumerate(sorted_words):
            if i < len(sorted_positions_by_number):
                pos, pos_score = sorted_positions_by_number[i]
                position_to_word[pos] = word
        
        # Create candidates in position order (0, 1, 2, 3, 4, 5, 6)
        for pos in range(7):  # We know there are 7 tokens
            if pos in position_to_word:
                word = position_to_word[pos]
                word_score = dict(sorted_words)[word]  # Get score from sorted_words
                pos_score = position_scores[pos]
                combined_score = word_score * pos_score
                
                # Create a new SimilarityResult
                from .core.models import SimilarityResult
                word_vector = self.item_memory.get_vector(word)
                ordered_candidates.append(SimilarityResult(
                    token=word,
                    similarity_score=combined_score,
                    vector=word_vector,
                    metadata={'position': pos, 'content_score': word_score, 'position_score': pos_score}
                ))
                logger.debug(f"Position {pos}: word '{word}' (combined score: {combined_score:.4f})")
        
        return ordered_candidates