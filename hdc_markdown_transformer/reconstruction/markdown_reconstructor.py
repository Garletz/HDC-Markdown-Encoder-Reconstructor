"""Markdown reconstruction logic combining similarity results with LLM generation."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from .llm_client import LLMReconstructor
from .prompt_builder import PromptBuilder
from ..core.models import SimilarityResult, MarkdownStructure, Header, ListItem, Link, CodeBlock, Emphasis
from ..core.interfaces import ReconstructorInterface


logger = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    """Result of markdown reconstruction process."""
    reconstructed_markdown: str
    confidence_score: float
    processing_time_ms: float
    candidate_count: int
    structure_preserved: bool
    error_message: Optional[str] = None


@dataclass
class ReconstructionConfig:
    """Configuration for reconstruction process."""
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


class CandidateFilter:
    """Filters and processes similarity candidates."""
    
    def __init__(self, config: ReconstructionConfig):
        """Initialize candidate filter.
        
        Args:
            config: Reconstruction configuration
        """
        self.config = config
    
    def filter_candidates(self, candidates: List[SimilarityResult]) -> List[SimilarityResult]:
        """Filter and process candidates for reconstruction.
        
        Args:
            candidates: Raw similarity results
            
        Returns:
            Filtered and processed candidates
        """
        if not candidates:
            return []
        
        # Filter by confidence threshold
        filtered = [
            c for c in candidates 
            if c.similarity_score >= self.config.min_confidence_threshold
        ]
        
        # Remove duplicates if enabled
        if self.config.filter_duplicates:
            filtered = self._remove_duplicates(filtered)
        
        # Rerank candidates if enabled
        if self.config.rerank_candidates:
            filtered = self._rerank_candidates(filtered)
        
        # Limit to max candidates
        return filtered[:self.config.max_candidates]
    
    def _remove_duplicates(self, candidates: List[SimilarityResult]) -> List[SimilarityResult]:
        """Remove duplicate candidates based on token similarity.
        
        Args:
            candidates: List of candidates
            
        Returns:
            Deduplicated candidates
        """
        seen_tokens = set()
        unique_candidates = []
        
        for candidate in candidates:
            # Normalize token for comparison
            normalized_token = candidate.token.lower().strip()
            
            if normalized_token not in seen_tokens:
                seen_tokens.add(normalized_token)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _rerank_candidates(self, candidates: List[SimilarityResult]) -> List[SimilarityResult]:
        """Rerank candidates based on various factors.
        
        Args:
            candidates: List of candidates to rerank
            
        Returns:
            Reranked candidates
        """
        def rerank_score(candidate: SimilarityResult) -> float:
            """Calculate reranking score for a candidate."""
            base_score = candidate.similarity_score
            
            # Boost longer, more meaningful tokens
            length_boost = min(len(candidate.token) / 20.0, 0.1)
            
            # Boost tokens that look like technical terms or proper nouns
            technical_boost = 0.05 if self._is_technical_term(candidate.token) else 0.0
            
            # Penalize very common words
            common_penalty = -0.05 if self._is_common_word(candidate.token) else 0.0
            
            return base_score + length_boost + technical_boost + common_penalty
        
        # Sort by reranking score
        return sorted(candidates, key=rerank_score, reverse=True)
    
    def _is_technical_term(self, token: str) -> bool:
        """Check if token appears to be a technical term.
        
        Args:
            token: Token to check
            
        Returns:
            True if token appears technical
        """
        # Simple heuristics for technical terms
        technical_patterns = [
            r'^[A-Z][a-z]+[A-Z]',  # CamelCase
            r'[a-z]+_[a-z]+',      # snake_case
            r'[a-z]+-[a-z]+',      # kebab-case
            r'\.[a-z]+$',          # file extensions
            r'^[A-Z]{2,}$',        # ACRONYMS
        ]
        
        return any(re.match(pattern, token) for pattern in technical_patterns)
    
    def _is_common_word(self, token: str) -> bool:
        """Check if token is a very common word.
        
        Args:
            token: Token to check
            
        Returns:
            True if token is common
        """
        # Simple list of very common words to penalize
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you', 'he', 'she'
        }
        
        return token.lower() in common_words


class StructureValidator:
    """Validates and corrects markdown structure preservation."""
    
    def __init__(self, original_structure: MarkdownStructure):
        """Initialize structure validator.
        
        Args:
            original_structure: Original document structure
        """
        self.original_structure = original_structure
    
    def validate_structure(self, reconstructed_markdown: str) -> Tuple[bool, List[str]]:
        """Validate that structure is preserved in reconstructed markdown.
        
        Args:
            reconstructed_markdown: Reconstructed markdown content
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check headers
        header_issues = self._validate_headers(reconstructed_markdown)
        issues.extend(header_issues)
        
        # Check lists
        list_issues = self._validate_lists(reconstructed_markdown)
        issues.extend(list_issues)
        
        # Check links
        link_issues = self._validate_links(reconstructed_markdown)
        issues.extend(link_issues)
        
        # Check code blocks
        code_issues = self._validate_code_blocks(reconstructed_markdown)
        issues.extend(code_issues)
        
        return len(issues) == 0, issues
    
    def _validate_headers(self, markdown: str) -> List[str]:
        """Validate header structure preservation."""
        issues = []
        
        if not self.original_structure.headers:
            return issues
        
        # Extract headers from reconstructed markdown
        header_pattern = r'^(#{1,6})\s+(.+)$'
        found_headers = []
        
        for line in markdown.split('\n'):
            match = re.match(header_pattern, line.strip())
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                found_headers.append((level, text))
        
        # Check if we have roughly the same number of headers
        expected_count = len(self.original_structure.headers)
        found_count = len(found_headers)
        
        if found_count < expected_count * 0.5:  # Allow some flexibility
            issues.append(f"Expected ~{expected_count} headers, found {found_count}")
        
        # Check header hierarchy
        if found_headers:
            prev_level = 0
            for level, text in found_headers:
                if level > prev_level + 1:
                    issues.append(f"Header hierarchy skip: level {prev_level} to {level}")
                prev_level = level
        
        return issues
    
    def _validate_lists(self, markdown: str) -> List[str]:
        """Validate list structure preservation."""
        issues = []
        
        if not self.original_structure.lists:
            return issues
        
        # Count list items in reconstructed markdown
        list_pattern = r'^\s*[-*+]\s+|^\s*\d+\.\s+'
        found_lists = 0
        
        for line in markdown.split('\n'):
            if re.match(list_pattern, line):
                found_lists += 1
        
        expected_count = len(self.original_structure.lists)
        
        if found_lists < expected_count * 0.5:  # Allow some flexibility
            issues.append(f"Expected ~{expected_count} list items, found {found_lists}")
        
        return issues
    
    def _validate_links(self, markdown: str) -> List[str]:
        """Validate link structure preservation."""
        issues = []
        
        if not self.original_structure.links:
            return issues
        
        # Count links in reconstructed markdown
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        found_links = len(re.findall(link_pattern, markdown))
        
        expected_count = len(self.original_structure.links)
        
        if found_links < expected_count * 0.5:  # Allow some flexibility
            issues.append(f"Expected ~{expected_count} links, found {found_links}")
        
        return issues
    
    def _validate_code_blocks(self, markdown: str) -> List[str]:
        """Validate code block structure preservation."""
        issues = []
        
        if not self.original_structure.code_blocks:
            return issues
        
        # Count code blocks in reconstructed markdown
        code_block_pattern = r'```[\s\S]*?```'
        found_blocks = len(re.findall(code_block_pattern, markdown))
        
        expected_count = len(self.original_structure.code_blocks)
        
        if found_blocks < expected_count * 0.5:  # Allow some flexibility
            issues.append(f"Expected ~{expected_count} code blocks, found {found_blocks}")
        
        return issues
    
    def correct_structure(self, markdown: str, issues: List[str]) -> str:
        """Attempt to correct structural issues in markdown.
        
        Args:
            markdown: Markdown with structural issues
            issues: List of identified issues
            
        Returns:
            Corrected markdown
        """
        corrected = markdown
        
        # Simple corrections for common issues
        for issue in issues:
            if "Header hierarchy skip" in issue:
                corrected = self._fix_header_hierarchy(corrected)
            elif "Expected" in issue and "headers" in issue:
                corrected = self._ensure_minimum_headers(corrected)
        
        return corrected
    
    def _fix_header_hierarchy(self, markdown: str) -> str:
        """Fix header hierarchy issues."""
        lines = markdown.split('\n')
        corrected_lines = []
        prev_level = 0
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2)
                
                # Ensure proper hierarchy
                if level > prev_level + 1:
                    level = prev_level + 1
                
                corrected_lines.append('#' * level + ' ' + text)
                prev_level = level
            else:
                corrected_lines.append(line)
        
        return '\n'.join(corrected_lines)
    
    def _ensure_minimum_headers(self, markdown: str) -> str:
        """Ensure minimum number of headers are present."""
        if not self.original_structure.headers:
            return markdown
        
        # If no headers found, add a main header
        if not re.search(r'^#{1,6}\s+', markdown, re.MULTILINE):
            main_header = self.original_structure.headers[0]
            header_line = '#' * main_header.level + ' ' + main_header.text
            return header_line + '\n\n' + markdown
        
        return markdown


class MarkdownReconstructor(ReconstructorInterface):
    """Main markdown reconstruction system combining similarity results with LLM generation."""
    
    def __init__(self, 
                 llm_reconstructor: LLMReconstructor,
                 config: Optional[ReconstructionConfig] = None):
        """Initialize markdown reconstructor.
        
        Args:
            llm_reconstructor: LLM client for text generation
            config: Reconstruction configuration
        """
        self.llm_reconstructor = llm_reconstructor
        self.config = config or ReconstructionConfig()
        self.encoded_vector_path = None  # Always present, set by pipeline or CLI
        self.prompt_builder = PromptBuilder(language=self.config.language)
        self.candidate_filter = CandidateFilter(self.config)
        
        logger.info(f"Initialized MarkdownReconstructor with config: {self.config}")
    
    async def reconstruct_markdown(self, 
                                 candidates: List[SimilarityResult], 
                                 structure: MarkdownStructure,
                                 context: str = "") -> ReconstructionResult:
        """Reconstruct markdown from candidates and structure information.
        
        Args:
            candidates: List of similarity search results
            structure: Original markdown structure
            context: Additional context for reconstruction
            
        Returns:
            Reconstruction result with generated markdown
        """
        import time
        start_time = time.time()
        
        try:
            # Simple approach: temporarily set threshold to 0.0 for no-LLM mode
            # We'll detect no-LLM mode later and know we need all candidates
            original_threshold = self.config.min_confidence_threshold
            
            # Filter and process candidates
            filtered_candidates = self.candidate_filter.filter_candidates(candidates)
            
            if not filtered_candidates:
                logger.warning("No candidates available after filtering")
                return ReconstructionResult(
                    reconstructed_markdown="# Document\n\nNo content could be reconstructed.",
                    confidence_score=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    candidate_count=0,
                    structure_preserved=False,
                    error_message="No candidates available after filtering"
                )
            # Mode sans LLM : on utilise le nombre de tokens du fichier encodé pour limiter la sortie
            # IMPORTANT: For no-LLM mode, we need ALL candidates including those with score 0.0
            # So we re-filter with threshold 0.0 to get all HDC candidates
            logger.info("Entering no-LLM mode: re-filtering candidates with threshold 0.0 to include all HDC results")
            
            # Temporarily set threshold to 0.0 and re-filter
            self.config.min_confidence_threshold = 0.0
            all_candidates = self.candidate_filter.filter_candidates(candidates)
            self.config.min_confidence_threshold = original_threshold  # Restore original
            
            import os
            from hdc_markdown_transformer.encoding.save_hypervector import parse_token_count_from_filename
            try:
                filename = os.path.basename(getattr(self, 'encoded_vector_path', ''))
                if not filename and candidates and hasattr(candidates[0], 'metadata'):
                    # Try to get filename from candidate metadata if available
                    filename = candidates[0].metadata.get('vector_filename', '')
                token_count = parse_token_count_from_filename(filename) if filename else self.config.max_candidates
            except Exception as e:
                logger.warning(f"Could not parse token count from filename: {e}")
                token_count = self.config.max_candidates
            logger.info(f"No-LLM mode: reconstructing markdown as token list of {token_count} tokens.")
            logger.info(f"Using {len(all_candidates)} candidates (including score 0.0) instead of {len(filtered_candidates)} filtered candidates")
            
            # Sort candidates by position to maintain word order
            # Position is stored in metadata['position'], not as direct attribute
            sorted_candidates = sorted(all_candidates[:token_count], key=lambda c: c.metadata.get('position', 0))
            
            # Debug: log the sorting
            logger.info(f"Candidates before sorting: {[(c.token, c.metadata.get('position', 'N/A')) for c in all_candidates[:token_count]]}")
            logger.info(f"Candidates after sorting: {[(c.token, c.metadata.get('position', 'N/A')) for c in sorted_candidates]}")
            reconstructed_markdown = ' '.join([c.token for c in sorted_candidates])
            processing_time = (time.time() - start_time) * 1000
            return ReconstructionResult(
                reconstructed_markdown=reconstructed_markdown,
                confidence_score=1.0,
                processing_time_ms=processing_time,
                candidate_count=token_count,
                structure_preserved=True
            )
            # Log détaillé des candidats transmis au LLM
            logger.info(f"Candidats transmis au LLM (top {len(filtered_candidates)}): {[c.token for c in filtered_candidates]}")
            # Générer le markdown reconstruit via le LLM
            prompt = self.prompt_builder.build_prompt(filtered_candidates, structure, context)
            reconstructed_markdown = await self.llm_reconstructor.generate_markdown(prompt)
            # Validate and correct structure si demandé
            structure_preserved = True
            if self.config.preserve_structure:
                validator = StructureValidator(structure)
                is_valid, issues = validator.validate_structure(reconstructed_markdown)
                if not is_valid:
                    logger.info(f"Structure issues found: {issues}")
                    reconstructed_markdown = validator.correct_structure(reconstructed_markdown, issues)
                    # Re-validate after correction
                    is_valid, _ = validator.validate_structure(reconstructed_markdown)
                structure_preserved = is_valid
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                filtered_candidates, structure, reconstructed_markdown
            )
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Reconstruction completed in {processing_time:.2f}ms with confidence {confidence_score:.3f}")
            return ReconstructionResult(
                reconstructed_markdown=reconstructed_markdown,
                confidence_score=confidence_score,
                processing_time_ms=processing_time,
                candidate_count=len(filtered_candidates),
                structure_preserved=structure_preserved
            )
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ReconstructionResult(
                reconstructed_markdown="# Error\n\nReconstruction failed.",
                confidence_score=0.0,
                processing_time_ms=processing_time,
                candidate_count=len(candidates) if candidates else 0,
                structure_preserved=False,
                error_message=str(e)
            )
    
    async def _generate_with_retries(self, prompt: str) -> str:
        """Generate text with retry logic.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated markdown text
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await self.llm_reconstructor.generate_text(prompt)
                if result and result.strip():
                    return result.strip()
                else:
                    raise ValueError("Empty response from LLM")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # If all retries failed, raise the last error
        raise last_error or Exception("All generation attempts failed")
    
    def _calculate_confidence_score(self, 
                                  candidates: List[SimilarityResult],
                                  structure: MarkdownStructure,
                                  reconstructed_markdown: str) -> float:
        """Calculate confidence score for reconstruction.
        
        Args:
            candidates: Filtered candidates used
            structure: Original structure
            reconstructed_markdown: Generated markdown
            
        Returns:
            Confidence score between 0 and 1
        """
        if not candidates:
            return 0.0
        
        # Base score from candidate quality
        avg_similarity = sum(c.similarity_score for c in candidates) / len(candidates)
        candidate_score = min(avg_similarity, 1.0)
        
        # Structure preservation score
        if structure:
            validator = StructureValidator(structure)
            is_valid, issues = validator.validate_structure(reconstructed_markdown)
            structure_score = 1.0 if is_valid else max(0.5 - len(issues) * 0.1, 0.0)
        else:
            structure_score = 1.0
        
        # Content quality score (simple heuristics)
        content_score = self._assess_content_quality(reconstructed_markdown)
        
        # Weighted combination
        confidence = (
            candidate_score * 0.4 +
            structure_score * 0.3 +
            content_score * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _assess_content_quality(self, markdown: str) -> float:
        """Assess the quality of generated content.
        
        Args:
            markdown: Generated markdown content
            
        Returns:
            Quality score between 0 and 1
        """
        if not markdown or len(markdown.strip()) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for proper markdown structure
        if re.search(r'^#{1,6}\s+', markdown, re.MULTILINE):
            score += 0.2  # Has headers
        
        if re.search(r'^\s*[-*+]\s+|^\s*\d+\.\s+', markdown, re.MULTILINE):
            score += 0.1  # Has lists
        
        if re.search(r'\[([^\]]+)\]\(([^)]+)\)', markdown):
            score += 0.1  # Has links
        
        # Check for reasonable length
        word_count = len(markdown.split())
        if word_count > 20:
            score += 0.1
        
        return min(score, 1.0)
    
    def update_config(self, config: ReconstructionConfig) -> None:
        """Update reconstruction configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
        self.prompt_builder.set_language(config.language)
        self.candidate_filter = CandidateFilter(config)
        logger.info(f"Updated reconstruction config: {config}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reconstruction statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "config": {
                "max_candidates": self.config.max_candidates,
                "language": self.config.language,
                "document_type": self.config.document_type,
                "preserve_structure": self.config.preserve_structure
            },
            "llm_stats": self.llm_reconstructor.get_stats()
        }


# Synchronous wrapper
class SyncMarkdownReconstructor:
    """Synchronous wrapper for MarkdownReconstructor."""
    
    def __init__(self, *args, **kwargs):
        """Initialize sync wrapper."""
        self.async_reconstructor = MarkdownReconstructor(*args, **kwargs)
    
    def reconstruct_markdown(self, 
                           candidates: List[SimilarityResult], 
                           structure: MarkdownStructure,
                           context: str = "") -> ReconstructionResult:
        """Synchronous markdown reconstruction."""
        return asyncio.run(self.async_reconstructor.reconstruct_markdown(
            candidates, structure, context
        ))
    
    def update_config(self, config: ReconstructionConfig) -> None:
        """Update configuration."""
        self.async_reconstructor.update_config(config)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.async_reconstructor.get_stats()