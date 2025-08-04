"""Prompt construction system for LLM-based markdown reconstruction."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.models import SimilarityResult, MarkdownStructure


logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Template for constructing prompts."""
    system_prompt: str
    user_prompt_template: str
    context_template: str
    candidates_template: str
    structure_template: str


class PromptBuilder:
    """Builder for structured prompt generation with candidate integration."""
    
    def __init__(self, language: str = "en"):
        """Initialize prompt builder.
        
        Args:
            language: Language for prompts (en, fr, etc.)
        """
        self.language = language
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different languages."""
        templates = {}
        
        # English templates
        templates["en"] = PromptTemplate(
            system_prompt="""You are an expert markdown reconstruction assistant. Your task is to reconstruct coherent markdown documents from similarity search candidates while preserving the original structure and meaning.

Key principles:
1. Maintain markdown structure (headers, lists, links, code blocks, emphasis)
2. Use candidates as semantic hints, not literal text
3. Generate natural, coherent content that flows well
4. Preserve the logical hierarchy and organization
5. Ensure technical accuracy when reconstructing code or technical content""",
            
            user_prompt_template="""Reconstruct a markdown document using the following information:

{context}

{candidates}

{structure}

Requirements:
- Generate coherent, well-structured markdown
- Use candidates as semantic guidance, not verbatim text
- Maintain the original document structure and hierarchy
- Ensure content flows naturally and makes sense
- Preserve any technical details or code snippets accurately

Please provide the reconstructed markdown document:""",
            
            context_template="""## Context Information:
{context_text}""",
            
            candidates_template="""## Similarity Candidates:
The following words/phrases were identified as semantically similar to the original content:

{candidate_list}

Use these as semantic hints to understand the document's topic and content, but generate natural, coherent text rather than simply listing these terms.""",
            
            structure_template="""## Document Structure:
The original document had the following structure:

{structure_info}

Please maintain this structural organization in your reconstruction."""
        )
        
        # French templates
        templates["fr"] = PromptTemplate(
            system_prompt="""Vous êtes un assistant expert en reconstruction de documents markdown. Votre tâche est de reconstruire des documents markdown cohérents à partir de candidats de recherche de similarité tout en préservant la structure et le sens originaux.

Principes clés:
1. Maintenir la structure markdown (titres, listes, liens, blocs de code, emphase)
2. Utiliser les candidats comme indices sémantiques, pas comme texte littéral
3. Générer un contenu naturel et cohérent qui s'enchaîne bien
4. Préserver la hiérarchie et l'organisation logiques
5. Assurer la précision technique lors de la reconstruction de code ou de contenu technique""",
            
            user_prompt_template="""Reconstruisez un document markdown en utilisant les informations suivantes:

{context}

{candidates}

{structure}

Exigences:
- Générer un markdown cohérent et bien structuré
- Utiliser les candidats comme guide sémantique, pas comme texte verbatim
- Maintenir la structure et la hiérarchie du document original
- S'assurer que le contenu s'enchaîne naturellement et a du sens
- Préserver avec précision tous les détails techniques ou extraits de code

Veuillez fournir le document markdown reconstruit:""",
            
            context_template="""## Informations de contexte:
{context_text}""",
            
            candidates_template="""## Candidats de similarité:
Les mots/phrases suivants ont été identifiés comme sémantiquement similaires au contenu original:

{candidate_list}

Utilisez-les comme indices sémantiques pour comprendre le sujet et le contenu du document, mais générez un texte naturel et cohérent plutôt que de simplement lister ces termes.""",
            
            structure_template="""## Structure du document:
Le document original avait la structure suivante:

{structure_info}

Veuillez maintenir cette organisation structurelle dans votre reconstruction."""
        )
        
        return templates
    
    def build_prompt(self, 
                    candidates: List[SimilarityResult], 
                    structure: MarkdownStructure,
                    context: str = "",
                    max_candidates: int = 20,
                    include_scores: bool = False) -> str:
        """Build a structured prompt for markdown reconstruction.
        
        Args:
            candidates: List of similarity search results
            structure: Original markdown structure information
            context: Additional context information
            max_candidates: Maximum number of candidates to include
            include_scores: Whether to include similarity scores
            
        Returns:
            Formatted prompt string
        """
        template = self.templates.get(self.language, self.templates["en"])
        
        # Build context section
        context_section = ""
        if context.strip():
            context_section = template.context_template.format(context_text=context)
        
        # Build candidates section
        candidates_section = self._format_candidates(
            candidates[:max_candidates], 
            template.candidates_template,
            include_scores
        )
        
        # Build structure section
        structure_section = self._format_structure(structure, template.structure_template)
        
        # Combine all sections
        user_prompt = template.user_prompt_template.format(
            context=context_section,
            candidates=candidates_section,
            structure=structure_section
        )
        
        return f"{template.system_prompt}\n\n{user_prompt}"
    
    def _format_candidates(self, 
                          candidates: List[SimilarityResult], 
                          template: str,
                          include_scores: bool) -> str:
        """Format similarity candidates for the prompt.
        
        Args:
            candidates: List of similarity results
            template: Template for candidates section
            include_scores: Whether to include similarity scores
            
        Returns:
            Formatted candidates section
        """
        if not candidates:
            return template.format(candidate_list="No similarity candidates available.")
        
        candidate_items = []
        for i, candidate in enumerate(candidates, 1):
            if include_scores:
                item = f"{i}. {candidate.token} (similarity: {candidate.similarity_score:.3f})"
            else:
                item = f"{i}. {candidate.token}"
            candidate_items.append(item)
        
        candidate_list = "\n".join(candidate_items)
        return template.format(candidate_list=candidate_list)
    
    def _format_structure(self, structure: MarkdownStructure, template: str) -> str:
        """Format markdown structure information for the prompt.
        
        Args:
            structure: Markdown structure object
            template: Template for structure section
            
        Returns:
            Formatted structure section
        """
        if not structure:
            return template.format(structure_info="No structure information available.")
        
        structure_items = []
        
        # Format headers
        if hasattr(structure, 'headers') and structure.headers:
            structure_items.append("Headers:")
            for header in structure.headers:
                level_indicator = "#" * header.level
                structure_items.append(f"  {level_indicator} {header.text} (level {header.level})")
        
        # Format lists
        if hasattr(structure, 'lists') and structure.lists:
            structure_items.append("\nLists:")
            for list_item in structure.lists:
                list_type = "ordered" if hasattr(list_item, 'ordered') and list_item.ordered else "unordered"
                structure_items.append(f"  - {list_type} list item: {list_item.text}")
        
        # Format links
        if hasattr(structure, 'links') and structure.links:
            structure_items.append("\nLinks:")
            for link in structure.links:
                structure_items.append(f"  - [{link.text}]({link.url})")
        
        # Format code blocks
        if hasattr(structure, 'code_blocks') and structure.code_blocks:
            structure_items.append("\nCode blocks:")
            for code_block in structure.code_blocks:
                language = getattr(code_block, 'language', 'unknown')
                structure_items.append(f"  - {language} code block")
        
        # Format emphasis
        if hasattr(structure, 'emphasis') and structure.emphasis:
            structure_items.append("\nEmphasis:")
            for emphasis in structure.emphasis:
                emphasis_type = getattr(emphasis, 'type', 'unknown')
                structure_items.append(f"  - {emphasis_type}: {emphasis.text}")
        
        if not structure_items:
            structure_info = "No specific structure elements identified."
        else:
            structure_info = "\n".join(structure_items)
        
        return template.format(structure_info=structure_info)
    
    def build_simple_prompt(self, candidates: List[str], context: str = "") -> str:
        """Build a simple prompt from candidate words.
        
        Args:
            candidates: List of candidate words/phrases
            context: Additional context
            
        Returns:
            Simple prompt string
        """
        template = self.templates.get(self.language, self.templates["en"])
        
        # Convert string candidates to SimilarityResult objects
        similarity_candidates = [
            SimilarityResult(token=candidate, similarity_score=1.0, vector=None, metadata={})
            for candidate in candidates
        ]
        
        # Create empty structure
        empty_structure = MarkdownStructure(
            headers=[], lists=[], links=[], code_blocks=[], emphasis=[]
        )
        
        return self.build_prompt(similarity_candidates, empty_structure, context)
    
    def build_contextual_prompt(self, 
                               candidates: List[SimilarityResult],
                               structure: MarkdownStructure,
                               document_type: str = "general",
                               target_audience: str = "general",
                               additional_instructions: str = "") -> str:
        """Build a context-aware prompt with specific instructions.
        
        Args:
            candidates: Similarity search results
            structure: Document structure
            document_type: Type of document (technical, tutorial, reference, etc.)
            target_audience: Target audience (developers, users, etc.)
            additional_instructions: Additional specific instructions
            
        Returns:
            Context-aware prompt
        """
        # Build context based on document type and audience
        context_parts = []
        
        if document_type != "general":
            context_parts.append(f"Document type: {document_type}")
        
        if target_audience != "general":
            context_parts.append(f"Target audience: {target_audience}")
        
        if additional_instructions:
            context_parts.append(f"Additional instructions: {additional_instructions}")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        return self.build_prompt(candidates, structure, context)
    
    def get_available_languages(self) -> List[str]:
        """Get list of available prompt languages.
        
        Returns:
            List of language codes
        """
        return list(self.templates.keys())
    
    def set_language(self, language: str) -> bool:
        """Set the prompt language.
        
        Args:
            language: Language code
            
        Returns:
            True if language was set successfully, False otherwise
        """
        if language in self.templates:
            self.language = language
            logger.info(f"Prompt language set to: {language}")
            return True
        else:
            logger.warning(f"Language '{language}' not available. Available: {list(self.templates.keys())}")
            return False
    
    def add_custom_template(self, language: str, template: PromptTemplate) -> None:
        """Add a custom prompt template for a language.
        
        Args:
            language: Language code
            template: Custom prompt template
        """
        self.templates[language] = template
        logger.info(f"Added custom template for language: {language}")
    
    def validate_prompt_length(self, prompt: str, max_tokens: int = 4000) -> Dict[str, Any]:
        """Validate prompt length and provide statistics.
        
        Args:
            prompt: Generated prompt
            max_tokens: Maximum allowed tokens (rough estimate)
            
        Returns:
            Dictionary with validation results and statistics
        """
        # Rough token estimation (1 token ≈ 4 characters for English)
        estimated_tokens = len(prompt) // 4
        word_count = len(prompt.split())
        char_count = len(prompt)
        
        is_valid = estimated_tokens <= max_tokens
        
        return {
            "is_valid": is_valid,
            "estimated_tokens": estimated_tokens,
            "max_tokens": max_tokens,
            "word_count": word_count,
            "char_count": char_count,
            "token_usage_ratio": estimated_tokens / max_tokens if max_tokens > 0 else 0
        }