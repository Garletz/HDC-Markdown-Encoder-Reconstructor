"""LLM client abstraction with multiple provider support."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import aiohttp
import openai
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..core.interfaces import LLMInterface
from ..core.models import SimilarityResult, MarkdownStructure


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    tokens_used: int
    model: str
    provider: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make API call."""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
                return await self.acquire()
        
        # Record this call
        self.calls.append(now)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, max_tokens: int = 2000, temperature: float = 0.3):
        """Initialize base LLM client.
        
        Args:
            model: Model name/identifier
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limiter = RateLimiter(max_calls=60, time_window=60.0)  # 60 calls per minute
    
    @abstractmethod
    async def _generate_text_impl(self, prompt: str) -> LLMResponse:
        """Implementation-specific text generation."""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    async def generate_text(self, prompt: str) -> LLMResponse:
        """Generate text with retry logic and rate limiting.
        
        Args:
            prompt: Input prompt
            
        Returns:
            LLM response with generated text
        """
        await self.rate_limiter.acquire()
        
        start_time = time.time()
        try:
            response = await self._generate_text_impl(prompt)
            response.latency_ms = (time.time() - start_time) * 1000
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return LLMResponse(
                text="",
                tokens_used=0,
                model=self.model,
                provider=self.__class__.__name__,
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error_message=str(e)
            )


class OpenAIClient(BaseLLMClient):
    """OpenAI API client implementation."""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo", **kwargs):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4-turbo, gpt-3.5-turbo, etc.)
            **kwargs: Additional parameters for base class
        """
        super().__init__(model, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Adjust rate limits based on model
        if "gpt-4" in model:
            self.rate_limiter = RateLimiter(max_calls=40, time_window=60.0)  # More conservative for GPT-4
        else:
            self.rate_limiter = RateLimiter(max_calls=60, time_window=60.0)
    
    async def _generate_text_impl(self, prompt: str) -> LLMResponse:
        """Generate text using OpenAI API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return LLMResponse(
                text=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self.model,
                provider="OpenAI",
                latency_ms=0,  # Will be set by caller
                success=True
            )
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected OpenAI error: {e}")
            raise


class GeminiClient(BaseLLMClient):
    """Google Gemini API client implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro", **kwargs):
        """Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
            model: Model name (gemini-pro, etc.)
            **kwargs: Additional parameters for base class
        """
        super().__init__(model, **kwargs)
        
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model)
        
        # Set rate limits (Gemini has different rate limits than OpenAI)
        self.rate_limiter = RateLimiter(max_calls=60, time_window=60.0)  # 60 RPM by default
        
        logger.info(f"Initialized Gemini client with model: {model}")
    
    async def _generate_text_impl(self, prompt: str) -> LLMResponse:
        """Generate text using Gemini API."""
        try:
            # Generate content with retry logic
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    }
                )
            )
            
            # Extract the generated text
            generated_text = response.text if hasattr(response, 'text') else ""
            
            # Estimate token usage (Gemini doesn't return token count in response)
            estimated_tokens = len(generated_text.split())  # Rough estimate
            
            return LLMResponse(
                text=generated_text,
                tokens_used=estimated_tokens,
                model=self.model,
                provider="Gemini",
                latency_ms=0,  # Will be set by caller
                success=True
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return LLMResponse(
                text="",
                tokens_used=0,
                model=self.model,
                provider="Gemini",
                latency_ms=0,
                success=False,
                error_message=str(e)
            )


class LocalLLMClient(BaseLLMClient):
    """Local LLM client implementation (placeholder for local models)."""
    
    def __init__(self, model_path: str, model: str = "local", **kwargs):
        """Initialize local LLM client.
        
        Args:
            model_path: Path to local model
            model: Model identifier
            **kwargs: Additional parameters for base class
        """
        super().__init__(model, **kwargs)
        self.model_path = model_path
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60.0)  # More generous for local
        
        # TODO: Initialize local model (e.g., using transformers, llama.cpp, etc.)
        logger.warning("Local LLM client is a placeholder implementation")
    
    async def _generate_text_impl(self, prompt: str) -> LLMResponse:
        """Generate text using local model."""
        # TODO: Implement actual local model inference
        # This is a placeholder that simulates local generation
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return LLMResponse(
            text="[Local model placeholder response]",
            tokens_used=len(prompt.split()) + 10,  # Rough estimate
            model=self.model,
            provider="Local",
            latency_ms=0,  # Will be set by caller
            success=True
        )


class LLMReconstructor(LLMInterface):
    """Main LLM reconstructor with multiple provider support."""
    
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, 
                 local_model_path: Optional[str] = None, **kwargs):
        """Initialize LLM reconstructor.
        
        Args:
            provider: LLM provider ("openai", "gemini", or "local")
            model: Model name/identifier
            api_key: API key for cloud providers
            local_model_path: Path to local model
            **kwargs: Additional parameters for LLM client
        """
        self.provider = provider.lower()
        self.model = model
        
        # Initialize appropriate client
        if self.provider == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI provider")
            self.client = OpenAIClient(api_key=api_key, model=model, **kwargs)
        elif self.provider == "gemini":
            if not api_key:
                raise ValueError("API key required for Gemini provider")
            self.client = GeminiClient(api_key=api_key, model=model, **kwargs)
        elif self.provider == "local":
            if not local_model_path:
                raise ValueError("Model path required for local provider")
            self.client = LocalLLMClient(model_path=local_model_path, model=model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Initialized LLM reconstructor with {provider} provider, model: {model}")
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        # Temporarily override max_tokens if specified
        original_max_tokens = self.client.max_tokens
        if max_tokens != 1000:  # Only override if different from default
            self.client.max_tokens = max_tokens
        
        try:
            response = await self.client.generate_text(prompt)
            if response.success:
                return response.text
            else:
                logger.error(f"Text generation failed: {response.error_message}")
                return ""
        finally:
            # Restore original max_tokens
            self.client.max_tokens = original_max_tokens
    
    async def reconstruct_markdown(self, 
                                 candidates: List[str], 
                                 structure_info: Dict[str, Any],
                                 context: str = "") -> str:
        """Reconstruct markdown from candidates and structure information.
        
        Args:
            candidates: List of candidate words/phrases from similarity search
            structure_info: Information about markdown structure
            context: Additional context for reconstruction
            
        Returns:
            Reconstructed markdown text
        """
        # This will be implemented in the prompt construction system (task 6.2)
        # For now, return a placeholder
        prompt = f"Reconstruct markdown using candidates: {candidates[:10]}"  # Limit for brevity
        return await self.generate_text(prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics.
        
        Returns:
            Dictionary with client statistics
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "rate_limit_calls": len(self.client.rate_limiter.calls),
            "rate_limit_max": self.client.rate_limiter.max_calls,
            "rate_limit_window": self.client.rate_limiter.time_window
        }


# Synchronous wrapper for backward compatibility
class SyncLLMReconstructor:
    """Synchronous wrapper for LLMReconstructor."""
    
    def __init__(self, *args, **kwargs):
        """Initialize sync wrapper."""
        self.async_reconstructor = LLMReconstructor(*args, **kwargs)
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """Synchronous text generation."""
        return asyncio.run(self.async_reconstructor.generate_text(prompt, max_tokens))
    
    def reconstruct_markdown(self, 
                           candidates: List[str], 
                           structure_info: Dict[str, Any],
                           context: str = "") -> str:
        """Synchronous markdown reconstruction."""
        return asyncio.run(self.async_reconstructor.reconstruct_markdown(
            candidates, structure_info, context
        ))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return self.async_reconstructor.get_stats()