"""
OpenAI API client adapter.

Supports OpenAI and Azure OpenAI endpoints.
"""

import os
from typing import List, Dict, Any, Optional
import time

try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AzureOpenAI = None

from .base import BaseModelClient


class OpenAIClient(BaseModelClient):
    """
    Client for OpenAI API (including Azure OpenAI).
    
    Supports:
    - OpenAI API (api.openai.com)
    - Azure OpenAI Service
    - Custom OpenAI-compatible endpoints
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        is_azure: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 100,
        timeout: int = 30,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model identifier (e.g., "gpt-3.5-turbo", "gpt-4")
            api_key: API key (or set OPENAI_API_KEY env var)
            api_base: API base URL (for Azure or custom endpoints)
            api_version: API version (for Azure)
            is_azure: Whether using Azure OpenAI
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        super().__init__(model_name, {
            "api_key": api_key,
            "api_base": api_base,
            "api_version": api_version,
            "is_azure": is_azure,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            **kwargs
        })
        
        # Get API key from config or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set via api_key parameter or OPENAI_API_KEY env var"
            )
        
        self.api_base = api_base
        self.api_version = api_version
        self.is_azure = is_azure
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = kwargs
        
        # Initialize OpenAI client
        if is_azure:
            if not api_base or not api_version:
                raise ValueError(
                    "Azure OpenAI requires api_base and api_version parameters"
                )
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=api_base,
                api_version=api_version,
                timeout=timeout
            )
        else:
            client_kwargs = {"api_key": self.api_key, "timeout": timeout}
            if api_base:
                client_kwargs["base_url"] = api_base
            self.client = OpenAI(**client_kwargs)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a single response from OpenAI.
        
        Args:
            prompt: Input prompt text
            **kwargs: Override default parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails
        """
        # Merge kwargs with defaults
        gen_kwargs = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        gen_kwargs.update({k: v for k, v in kwargs.items() 
                          if k not in ["temperature", "max_tokens"]})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **gen_kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Currently uses sequential calls. Can be optimized with async or batch API.
        
        Args:
            prompts: List of input prompts
            **kwargs: Override default parameters
            
        Returns:
            List of generated responses
        """
        # Simple sequential implementation
        # TODO: Implement async batching for better performance
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                # Store error as result
                results.append(f"[ERROR: {str(e)}]")
        
        return results
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OpenAIClient":
        """
        Create client from configuration dictionary.
        
        Args:
            config: Configuration dict with keys matching __init__ parameters
            
        Returns:
            Initialized OpenAIClient
        """
        return cls(**config)


class AzureOpenAIClient(OpenAIClient):
    """
    Convenience wrapper for Azure OpenAI.
    
    Automatically sets is_azure=True.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str = "2023-12-01-preview",
        **kwargs
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            model_name: Deployment name in Azure
            api_key: Azure API key
            azure_endpoint: Azure endpoint URL
            api_version: API version
            **kwargs: Additional parameters
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            api_base=azure_endpoint,
            api_version=api_version,
            is_azure=True,
            **kwargs
        )
