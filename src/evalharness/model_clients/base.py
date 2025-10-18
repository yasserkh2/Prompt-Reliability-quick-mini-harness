"""
Base protocol for model clients.

All LLM adapters should implement the ModelClient interface.
"""

from typing import Protocol, List, Dict, Any, Optional


class ModelClient(Protocol):
    """
    Protocol defining the interface for LLM clients.
    
    Any model adapter (OpenAI, Anthropic, etc.) should implement this interface.
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a single response from the model.
        
        Args:
            prompt: The input prompt text
            **kwargs: Model-specific parameters (temperature, max_tokens, etc.)
            
        Returns:
            The generated text response
            
        Raises:
            Exception: If the API call fails
        """
        ...
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        This is an optional optimization. Default implementation can call
        generate() in a loop.
        
        Args:
            prompts: List of input prompt texts
            **kwargs: Model-specific parameters
            
        Returns:
            List of generated text responses (same order as prompts)
            
        Raises:
            Exception: If any API call fails
        """
        ...
    
    @property
    def model_name(self) -> str:
        """Return the name/identifier of the model being used."""
        ...
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return the configuration dictionary for this client."""
        ...


class BaseModelClient:
    """
    Base class with common functionality for model clients.
    
    Concrete clients can inherit from this to get default implementations.
    """
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base client.
        
        Args:
            model_name: Identifier for the model
            config: Configuration dictionary
        """
        self._model_name = model_name
        self._config = config or {}
    
    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return the configuration."""
        return self._config
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Default implementation: call generate() for each prompt.
        
        Subclasses can override for better batch performance.
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate()")
