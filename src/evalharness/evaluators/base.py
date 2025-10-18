"""
Base protocols and dataclasses for evaluators.

Defines the interface that all evaluators must implement and common result structures.
"""

from dataclasses import dataclass, field
from typing import Protocol, List, Dict, Any, Optional
from enum import Enum


class EvaluationType(Enum):
    """Types of evaluations supported."""
    INVARIANCE = "invariance"
    PERTURBATION = "perturbation"


@dataclass
class VariantResult:
    """Result for a single prompt variant."""
    variant: str
    actual_output: str
    expected_output: str
    match: bool
    match_score: float = 0.0  # 0.0 to 1.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a test case."""
    test_case_id: str
    original_prompt: str
    expected_output: str
    variants_tested: int
    passed: int
    failed: int
    consistency_score: float  # Overall score (0.0 to 1.0)
    variant_results: List[VariantResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.variants_tested == 0:
            return 0.0
        return (self.passed / self.variants_tested) * 100


@dataclass
class TestCase:
    """A single test case with input and expected output."""
    id: str
    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    """Protocol for LLM clients."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text response
        """
        ...
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts (optional optimization).
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of generated text responses
        """
        ...


class Evaluator(Protocol):
    """Protocol for evaluators."""
    
    def evaluate(self, test_case: TestCase, model_client: ModelClient) -> EvaluationResult:
        """
        Evaluate a test case using the model client.
        
        Args:
            test_case: The test case to evaluate
            model_client: The model client to query
            
        Returns:
            Evaluation result with scores and details
        """
        ...
    
    def evaluate_batch(
        self, 
        test_cases: List[TestCase], 
        model_client: ModelClient
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases
            model_client: The model client to query
            
        Returns:
            List of evaluation results
        """
        ...


@dataclass
class Config:
    """Configuration for evaluation runs."""
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 100
    timeout: int = 30
    extra_params: Dict[str, Any] = field(default_factory=dict)
