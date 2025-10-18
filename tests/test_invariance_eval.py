"""
Unit tests for invariance evaluator.
"""

import pytest
from evalharness.evaluators.base import TestCase
from evalharness.evaluators.invariance import (
    InvarianceEvaluator,
    case_variation,
    whitespace_variation,
    punctuation_variation,
    politeness_prefix,
    rephrasing_simple,
    instruction_format_variation,
    numerical_format_variation,
    spelling_typos_minor,
    context_neutral_additions,
    language_formality_shift
)


class MockModelClient:
    """Mock model client for testing."""
    
    def __init__(self, response: str = "Paris"):
        self.response = response
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        return self.response
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        self.call_count += len(prompts)
        return [self.response] * len(prompts)
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    @property
    def config(self) -> dict:
        return {}


class TestTransformationFunctions:
    """Tests for individual transformation functions."""
    
    def test_case_variation(self):
        variants = case_variation("What is Paris?")
        assert len(variants) == 3
        assert "what is paris?" in variants  # lowercase
        assert "WHAT IS PARIS?" in variants  # uppercase
        assert "What Is Paris?" in variants  # title case
    
    def test_whitespace_variation(self):
        variants = whitespace_variation("What is Paris?")
        assert len(variants) > 0
        # Check for extra spaces
        assert any("  " in v for v in variants)
    
    def test_punctuation_variation(self):
        variants = punctuation_variation("What is Paris")
        assert len(variants) > 0
        # Should have versions with ?, ., !, and without
        assert any(v.endswith("?") for v in variants)
        assert any(v.endswith(".") for v in variants)
    
    def test_politeness_prefix(self):
        variants = politeness_prefix("What is Paris?")
        assert len(variants) >= 5
        assert any("please" in v.lower() for v in variants)
        assert any("could you" in v.lower() for v in variants)
    
    def test_rephrasing_simple(self):
        variants = rephrasing_simple("What is HTTP?")
        # Should generate rephrasings
        assert len(variants) > 0
        assert any("tell me" in v.lower() or "explain" in v.lower() for v in variants)
    
    def test_instruction_format_variation(self):
        variants = instruction_format_variation("What is Paris?")
        assert len(variants) >= 5
        assert any(v.startswith("Question:") for v in variants)
        assert any(v.startswith("Q:") for v in variants)
    
    def test_numerical_format_variation(self):
        variants = numerical_format_variation("What is 2 + 2?")
        assert len(variants) > 0
        # Should convert numbers to words
        assert any("two" in v.lower() for v in variants)
    
    def test_spelling_typos_minor(self):
        variants = spelling_typos_minor("What is the capital?")
        # Should generate some typos
        assert len(variants) > 0
    
    def test_context_neutral_additions(self):
        variants = context_neutral_additions("What is Paris?")
        assert len(variants) >= 5
        assert any("quick question" in v.lower() for v in variants)
    
    def test_language_formality_shift(self):
        # Test contraction expansion
        variants = language_formality_shift("What's the capital?")
        assert any("what is" in v.lower() for v in variants)
        
        # Test contraction addition
        variants = language_formality_shift("What is the capital?")
        assert any("what's" in v.lower() for v in variants)


class TestInvarianceEvaluator:
    """Tests for InvarianceEvaluator class."""
    
    def test_init_default(self):
        evaluator = InvarianceEvaluator()
        # Should include all 10 transformations by default
        assert len(evaluator.transformations) == 10
    
    def test_init_custom_transformations(self):
        evaluator = InvarianceEvaluator(transformations=["case", "whitespace"])
        assert len(evaluator.transformations) == 2
        assert "case" in evaluator.transformations
        assert "whitespace" in evaluator.transformations
    
    def test_init_invalid_transformation(self):
        with pytest.raises(ValueError):
            InvarianceEvaluator(transformations=["invalid"])
    
    def test_generate_variants(self):
        evaluator = InvarianceEvaluator(transformations=["case", "punctuation"])
        variants = evaluator.generate_variants("What is Paris?")
        
        assert "case" in variants
        assert "punctuation" in variants
        assert len(variants["case"]) > 0
        assert len(variants["punctuation"]) > 0
    
    def test_evaluate_single_test_case(self):
        evaluator = InvarianceEvaluator(transformations=["case"])
        test_case = TestCase(
            id="test1",
            input="What is the capital of France?",
            output="Paris"
        )
        
        # Mock client that always returns "Paris"
        mock_client = MockModelClient(response="Paris")
        
        result = evaluator.evaluate(test_case, mock_client)
        
        assert result.test_case_id == "test1"
        assert result.original_prompt == "What is the capital of France?"
        assert result.expected_output == "Paris"
        assert result.variants_tested > 0
        assert result.passed + result.failed == result.variants_tested
        assert 0.0 <= result.consistency_score <= 1.0
    
    def test_evaluate_all_match(self):
        evaluator = InvarianceEvaluator(transformations=["case"])
        test_case = TestCase(
            id="test1",
            input="What is Paris?",
            output="Paris"
        )
        
        mock_client = MockModelClient(response="Paris")
        result = evaluator.evaluate(test_case, mock_client)
        
        # All should pass
        assert result.passed == result.variants_tested
        assert result.failed == 0
        assert result.consistency_score == 1.0
    
    def test_evaluate_none_match(self):
        evaluator = InvarianceEvaluator(transformations=["case"])
        test_case = TestCase(
            id="test1",
            input="What is Paris?",
            output="Paris"
        )
        
        # Mock returns wrong answer
        mock_client = MockModelClient(response="London")
        result = evaluator.evaluate(test_case, mock_client)
        
        # All should fail
        assert result.passed == 0
        assert result.failed == result.variants_tested
        assert result.consistency_score == 0.0
    
    def test_evaluate_batch(self):
        evaluator = InvarianceEvaluator(transformations=["case"])
        test_cases = [
            TestCase(id="test1", input="What is Paris?", output="Paris"),
            TestCase(id="test2", input="What is London?", output="London")
        ]
        
        mock_client = MockModelClient(response="Paris")
        results = evaluator.evaluate_batch(test_cases, mock_client)
        
        assert len(results) == 2
        assert results[0].test_case_id == "test1"
        assert results[1].test_case_id == "test2"
    
    def test_variant_results_metadata(self):
        evaluator = InvarianceEvaluator(transformations=["case", "punctuation"])
        test_case = TestCase(
            id="test1",
            input="What is Paris?",
            output="Paris"
        )
        
        mock_client = MockModelClient(response="Paris")
        result = evaluator.evaluate(test_case, mock_client)
        
        # Check that variant results have metadata
        for vr in result.variant_results:
            assert "transformation_type" in vr.metadata
            assert vr.metadata["transformation_type"] in ["case", "punctuation"]


class TestInvarianceEvaluatorIntegration:
    """Integration tests with more complex scenarios."""
    
    def test_all_transformations(self):
        """Test with all 10 transformations enabled."""
        evaluator = InvarianceEvaluator()  # All transformations
        test_case = TestCase(
            id="test1",
            input="What is 2 + 2?",
            output="4"
        )
        
        mock_client = MockModelClient(response="4")
        result = evaluator.evaluate(test_case, mock_client)
        
        # Should have many variants (10 transformation types)
        assert result.variants_tested > 10
        assert result.consistency_score == 1.0
    
    def test_mixed_results(self):
        """Test case where some variants pass and some fail."""
        
        class VariableClient:
            """Client that returns different answers based on prompt."""
            call_count = 0
            
            def generate(self, prompt, **kwargs):
                self.call_count += 1
                # Return wrong answer for uppercase prompts
                if prompt.isupper():
                    return "wrong"
                return "correct"
            
            def batch_generate(self, prompts, **kwargs):
                return [self.generate(p) for p in prompts]
            
            @property
            def model_name(self):
                return "variable-model"
            
            @property
            def config(self):
                return {}
        
        evaluator = InvarianceEvaluator(transformations=["case"])
        test_case = TestCase(
            id="test1",
            input="What is the answer?",
            output="correct"
        )
        
        client = VariableClient()
        result = evaluator.evaluate(test_case, client)
        
        # Should have mixed results
        assert result.passed > 0
        assert result.failed > 0
        assert 0.0 < result.consistency_score < 1.0
