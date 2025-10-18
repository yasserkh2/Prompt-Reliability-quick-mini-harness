"""
Invariance evaluator for LLM prompt testing.

Tests whether LLMs produce consistent answers under label-preserving transformations.
Includes 10 different transformation types for comprehensive robustness testing.
"""

import re
import random
from typing import List, Callable, Dict, Any
from dataclasses import dataclass

from .base import TestCase, EvaluationResult, VariantResult
from ..model_clients.base import ModelClient
from ..scoring import score_match


# ============================================================================
# TRANSFORMATION FUNCTIONS (10 Total)
# ============================================================================

def case_variation(prompt: str) -> List[str]:
    """
    Generate case variations of the prompt.
    
    Transformations:
    - All lowercase
    - ALL UPPERCASE
    - Title Case
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of case-varied prompts
    """
    variants = []
    
    # Lowercase
    variants.append(prompt.lower())
    
    # Uppercase
    variants.append(prompt.upper())
    
    # Title case
    variants.append(prompt.title())
    
    return variants


def whitespace_variation(prompt: str) -> List[str]:
    """
    Generate whitespace variations.
    
    Transformations:
    - Extra spaces between words
    - Leading/trailing whitespace
    - Tab characters instead of spaces
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of whitespace-varied prompts
    """
    variants = []
    
    # Extra spaces between words
    variants.append(re.sub(r' ', '  ', prompt))
    
    # Leading and trailing spaces
    variants.append(f"  {prompt}  ")
    variants.append(f"\n{prompt}\n")
    
    # Tabs
    variants.append(prompt.replace(' ', '\t'))
    
    return variants


def punctuation_variation(prompt: str) -> List[str]:
    """
    Generate punctuation variations.
    
    Transformations:
    - Add question mark if missing
    - Remove trailing punctuation
    - Add period if missing
    - Add exclamation mark
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of punctuation-varied prompts
    """
    variants = []
    
    # Strip existing punctuation
    stripped = prompt.rstrip('?.!')
    
    # Add different punctuation
    if not prompt.endswith('?'):
        variants.append(f"{stripped}?")
    
    if not prompt.endswith('.'):
        variants.append(f"{stripped}.")
    
    if not prompt.endswith('!'):
        variants.append(f"{stripped}!")
    
    # Remove all punctuation
    variants.append(stripped)
    
    return variants


def politeness_prefix(prompt: str) -> List[str]:
    """
    Add polite prefixes to the prompt.
    
    Transformations:
    - "Please" prefix
    - "Could you" prefix
    - "I'd like to know" prefix
    - "Kindly tell me" prefix
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with politeness prefixes
    """
    prefixes = [
        "Please",
        "Could you tell me",
        "I'd like to know",
        "Kindly tell me",
        "Can you please answer",
    ]
    
    variants = []
    for prefix in prefixes:
        # Handle if prompt already has question mark
        if prompt.endswith('?'):
            variants.append(f"{prefix}: {prompt}")
        else:
            variants.append(f"{prefix}: {prompt}?")
    
    return variants


def rephrasing_simple(prompt: str) -> List[str]:
    """
    Generate simple rephrasings using common patterns.
    
    Transformations:
    - "What is X?" → "Tell me about X"
    - "X meaning" → "What does X mean?"
    - Question → Statement form
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of rephrased prompts
    """
    variants = []
    prompt_lower = prompt.lower()
    
    # Pattern: "what is X?" → "tell me about X"
    match = re.match(r'what is (.*?)[\?.]?$', prompt_lower, re.IGNORECASE)
    if match:
        subject = match.group(1)
        variants.append(f"Tell me about {subject}")
        variants.append(f"Explain {subject}")
        variants.append(f"Define {subject}")
    
    # Pattern: "X meaning" → "What does X mean?"
    match = re.match(r'(.*?) meaning[\?.]?$', prompt_lower, re.IGNORECASE)
    if match:
        subject = match.group(1)
        variants.append(f"What does {subject} mean?")
        variants.append(f"What is the meaning of {subject}?")
    
    # Pattern: "Which X?" → "What X?"
    if prompt_lower.startswith('which'):
        variants.append(prompt.replace('Which', 'What', 1).replace('which', 'what', 1))
    
    # Pattern: "What X?" → "Which X?"
    if prompt_lower.startswith('what'):
        variants.append(prompt.replace('What', 'Which', 1).replace('what', 'which', 1))
    
    return variants if variants else [prompt]  # Return original if no patterns matched


def instruction_format_variation(prompt: str) -> List[str]:
    """
    Change instruction formatting styles.
    
    Transformations:
    - "Question: X"
    - "Q: X"
    - "User asks: X"
    - "Answer the following: X"
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of instruction-formatted prompts
    """
    formats = [
        f"Question: {prompt}",
        f"Q: {prompt}",
        f"User asks: {prompt}",
        f"Answer the following: {prompt}",
        f"[Question] {prompt}",
    ]
    
    return formats


def numerical_format_variation(prompt: str) -> List[str]:
    """
    Vary numerical representations in the prompt.
    
    Transformations:
    - "2 + 2" → "two plus two"
    - "100°C" → "100 degrees Celsius"
    - Numbers to words and vice versa
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of numerically-varied prompts
    """
    variants = []
    
    # Number word mappings
    num_to_word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
    }
    word_to_num = {v: k for k, v in num_to_word.items()}
    
    # Convert digits to words
    for digit, word in num_to_word.items():
        if digit in prompt:
            variants.append(prompt.replace(digit, word))
    
    # Convert words to digits
    for word, digit in word_to_num.items():
        if word in prompt.lower():
            variants.append(re.sub(word, digit, prompt, flags=re.IGNORECASE))
    
    # Math operators
    replacements = [
        ('+', 'plus'),
        ('-', 'minus'),
        ('*', 'times'),
        ('/', 'divided by'),
    ]
    
    for symbol, word in replacements:
        if symbol in prompt:
            variants.append(prompt.replace(symbol, word))
    
    return variants if variants else [prompt]


def spelling_typos_minor(prompt: str) -> List[str]:
    """
    Introduce minor spelling typos.
    
    Transformations:
    - Swap adjacent characters
    - Drop a character
    - Duplicate a character
    
    Note: Only 1-2 typos to keep prompt understandable
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with minor typos
    """
    variants = []
    words = prompt.split()
    
    if len(words) == 0:
        return [prompt]
    
    # Typo type 1: Swap adjacent characters in a word
    for i, word in enumerate(words):
        if len(word) > 3:  # Only typo longer words
            pos = random.randint(0, len(word) - 2)
            typo_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            typo_prompt = ' '.join(words[:i] + [typo_word] + words[i+1:])
            variants.append(typo_prompt)
            break
    
    # Typo type 2: Drop a character
    for i, word in enumerate(words):
        if len(word) > 4:
            pos = random.randint(1, len(word) - 2)
            typo_word = word[:pos] + word[pos+1:]
            typo_prompt = ' '.join(words[:i] + [typo_word] + words[i+1:])
            variants.append(typo_prompt)
            break
    
    # Typo type 3: Duplicate a character
    for i, word in enumerate(words):
        if len(word) > 3:
            pos = random.randint(1, len(word) - 1)
            typo_word = word[:pos] + word[pos] + word[pos:]
            typo_prompt = ' '.join(words[:i] + [typo_word] + words[i+1:])
            variants.append(typo_prompt)
            break
    
    return variants if variants else [prompt]


def context_neutral_additions(prompt: str) -> List[str]:
    """
    Add neutral contextual phrases that don't change the question.
    
    Transformations:
    - "Quick question: X"
    - "In general, X"
    - "For reference, X"
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with neutral context
    """
    contexts = [
        f"Quick question: {prompt}",
        f"In general, {prompt}",
        f"For reference, {prompt}",
        f"Just curious: {prompt}",
        f"By the way, {prompt}",
    ]
    
    return contexts


def language_formality_shift(prompt: str) -> List[str]:
    """
    Shift between formal and informal language.
    
    Transformations:
    - "What's" → "What is"
    - "don't" → "do not"
    - Contractions expansion/addition
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of formality-shifted prompts
    """
    variants = []
    
    # Contractions to expand (informal → formal)
    contractions = {
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "it's": "it is",
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
    }
    
    # Expand contractions
    expanded = prompt
    for contraction, expansion in contractions.items():
        expanded = re.sub(contraction, expansion, expanded, flags=re.IGNORECASE)
    if expanded != prompt:
        variants.append(expanded)
    
    # Add contractions (formal → informal)
    contracted = prompt
    for contraction, expansion in contractions.items():
        contracted = re.sub(expansion, contraction, contracted, flags=re.IGNORECASE)
    if contracted != prompt:
        variants.append(contracted)
    
    return variants if variants else [prompt]


# ============================================================================
# INVARIANCE EVALUATOR CLASS
# ============================================================================

@dataclass
class TransformationType:
    """Metadata about a transformation type."""
    name: str
    function: Callable[[str], List[str]]
    description: str


class InvarianceEvaluator:
    """
    Evaluator that tests LLM invariance under prompt transformations.
    
    Tests whether the model produces consistent outputs when the prompt
    is transformed in ways that preserve semantic meaning.
    """
    
    # All available transformations
    TRANSFORMATIONS: Dict[str, TransformationType] = {
        "case": TransformationType(
            "case_variation",
            case_variation,
            "Tests lowercase, UPPERCASE, Title Case variations"
        ),
        "whitespace": TransformationType(
            "whitespace_variation",
            whitespace_variation,
            "Tests extra spaces, leading/trailing whitespace, tabs"
        ),
        "punctuation": TransformationType(
            "punctuation_variation",
            punctuation_variation,
            "Tests adding/removing question marks, periods, exclamation marks"
        ),
        "politeness": TransformationType(
            "politeness_prefix",
            politeness_prefix,
            "Tests polite prefixes like 'Please', 'Could you'"
        ),
        "rephrasing": TransformationType(
            "rephrasing_simple",
            rephrasing_simple,
            "Tests simple rephrasings with equivalent meaning"
        ),
        "instruction": TransformationType(
            "instruction_format_variation",
            instruction_format_variation,
            "Tests different instruction formats: 'Q:', 'Question:', etc."
        ),
        "numerical": TransformationType(
            "numerical_format_variation",
            numerical_format_variation,
            "Tests number words vs digits: '2' vs 'two'"
        ),
        "typos": TransformationType(
            "spelling_typos_minor",
            spelling_typos_minor,
            "Tests minor spelling errors (1-2 character changes)"
        ),
        "context": TransformationType(
            "context_neutral_additions",
            context_neutral_additions,
            "Tests neutral context additions: 'Quick question:', etc."
        ),
        "formality": TransformationType(
            "language_formality_shift",
            language_formality_shift,
            "Tests formal vs informal: 'what's' vs 'what is'"
        ),
    }
    
    def __init__(
        self,
        transformations: List[str] = None,
        scoring_method: str = "normalized_equal",
        scoring_threshold: float = 0.8,
        similarity_method: str = "overlap"
    ):
        """
        Initialize the invariance evaluator.
        
        Args:
            transformations: List of transformation names to use (default: all)
            scoring_method: Method for comparing outputs
            scoring_threshold: Threshold for similarity-based scoring
            similarity_method: Similarity algorithm to use ("jaccard", "overlap", "levenshtein")
        """
        if transformations is None:
            self.transformations = list(self.TRANSFORMATIONS.keys())
        else:
            # Validate transformation names
            invalid = set(transformations) - set(self.TRANSFORMATIONS.keys())
            if invalid:
                raise ValueError(f"Invalid transformations: {invalid}")
            self.transformations = transformations
        
        self.scoring_method = scoring_method
        self.scoring_threshold = scoring_threshold
        self.similarity_method = similarity_method
    
    def generate_variants(self, prompt: str) -> Dict[str, List[str]]:
        """
        Generate all variants for a prompt using configured transformations.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Dictionary mapping transformation name to list of variants
        """
        all_variants = {}
        
        for trans_name in self.transformations:
            trans_info = self.TRANSFORMATIONS[trans_name]
            variants = trans_info.function(prompt)
            all_variants[trans_name] = variants
        
        return all_variants
    
    def evaluate(self, test_case: TestCase, model_client: ModelClient) -> EvaluationResult:
        """
        Evaluate a single test case for invariance.
        
        Args:
            test_case: Test case with input and expected output
            model_client: Model client to query
            
        Returns:
            Evaluation result with consistency scores
        """
        # Generate all variants
        variants_by_type = self.generate_variants(test_case.input)
        
        # Flatten variants and track which transformation they came from
        all_variants = []
        variant_sources = []
        for trans_name, variants in variants_by_type.items():
            for variant in variants:
                all_variants.append(variant)
                variant_sources.append(trans_name)
        
        # Query model for all variants
        try:
            outputs = model_client.batch_generate(all_variants)
        except Exception as e:
            # Fallback to sequential if batch fails
            outputs = [model_client.generate(v) for v in all_variants]
        
        # Score each variant
        variant_results = []
        passed = 0
        failed = 0
        total_score = 0.0
        
        for variant, output, source in zip(all_variants, outputs, variant_sources):
            match, score = score_match(
                expected=test_case.output,
                actual=output,
                method=self.scoring_method,
                threshold=self.scoring_threshold,
                similarity_method=self.similarity_method
            )
            
            variant_result = VariantResult(
                variant=variant,
                actual_output=output,
                expected_output=test_case.output,
                match=match,
                match_score=score,
                metadata={"transformation_type": source}
            )
            
            variant_results.append(variant_result)
            total_score += score
            
            if match:
                passed += 1
            else:
                failed += 1
        
        # Calculate overall consistency score
        consistency_score = total_score / len(all_variants) if all_variants else 0.0
        
        return EvaluationResult(
            test_case_id=test_case.id,
            original_prompt=test_case.input,
            expected_output=test_case.output,
            variants_tested=len(all_variants),
            passed=passed,
            failed=failed,
            consistency_score=consistency_score,
            variant_results=variant_results,
            metadata={
                "transformations_used": self.transformations,
                "scoring_method": self.scoring_method,
                "scoring_threshold": self.scoring_threshold
            }
        )
    
    def evaluate_batch(
        self,
        test_cases: List[TestCase],
        model_client: ModelClient
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of test cases
            model_client: Model client to query
            
        Returns:
            List of evaluation results
        """
        return [self.evaluate(tc, model_client) for tc in test_cases]
