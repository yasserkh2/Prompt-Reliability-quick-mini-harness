"""
Perturbation evaluator for LLM robustness testing.

Tests how LLMs respond to distracting or misleading perturbations:
- Order changes in multiple-choice questions
- Addition of distractors
- Irrelevant context
- Contradictory information
- Position bias
- Misleading formatting

Different from invariance testing - these changes SHOULD be handled robustly
but are more challenging as they add noise or misleading information.
"""

import random
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import TestCase, EvaluationResult, VariantResult
from ..model_clients.base import ModelClient
from ..scoring import score_match


# ============================================================================
# PERTURBATION FUNCTIONS (10 Total)
# ============================================================================

def shuffle_answer_order(prompt: str, answer: Optional[str] = None) -> List[str]:
    """
    Perturbation 1: Shuffle the order of elements in lists or sequences.
    
    Tests if model is influenced by presentation order when order shouldn't matter.
    For questions with multiple valid orderings, tests position bias.
    
    Args:
        prompt: Original prompt
        answer: Expected answer (to avoid breaking correct order)
        
    Returns:
        List of prompts with shuffled elements
    """
    variants = []
    
    # Detect lists in prompt (comma-separated or "and" separated)
    if "," in prompt or " and " in prompt.lower():
        words = prompt.split()
        
        # Find segments that might be reorderable
        # Example: "A, B, and C" or "red, blue, green"
        list_pattern = r'([A-Za-z]+(?:,\s*[A-Za-z]+)+(?:,?\s+(?:and|or)\s+[A-Za-z]+)?)'
        matches = re.finditer(list_pattern, prompt)
        
        for match in matches:
            list_text = match.group(1)
            items = [item.strip() for item in re.split(r',|\s+and\s+|\s+or\s+', list_text)]
            
            if len(items) >= 2:
                # Create shuffled versions
                for _ in range(2):
                    shuffled = items.copy()
                    random.shuffle(shuffled)
                    shuffled_text = ", ".join(shuffled[:-1]) + " and " + shuffled[-1]
                    variant = prompt.replace(list_text, shuffled_text)
                    variants.append(variant)
    
    # If no lists found, try reversing word order in key phrases
    if not variants:
        # Try reversing phrases after "between" or "or"
        if " or " in prompt.lower():
            parts = prompt.split(" or ")
            if len(parts) == 2:
                variants.append(f"{parts[1]} or {parts[0]}")
    
    return variants if variants else [prompt]


def add_distractor_facts(prompt: str, domain: str = "general") -> List[str]:
    """
    Perturbation 2: Add factually correct but irrelevant information.
    
    Tests if model can filter relevant from irrelevant facts.
    
    Args:
        prompt: Original prompt
        domain: Domain for generating relevant distractors
        
    Returns:
        List of prompts with distractor facts
    """
    # Domain-specific distractors
    distractor_sets = {
        "geography": [
            "The Earth's circumference is approximately 40,075 kilometers.",
            "The Mariana Trench is the deepest point in the ocean.",
            "The Amazon rainforest produces 20% of the world's oxygen.",
        ],
        "general": [
            "The speed of light is 299,792,458 meters per second.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The human body contains approximately 37 trillion cells.",
            "Shakespeare wrote 37 plays during his lifetime.",
            "The Great Wall of China is visible from space.",
        ],
        "tech": [
            "The first computer mouse was invented in 1964.",
            "HTML was created by Tim Berners-Lee in 1991.",
            "The first iPhone was released in 2007.",
        ]
    }
    
    distractors = distractor_sets.get(domain, distractor_sets["general"])
    variants = []
    
    for distractor in distractors[:3]:  # Use up to 3 distractors
        # Add before
        variants.append(f"{distractor} However, {prompt}")
        # Add after
        variants.append(f"{prompt} Note: {distractor}")
        # Add in middle (if prompt has multiple sentences)
        if ". " in prompt:
            parts = prompt.split(". ", 1)
            variants.append(f"{parts[0]}. {distractor} {parts[1]}")
    
    return variants


def add_misleading_prefix(prompt: str, expected_answer: str) -> List[str]:
    """
    Perturbation 3: Add prefixes that might mislead toward wrong answer.
    
    Tests if model can resist suggestion bias.
    
    Args:
        prompt: Original prompt
        expected_answer: Expected answer (to generate misleading alternatives)
        
    Returns:
        List of prompts with misleading prefixes
    """
    misleading_templates = [
        f"Many people incorrectly think it's not {expected_answer}, but {prompt}",
        f"Some sources claim otherwise, however {prompt}",
        f"This is a tricky question. {prompt}",
        f"Be careful not to confuse this with similar concepts. {prompt}",
        f"Common mistake: assuming a different answer. {prompt}",
        f"Unlike what you might initially think, {prompt}",
    ]
    
    return misleading_templates


def add_position_bias_options(prompt: str, correct_answer: str) -> List[str]:
    """
    Perturbation 4: Place correct answer in different positions.
    
    Tests positional bias (tendency to pick first, last, or middle options).
    
    Args:
        prompt: Original prompt
        correct_answer: The correct answer
        
    Returns:
        List of prompts with answer in different positions
    """
    # Generate plausible wrong answers based on domain
    wrong_answers = [
        "London", "Berlin", "Madrid", "Rome",  # Cities
        "Blue", "Red", "Green", "Yellow",  # Colors
        "5", "10", "15", "20",  # Numbers
    ]
    
    variants = []
    
    # Position 1: Correct answer first
    options = [correct_answer] + wrong_answers[:3]
    variants.append(f"{prompt}\nOptions: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}")
    
    # Position 2: Correct answer second
    options = [wrong_answers[0], correct_answer] + wrong_answers[1:3]
    variants.append(f"{prompt}\nOptions: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}")
    
    # Position 3: Correct answer last
    options = wrong_answers[:3] + [correct_answer]
    variants.append(f"{prompt}\nOptions: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}")
    
    return variants


def add_verbose_wrapper(prompt: str) -> List[str]:
    """
    Perturbation 5: Wrap question in excessive verbose context.
    
    Tests if model can extract key question from verbose text.
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with verbose wrappers
    """
    wrappers = [
        (
            f"In the context of comprehensive knowledge evaluation and "
            f"assessment of fundamental concepts, considering various "
            f"perspectives and scholarly interpretations: {prompt} "
            f"Please provide a concise and accurate response."
        ),
        (
            f"Context: This is a standardized assessment question designed "
            f"to evaluate baseline knowledge. Instructions: Read carefully "
            f"and provide the most accurate answer. Question: {prompt} "
            f"Additional note: Answer should be brief and precise."
        ),
        (
            f"Dear respondent, we kindly request your attention to the "
            f"following inquiry which has been formulated for the purpose "
            f"of knowledge verification: {prompt} We appreciate your "
            f"cooperation in providing an accurate response."
        ),
        (
            f"[SYSTEM PROMPT] You are a knowledgeable assistant. "
            f"[USER QUERY] {prompt} [INSTRUCTIONS] Provide factual answer only. "
            f"[END INSTRUCTIONS]"
        ),
    ]
    
    return wrappers


def add_format_interference(prompt: str) -> List[str]:
    """
    Perturbation 6: Add formatting that might confuse parsing.
    
    Tests if model is robust to formatting variations.
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with format interference
    """
    variants = [
        # Multiple line breaks
        f"{prompt}\n\n\n\n",
        # Mixed with code-like formatting
        f"```\n{prompt}\n```",
        # With markdown-style emphasis
        f"**{prompt}**",
        # With HTML-like tags
        f"<question>{prompt}</question>",
        # With bullet point
        f"• {prompt}",
        # With numbering
        f"1. {prompt}",
        # ALL CAPS
        f"{prompt.upper()}",
    ]
    
    return variants


def add_semantic_distractors(prompt: str, topic: str = "general") -> List[str]:
    """
    Perturbation 7: Add semantically related but irrelevant information.
    
    Tests if model can distinguish between related and relevant information.
    
    Args:
        prompt: Original prompt
        topic: Topic for generating related distractors
        
    Returns:
        List of prompts with semantic distractors
    """
    # Topic-specific related but irrelevant info
    semantic_distractors = {
        "capital": [
            "The city has a population of over 2 million residents.",
            "It was founded in the 15th century.",
            "The city is known for its historic architecture.",
        ],
        "math": [
            "Mathematics has been studied for thousands of years.",
            "Calculators were invented in the 20th century.",
            "Many mathematicians have contributed to number theory.",
        ],
        "general": [
            "This is a commonly asked question in various contexts.",
            "Many people are curious about this topic.",
            "The answer has been well-established through research.",
        ]
    }
    
    distractors = semantic_distractors.get(topic, semantic_distractors["general"])
    variants = []
    
    for distractor in distractors:
        variants.append(f"{distractor} Now, {prompt}")
        variants.append(f"{prompt} By the way, {distractor}")
    
    return variants


def add_negation_flip(prompt: str) -> List[str]:
    """
    Perturbation 8: Add double negatives or negation that requires careful parsing.
    
    Tests if model handles logical negations correctly.
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with negation flips
    """
    variants = []
    
    # Add double negatives
    if "is" in prompt.lower():
        variants.append(prompt.replace(" is ", " is not not ", 1))
    
    # Add negation context
    variants.extend([
        f"Which of the following is NOT incorrect: {prompt}",
        f"It is not true that the answer is not what you'd expect. {prompt}",
        f"Disregarding incorrect options, {prompt}",
    ])
    
    return variants


def add_multi_part_distraction(prompt: str) -> List[str]:
    """
    Perturbation 9: Add multiple sub-questions or parts to distract from main question.
    
    Tests if model can identify and answer the primary question.
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with multi-part distractions
    """
    variants = [
        f"Part A: What day is today? Part B: {prompt} (Answer Part B only)",
        f"First, consider the weather. Second, {prompt} Focus on the second part.",
        f"Context question: How are you? Main question: {prompt} Answer the main question.",
        f"{prompt} Also, as a side note, who invented the telephone?",
        f"Question 1: What is 1+1? Question 2: {prompt} Answer Question 2.",
    ]
    
    return variants


def add_length_variation_extreme(prompt: str) -> List[str]:
    """
    Perturbation 10: Add extremely long or short variations.
    
    Tests if model performance degrades with extreme length variations.
    
    Args:
        prompt: Original prompt
        
    Returns:
        List of prompts with extreme length variations
    """
    variants = []
    
    # Extremely verbose version
    filler = (
        "In consideration of the vast body of human knowledge and the "
        "extensive research conducted by scholars across multiple disciplines, "
        "taking into account historical context, contemporary understanding, "
        "and the synthesis of empirical evidence gathered through rigorous "
        "scientific methodology, while acknowledging the complexity and nuance "
        "inherent in the subject matter, and recognizing that this represents "
        "a fundamental question that has been extensively studied and documented, "
    )
    variants.append(f"{filler} {prompt}")
    
    # Extremely terse version (if possible)
    # Try to compress the question
    compressed = prompt.replace(" is ", " ").replace(" the ", " ").replace("?", "")
    if compressed != prompt:
        variants.append(compressed)
    
    # Repetitive padding
    variants.append(
        f"Important: {prompt} This is important. {prompt} "
        f"Please note: {prompt}"
    )
    
    return variants


# ============================================================================
# PERTURBATION EVALUATOR CLASS
# ============================================================================

@dataclass
class PerturbationType:
    """Metadata about a perturbation type."""
    name: str
    function: Any
    description: str
    requires_answer: bool = False


class PerturbationEvaluator:
    """
    Evaluator that tests LLM robustness under perturbations.
    
    Unlike invariance testing (where output SHOULD be identical),
    perturbation testing checks if the model maintains correct answers
    despite distracting or misleading changes.
    """
    
    # All available perturbations
    PERTURBATIONS: Dict[str, PerturbationType] = {
        "order_shuffle": PerturbationType(
            "order_shuffle",
            shuffle_answer_order,
            "Shuffles order of elements in lists/sequences",
            requires_answer=False
        ),
        "distractor_facts": PerturbationType(
            "distractor_facts",
            add_distractor_facts,
            "Adds factually correct but irrelevant information",
            requires_answer=False
        ),
        "misleading_prefix": PerturbationType(
            "misleading_prefix",
            add_misleading_prefix,
            "Adds prefixes that might mislead toward wrong answer",
            requires_answer=True
        ),
        "position_bias": PerturbationType(
            "position_bias",
            add_position_bias_options,
            "Tests positional bias with correct answer in different positions",
            requires_answer=True
        ),
        "verbose_wrapper": PerturbationType(
            "verbose_wrapper",
            add_verbose_wrapper,
            "Wraps question in excessive verbose context",
            requires_answer=False
        ),
        "format_interference": PerturbationType(
            "format_interference",
            add_format_interference,
            "Adds formatting that might confuse parsing",
            requires_answer=False
        ),
        "semantic_distractors": PerturbationType(
            "semantic_distractors",
            add_semantic_distractors,
            "Adds semantically related but irrelevant information",
            requires_answer=False
        ),
        "negation_flip": PerturbationType(
            "negation_flip",
            add_negation_flip,
            "Adds double negatives requiring careful parsing",
            requires_answer=False
        ),
        "multi_part": PerturbationType(
            "multi_part",
            add_multi_part_distraction,
            "Adds multiple sub-questions to distract from main question",
            requires_answer=False
        ),
        "length_extreme": PerturbationType(
            "length_extreme",
            add_length_variation_extreme,
            "Tests extremely long or short variations",
            requires_answer=False
        ),
    }
    
    def __init__(
        self,
        perturbation_types: List[str] = None,
        scoring_method: str = "normalized_equal",
        scoring_threshold: float = 0.8,
        similarity_method: str = "overlap"
    ):
        """
        Initialize perturbation evaluator.
        
        Args:
            perturbation_types: Types of perturbations to apply (default: all)
            scoring_method: Method for comparing outputs
            scoring_threshold: Threshold for similarity scoring
            similarity_method: Similarity algorithm to use ("jaccard", "overlap", "levenshtein")
        """
        if perturbation_types is None:
            self.perturbation_types = list(self.PERTURBATIONS.keys())
        else:
            # Validate perturbation types
            invalid = set(perturbation_types) - set(self.PERTURBATIONS.keys())
            if invalid:
                raise ValueError(f"Invalid perturbation types: {invalid}")
            self.perturbation_types = perturbation_types
        
        self.scoring_method = scoring_method
        self.scoring_threshold = scoring_threshold
        self.similarity_method = similarity_method
    
    def generate_perturbations(
        self,
        prompt: str,
        expected: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, List[str]]:
        """
        Generate perturbations for a prompt using configured perturbation types.
        
        Args:
            prompt: Original prompt
            expected: Expected answer
            metadata: Additional metadata (e.g., options for MCQ)
            
        Returns:
            Dictionary of perturbation type -> list of perturbed prompts
        """
        perturbations = {}
        
        for perturb_name in self.perturbation_types:
            perturb_info = self.PERTURBATIONS[perturb_name]
            
            # Call the perturbation function
            if perturb_info.requires_answer:
                variants = perturb_info.function(prompt, expected)
            else:
                variants = perturb_info.function(prompt)
            
            perturbations[perturb_name] = variants
        
        return perturbations
    
    def evaluate(self, test_case: TestCase, model_client: ModelClient) -> EvaluationResult:
        """
        Evaluate robustness under perturbations.
        
        Args:
            test_case: Test case to evaluate
            model_client: Model client to query
            
        Returns:
            Evaluation result with robustness scores
        """
        # Generate perturbations
        perturbations_by_type = self.generate_perturbations(
            test_case.input,
            test_case.output,
            test_case.metadata
        )
        
        # Flatten
        all_perturbations = []
        perturbation_sources = []
        for perturb_type, variants in perturbations_by_type.items():
            for variant in variants:
                all_perturbations.append(variant)
                perturbation_sources.append(perturb_type)
        
        # Query model
        try:
            outputs = model_client.batch_generate(all_perturbations)
        except:
            outputs = [model_client.generate(p) for p in all_perturbations]
        
        # Score
        variant_results = []
        passed = 0
        failed = 0
        total_score = 0.0
        
        for perturb, output, source in zip(all_perturbations, outputs, perturbation_sources):
            match, score = score_match(
                expected=test_case.output,
                actual=output,
                method=self.scoring_method,
                threshold=self.scoring_threshold,
                similarity_method=self.similarity_method
            )
            
            variant_result = VariantResult(
                variant=perturb,
                actual_output=output,
                expected_output=test_case.output,
                match=match,
                match_score=score,
                metadata={"perturbation_type": source}
            )
            
            variant_results.append(variant_result)
            total_score += score
            
            if match:
                passed += 1
            else:
                failed += 1
        
        # Calculate robustness score
        robustness_score = total_score / len(all_perturbations) if all_perturbations else 0.0
        
        return EvaluationResult(
            test_case_id=test_case.id,
            original_prompt=test_case.input,
            expected_output=test_case.output,
            variants_tested=len(all_perturbations),
            passed=passed,
            failed=failed,
            consistency_score=robustness_score,
            variant_results=variant_results,
            metadata={
                "perturbation_types": self.perturbation_types,
                "scoring_method": self.scoring_method
            }
        )
    
    def evaluate_batch(
        self,
        test_cases: List[TestCase],
        model_client: ModelClient
    ) -> List[EvaluationResult]:
        """Evaluate multiple test cases."""
        return [self.evaluate(tc, model_client) for tc in test_cases]
