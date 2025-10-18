"""
Scoring utilities for comparing LLM outputs to expected answers.

Provides multiple comparison strategies:
- Exact match (after normalization)
- Regex pattern matching
- Semantic similarity
"""

import re
from typing import Optional, Union
import unicodedata


def normalize(text: str) -> str:
    """
    Normalize text for comparison.
    
    Performs:
    - Unicode normalization (NFKC)
    - Lowercase conversion
    - Whitespace normalization (collapse multiple spaces, strip)
    - Remove common punctuation from ends
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Lowercase
    text = text.lower()
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    # Strip common trailing punctuation
    text = text.strip('.,!?;: \t\n\r')
    
    return text


def normalized_equal(text1: str, text2: str, strict: bool = False) -> bool:
    """
    Check if two texts are equal after normalization.
    
    Args:
        text1: First text
        text2: Second text
        strict: If True, only normalize whitespace; if False, full normalization
        
    Returns:
        True if texts match after normalization
    """
    if strict:
        # Only normalize whitespace
        t1 = ' '.join(str(text1).split())
        t2 = ' '.join(str(text2).split())
        return t1 == t2
    else:
        # Full normalization
        return normalize(text1) == normalize(text2)


def regex_search(pattern: str, text: str, case_sensitive: bool = False) -> bool:
    """
    Check if a regex pattern matches anywhere in the text.
    
    Useful for flexible matching when exact answers may vary in format.
    
    Args:
        pattern: Regex pattern string
        text: Text to search in
        case_sensitive: Whether to use case-sensitive matching
        
    Returns:
        True if pattern found in text
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return bool(re.search(pattern, text, flags=flags))
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")


def contains_substring(expected: str, actual: str, case_sensitive: bool = False) -> bool:
    """
    Check if expected text appears as substring in actual output.
    
    Args:
        expected: Expected substring
        actual: Actual output text
        case_sensitive: Whether to match case
        
    Returns:
        True if expected is substring of actual
    """
    if not case_sensitive:
        expected = expected.lower()
        actual = actual.lower()
    return expected in actual


def similarity_score(text1: str, text2: str, method: str = "jaccard") -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method - "jaccard", "overlap", or "levenshtein"
        
    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match)
    """
    # Normalize first
    t1 = normalize(text1)
    t2 = normalize(text2)
    
    if method == "jaccard":
        return _jaccard_similarity(t1, t2)
    elif method == "overlap":
        return _overlap_coefficient(t1, t2)
    elif method == "levenshtein":
        return _levenshtein_similarity(t1, t2)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def _jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity (intersection over union) of word sets.
    
    Args:
        text1: First text (already normalized)
        text2: Second text (already normalized)
        
    Returns:
        Jaccard similarity score
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def _overlap_coefficient(text1: str, text2: str) -> float:
    """
    Calculate overlap coefficient (intersection over minimum).
    
    More lenient than Jaccard for cases where one text is much longer.
    
    Args:
        text1: First text (already normalized)
        text2: Second text (already normalized)
        
    Returns:
        Overlap coefficient score
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    min_size = min(len(words1), len(words2))
    
    return intersection / min_size if min_size > 0 else 0.0


def _levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity based on Levenshtein (edit) distance.
    
    Args:
        text1: First text (already normalized)
        text2: Second text (already normalized)
        
    Returns:
        Similarity score (1 - normalized_distance)
    """
    # Simple Levenshtein distance implementation
    if text1 == text2:
        return 1.0
    
    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Create distance matrix
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize first row and column
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    
    # Calculate distances
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if text1[i-1] == text2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    
    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    
    return 1.0 - (distance / max_len)


def score_match(
    expected: str,
    actual: str,
    method: str = "normalized_equal",
    threshold: float = 0.8,
    **kwargs
) -> tuple[bool, float]:
    """
    Score a match using the specified method.
    
    Args:
        expected: Expected output
        actual: Actual output
        method: Scoring method - "normalized_equal", "regex", "similarity", "contains"
        threshold: Threshold for similarity-based methods (0.0 to 1.0)
        **kwargs: Additional arguments for specific methods
            - similarity_method: "jaccard", "overlap", or "levenshtein" (default: "overlap")
            - strict: For normalized_equal, only normalize whitespace
            - case_sensitive: For regex/contains matching
        
    Returns:
        Tuple of (is_match: bool, score: float)
    
    Recommendations:
        - Use "contains" for verbose LLM responses where expected is a substring
        - Use "similarity" with "overlap" method for verbose responses
        - Use "similarity" with "jaccard" method for similar-length responses
        - Use "normalized_equal" for exact matching after normalization
    """
    if method == "normalized_equal":
        match = normalized_equal(expected, actual, strict=kwargs.get("strict", False))
        return match, 1.0 if match else 0.0
    
    elif method == "regex":
        match = regex_search(expected, actual, case_sensitive=kwargs.get("case_sensitive", False))
        return match, 1.0 if match else 0.0
    
    elif method == "contains":
        match = contains_substring(expected, actual, case_sensitive=kwargs.get("case_sensitive", False))
        return match, 1.0 if match else 0.0
    
    elif method == "similarity":
        # Default to overlap coefficient for better handling of verbose responses
        sim_method = kwargs.get("similarity_method", "overlap")
        score = similarity_score(expected, actual, method=sim_method)
        match = score >= threshold
        return match, score
    
    else:
        raise ValueError(f"Unknown scoring method: {method}")
