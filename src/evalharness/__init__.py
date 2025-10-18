"""
LLM Prompt Reliability & Invariance Testing Harness

A lightweight framework for evaluating LLM consistency under prompt transformations.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .scoring import normalize, normalized_equal, regex_search, similarity_score

__all__ = [
    "normalize",
    "normalized_equal", 
    "regex_search",
    "similarity_score",
]
