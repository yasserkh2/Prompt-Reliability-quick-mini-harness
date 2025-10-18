"""
Unit tests for scoring utilities.
"""

import pytest
from evalharness.scoring import (
    normalize,
    normalized_equal,
    regex_search,
    contains_substring,
    similarity_score,
    score_match
)


class TestNormalize:
    """Tests for the normalize function."""
    
    def test_lowercase(self):
        assert normalize("HELLO") == "hello"
        assert normalize("HeLLo") == "hello"
    
    def test_whitespace_collapse(self):
        assert normalize("hello    world") == "hello world"
        assert normalize("  hello  world  ") == "hello world"
        assert normalize("hello\n\tworld") == "hello world"
    
    def test_punctuation_strip(self):
        assert normalize("hello.") == "hello"
        assert normalize("hello!") == "hello"
        assert normalize("hello?") == "hello"
        assert normalize("hello,") == "hello"
    
    def test_unicode_normalization(self):
        # NFKC normalization
        assert normalize("café") == "café"
    
    def test_combined(self):
        assert normalize("  HELLO   WORLD!  ") == "hello world"


class TestNormalizedEqual:
    """Tests for normalized_equal function."""
    
    def test_exact_match(self):
        assert normalized_equal("hello", "hello") is True
    
    def test_case_insensitive(self):
        assert normalized_equal("Hello", "hello") is True
        assert normalized_equal("HELLO", "hello") is True
    
    def test_whitespace_insensitive(self):
        assert normalized_equal("hello world", "hello  world") is True
        assert normalized_equal("  hello  ", "hello") is True
    
    def test_punctuation(self):
        assert normalized_equal("hello!", "hello") is True
        assert normalized_equal("hello?", "hello.") is True
    
    def test_not_equal(self):
        assert normalized_equal("hello", "world") is False
        assert normalized_equal("hello world", "world hello") is False
    
    def test_strict_mode(self):
        # Strict mode only normalizes whitespace
        assert normalized_equal("Hello", "hello", strict=True) is False
        assert normalized_equal("hello  world", "hello world", strict=True) is True


class TestRegexSearch:
    """Tests for regex_search function."""
    
    def test_simple_match(self):
        assert regex_search("world", "hello world") is True
        assert regex_search("foo", "hello world") is False
    
    def test_case_insensitive_default(self):
        assert regex_search("WORLD", "hello world") is True
        assert regex_search("World", "hello world") is True
    
    def test_case_sensitive(self):
        assert regex_search("WORLD", "hello world", case_sensitive=True) is False
        assert regex_search("world", "hello world", case_sensitive=True) is True
    
    def test_regex_patterns(self):
        assert regex_search(r"\d+", "hello 123 world") is True
        assert regex_search(r"^hello", "hello world") is True
        assert regex_search(r"world$", "hello world") is True
        assert regex_search(r"h.llo", "hello") is True
    
    def test_invalid_regex(self):
        with pytest.raises(ValueError):
            regex_search("[invalid", "text")


class TestContainsSubstring:
    """Tests for contains_substring function."""
    
    def test_simple_contains(self):
        assert contains_substring("world", "hello world") is True
        assert contains_substring("foo", "hello world") is False
    
    def test_case_insensitive_default(self):
        assert contains_substring("WORLD", "hello world") is True
        assert contains_substring("World", "hello world") is True
    
    def test_case_sensitive(self):
        assert contains_substring("WORLD", "hello world", case_sensitive=True) is False
        assert contains_substring("world", "hello world", case_sensitive=True) is True


class TestSimilarityScore:
    """Tests for similarity_score function."""
    
    def test_exact_match(self):
        assert similarity_score("hello", "hello", method="jaccard") == 1.0
        assert similarity_score("hello world", "hello world", method="jaccard") == 1.0
    
    def test_no_match(self):
        assert similarity_score("hello", "world", method="jaccard") == 0.0
    
    def test_partial_match(self):
        # "hello world" vs "hello there" - shares "hello"
        score = similarity_score("hello world", "hello there", method="jaccard")
        assert 0.0 < score < 1.0
    
    def test_jaccard_similarity(self):
        # "a b c" vs "b c d" -> intersection {b, c} = 2, union {a,b,c,d} = 4
        score = similarity_score("a b c", "b c d", method="jaccard")
        assert score == 0.5
    
    def test_overlap_coefficient(self):
        # "a b" vs "a b c d" -> intersection {a, b} = 2, min = 2
        score = similarity_score("a b", "a b c d", method="overlap")
        assert score == 1.0
    
    def test_levenshtein_similarity(self):
        score = similarity_score("kitten", "sitting", method="levenshtein")
        assert 0.0 < score < 1.0
        
        score = similarity_score("hello", "hello", method="levenshtein")
        assert score == 1.0
    
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            similarity_score("hello", "world", method="invalid")


class TestScoreMatch:
    """Tests for score_match function."""
    
    def test_normalized_equal_method(self):
        match, score = score_match("hello", "HELLO", method="normalized_equal")
        assert match is True
        assert score == 1.0
        
        match, score = score_match("hello", "world", method="normalized_equal")
        assert match is False
        assert score == 0.0
    
    def test_regex_method(self):
        match, score = score_match(r"\d+", "hello 123", method="regex")
        assert match is True
        assert score == 1.0
        
        match, score = score_match(r"\d+", "hello", method="regex")
        assert match is False
        assert score == 0.0
    
    def test_contains_method(self):
        match, score = score_match("world", "hello world", method="contains")
        assert match is True
        assert score == 1.0
    
    def test_similarity_method(self):
        match, score = score_match(
            "hello world",
            "hello there",
            method="similarity",
            threshold=0.3
        )
        assert match is True
        assert 0.0 < score < 1.0
        
        match, score = score_match(
            "hello",
            "world",
            method="similarity",
            threshold=0.5
        )
        assert match is False
        assert score == 0.0
    
    def test_invalid_method(self):
        with pytest.raises(ValueError):
            score_match("hello", "world", method="invalid")
