"""Normalization and dimension-guard tests — pure functions, no network
call and no Gemini API key needed."""

import math

import pytest

from fin_analyzer.embeddings import _assert_dimensions, _normalize


def test_normalize_produces_a_unit_length_vector():
    # A non-unit vector (magnitude 5), like a truncated gemini-embedding-001
    # output before it's been renormalized by hand.
    raw = [3.0, 4.0]
    normalized = _normalize(raw)
    norm = math.sqrt(sum(v * v for v in normalized))
    assert norm == pytest.approx(1.0)


def test_normalize_preserves_dimension_count():
    raw = [1.0] * 768
    assert len(_normalize(raw)) == 768


def test_assert_dimensions_accepts_matching_length():
    _assert_dimensions([0.0] * 768, expected=768)  # should not raise


def test_assert_dimensions_rejects_mismatched_length():
    with pytest.raises(ValueError):
        _assert_dimensions([0.0] * 100, expected=768)
