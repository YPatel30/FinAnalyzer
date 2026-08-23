"""Chunking tests use small token budgets (not the production 1000/150) so
the fixture text can stay short and the tests run fast — the sizing and
overlap behavior being tested doesn't depend on the specific numbers.
"""

import re

import tiktoken

from fin_analyzer.chunk import chunk_text

ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(ENCODER.encode(text))


def _make_text(sentence_count: int) -> str:
    # Each sentence carries a unique id so overlap between chunks is
    # verifiable, plus enough filler that a handful of sentences add up to
    # a real chunk budget in tokens.
    sentences = (
        f"Sentence{i} describes some filler content used to pad out the token count nicely."
        for i in range(sentence_count)
    )
    return " ".join(sentences)


def test_chunk_sizes_stay_within_budget():
    text = _make_text(sentence_count=300)
    chunks = chunk_text(text, chunk_size_tokens=200, chunk_overlap_tokens=50)

    assert len(chunks) > 1
    # The last chunk is whatever's left over and may be shorter — every
    # other chunk should respect the requested budget.
    for chunk in chunks[:-1]:
        assert _token_count(chunk) <= 200


def test_consecutive_chunks_overlap():
    text = _make_text(sentence_count=300)
    chunks = chunk_text(text, chunk_size_tokens=200, chunk_overlap_tokens=50)

    assert len(chunks) > 1
    first_ids = set(re.findall(r"Sentence(\d+)", chunks[0]))
    second_ids = set(re.findall(r"Sentence(\d+)", chunks[1]))
    shared = first_ids & second_ids
    assert shared, "consecutive chunks should share sentences from the overlap window"
