"""The oversized-chunk pre-check: gemini-embedding-001 caps input at 2048
tokens/text, and (per Google's docs) newer embedding models tend to
truncate silently rather than error on an over-limit input — so this must
be caught before any embedding call, not left to an exception handler.
"""

import pytest

from fin_analyzer.embed import OversizedChunksError, embed_missing_chunks


def _noop_index_step(db, settings):
    pass


def test_oversized_chunk_aborts_before_any_embedding_call(test_db, test_settings):
    # ~2500 tokens of real words (not "word " * N, which tiktoken would
    # collapse into far fewer tokens than characters suggest).
    huge_text = "The quick brown fox jumps over the lazy dog. " * 400

    test_db.chunks.insert_many(
        [
            {"ticker": "TEST", "filing_id": "fake-filing", "chunk_index": 0, "text": "a normal-sized chunk", "char_count": 21},
            {"ticker": "TEST", "filing_id": "fake-filing", "chunk_index": 1, "text": huge_text, "char_count": len(huge_text)},
        ]
    )

    calls = []

    def counting_fake_embed_fn(texts, task_type, settings, labels=None, client=None):
        calls.append(len(texts))
        return [[0.1] * 768 for _ in texts]

    with pytest.raises(OversizedChunksError):
        embed_missing_chunks(
            test_db, test_settings, embed_fn=counting_fake_embed_fn, ensure_index_fn=_noop_index_step, wait_ready_fn=_noop_index_step
        )

    # The whole run aborts before any embedding call — including for the
    # other, normal-sized chunk in the same batch.
    assert calls == []
    assert test_db.chunks.count_documents({"ticker": "TEST", "embedding": {"$exists": True}}) == 0
