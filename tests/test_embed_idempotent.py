"""embed_missing_chunks()'s idempotency, against real finanalyzer_test —
but with a fake embedder (no Gemini calls, no quota spent) and no-op index
functions (no second real Atlas search index spent on tests; M0 caps out
at 3 total, see CLAUDE.md).
"""

from fin_analyzer.embed import embed_missing_chunks

FAKE_VECTOR = [0.1] * 768


def _noop_index_step(db, settings):
    pass


def test_embed_missing_chunks_only_embeds_pending_and_is_idempotent(test_db, test_settings):
    test_db.chunks.insert_many(
        [
            {"ticker": "TEST", "filing_id": "fake-filing", "chunk_index": i, "text": f"chunk {i}", "char_count": 7}
            for i in range(3)
        ]
    )

    calls = []

    def counting_fake_embed_fn(texts, task_type, settings, labels=None, client=None):
        calls.append(len(texts))
        return [FAKE_VECTOR for _ in texts]

    first_count = embed_missing_chunks(
        test_db, test_settings, embed_fn=counting_fake_embed_fn, ensure_index_fn=_noop_index_step, wait_ready_fn=_noop_index_step
    )
    second_count = embed_missing_chunks(
        test_db, test_settings, embed_fn=counting_fake_embed_fn, ensure_index_fn=_noop_index_step, wait_ready_fn=_noop_index_step
    )

    assert first_count == 3
    assert second_count == 0
    # The fake embedder was called once (for the first run's 3 pending
    # chunks) and never again — that's the idempotency guarantee: a chunk
    # that already has a vector is never re-sent to the API on a rerun.
    assert calls == [3]

    stored = list(test_db.chunks.find({"ticker": "TEST"}))
    assert all(doc["embedding"] == FAKE_VECTOR for doc in stored)
