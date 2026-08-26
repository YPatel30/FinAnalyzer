"""search()'s query-building and result-mapping, tested without touching
Atlas or Gemini — db.chunks.aggregate() and the query embedding call are
both faked/mocked here.

This deliberately does NOT verify that Atlas's own $vectorSearch returns
results in descending-score order — that's Atlas's documented contract,
not something worth spending a second real Atlas search index (M0 caps out
at 3 total, see CLAUDE.md) to re-prove in a test. What's under test is that
our code preserves whatever order comes back and builds correct Chunk
objects from it — and that the query embeds with the correct (RETRIEVAL_QUERY)
task_type, not the chunk-side one.
"""

from unittest.mock import MagicMock, patch

from fin_analyzer.search import Chunk, search

# Shaped exactly like what our $project stage emits — already in the
# descending-score order $vectorSearch guarantees.
FAKE_RESULTS = [
    {"ticker": "AAPL", "chunk_index": 3, "text": "high relevance chunk", "score": 0.91},
    {"ticker": "MSFT", "chunk_index": 7, "text": "medium relevance chunk", "score": 0.85},
    {"ticker": "GOOGL", "chunk_index": 1, "text": "lower relevance chunk", "score": 0.79},
]


@patch("fin_analyzer.search.embed_texts")
def test_search_returns_results_in_descending_score_order(mock_embed_texts):
    mock_embed_texts.return_value = [[0.1] * 768]
    fake_db = MagicMock()
    fake_db.chunks.aggregate.return_value = FAKE_RESULTS
    fake_settings = MagicMock(vector_index_name="chunks_vector_index")

    results = search("any query", db=fake_db, settings=fake_settings)

    scores = [chunk.score for chunk in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0] == Chunk(ticker="AAPL", chunk_index=3, text="high relevance chunk", score=0.91)


@patch("fin_analyzer.search.embed_texts")
def test_search_embeds_the_query_as_retrieval_query(mock_embed_texts):
    mock_embed_texts.return_value = [[0.1] * 768]
    fake_db = MagicMock()
    fake_db.chunks.aggregate.return_value = []
    fake_settings = MagicMock(vector_index_name="chunks_vector_index")

    search("any query", db=fake_db, settings=fake_settings)

    call_args = mock_embed_texts.call_args.args
    texts, task_type = call_args[0], call_args[1]
    assert texts == ["any query"]
    assert task_type == "RETRIEVAL_QUERY"  # never RETRIEVAL_DOCUMENT — that's the chunk-side task_type
