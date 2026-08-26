"""$vectorSearch over the chunks collection.

search()'s primary signature intentionally takes just (query, ticker, limit)
so it's ergonomic to call directly (a REPL, eval.py, cli_search.py) without
wiring up Mongo each time — db/settings are keyword-only overrides, used by
eval.py (to reuse one connection across 15 calls instead of opening a new
one each time) and by tests (to inject a fake).
"""

from dataclasses import dataclass

from pymongo.database import Database

from fin_analyzer.config import Settings, get_settings
from fin_analyzer.db import get_db
from fin_analyzer.embeddings import embed_texts

RETRIEVAL_QUERY = "RETRIEVAL_QUERY"

# Atlas's own guidance: scan several times more candidates than you return,
# so the approximate nearest-neighbor search has enough to pick the true
# top `limit` from. 15x sits in the middle of Atlas's suggested 10-20x range.
CANDIDATE_MULTIPLIER = 15


@dataclass
class Chunk:
    ticker: str
    chunk_index: int
    text: str
    score: float


def search(
    query: str,
    ticker: str | None = None,
    limit: int = 5,
    *,
    db: Database | None = None,
    settings: Settings | None = None,
) -> list[Chunk]:
    settings = settings or get_settings()
    db = db if db is not None else get_db(settings)

    # The asymmetric half of task_type: a search query embeds as
    # RETRIEVAL_QUERY, never RETRIEVAL_DOCUMENT (that's for chunks, in embed.py).
    query_vector = embed_texts([query], RETRIEVAL_QUERY, settings)[0]

    pipeline = [{"$vectorSearch": _vector_search_stage(query_vector, ticker, limit, settings)}, _project_stage()]
    return [Chunk(**doc) for doc in db.chunks.aggregate(pipeline)]


def _vector_search_stage(query_vector: list[float], ticker: str | None, limit: int, settings: Settings) -> dict:
    stage = {
        "index": settings.vector_index_name,
        "path": "embedding",
        "queryVector": query_vector,
        "numCandidates": limit * CANDIDATE_MULTIPLIER,
        "limit": limit,
    }
    if ticker is not None:
        # Filtering requires "ticker" to be declared as a filter field in
        # the index definition (see vector_index.py's FILTER_FIELDS).
        stage["filter"] = {"ticker": {"$eq": ticker.upper()}}
    return stage


def _project_stage() -> dict:
    return {
        "$project": {
            "_id": 0,
            "ticker": 1,
            "chunk_index": 1,
            "text": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }
