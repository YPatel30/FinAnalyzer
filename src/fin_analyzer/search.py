"""$vectorSearch over the chunks collection, joined with filings for
filing_date.

search()'s primary signature intentionally takes just (query, ticker, limit)
so it's ergonomic to call directly (a REPL, eval.py, cli_search.py) without
wiring up Mongo each time — db/settings are keyword-only overrides, used by
eval.py (to reuse one connection across many calls) and by tests (to inject
a fake).
"""

from dataclasses import dataclass

from pymongo.database import Database
from pymongo.errors import OperationFailure

from fin_analyzer.companies import ticker_exists
from fin_analyzer.config import Settings, get_settings
from fin_analyzer.core.exceptions import IndexNotReady, TickerNotFound
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
    filing_date: str


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

    # A ticker filter that doesn't match any ingested company would
    # otherwise just silently return zero results, indistinguishable from
    # "valid ticker, nothing relevant" — checked here, in the business
    # logic, not just at an API boundary, so the CLI and any future caller
    # (Phase 5's MCP tools) get the same clean failure instead of quietly
    # searching nothing. ticker=None ("search everything") is unaffected.
    if ticker is not None and not ticker_exists(db, ticker):
        raise TickerNotFound(ticker)

    # The asymmetric half of task_type: a search query embeds as
    # RETRIEVAL_QUERY, never RETRIEVAL_DOCUMENT (that's for chunks, in embed.py).
    query_vector = embed_texts([query], RETRIEVAL_QUERY, settings)[0]

    pipeline = [
        {"$vectorSearch": _vector_search_stage(query_vector, ticker, limit, settings)},
        # $meta: "vectorSearchScore" is only reliably readable in the stage
        # immediately after $vectorSearch — pulled into a plain field here,
        # before the $lookup below, rather than risking it not surviving
        # the join.
        _extract_score_stage(),
        _join_filing_date_stage(),
        _final_shape_stage(),
    ]
    try:
        results = list(db.chunks.aggregate(pipeline))
    except OperationFailure as exc:
        # Missing or still-building index -- surfaced as a clean, named
        # exception rather than letting a raw pymongo error leak out of the
        # business-logic layer (the API maps this to 503).
        raise IndexNotReady(f"Vector index {settings.vector_index_name!r} is not ready: {exc}") from exc

    return [Chunk(**doc) for doc in results]


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


def _extract_score_stage() -> dict:
    return {
        "$project": {
            "_id": 0,
            "ticker": 1,
            "chunk_index": 1,
            "text": 1,
            "filing_id": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
    }


def _join_filing_date_stage() -> dict:
    # chunks only store filing_id (a reference) — see search.py's docstring
    # and CLAUDE.md for why this is a $lookup rather than a denormalized
    # filing_date field on each chunk.
    return {
        "$lookup": {
            "from": "filings",
            "localField": "filing_id",
            "foreignField": "_id",
            "as": "filing",
        }
    }


def _final_shape_stage() -> dict:
    return {
        "$project": {
            "ticker": 1,
            "chunk_index": 1,
            "text": 1,
            "score": 1,
            # $lookup always produces an array (one match here, since
            # filing_id -> filings._id is one-to-one); pull out the single
            # filing_date instead of returning a one-element array of docs.
            "filing_date": {"$arrayElemAt": ["$filing.filing_date", 0]},
        }
    }
