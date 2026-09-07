"""Request/response models for the HTTP boundary — deliberately separate
from core/models.py's Answer/Source and search.py's Chunk, even though the
fields mostly line up. Two reasons: (1) Mongo's ObjectId isn't JSON-
serializable and ends up nowhere near these classes, rather than being
handled as a one-off surprise; (2) the internal models are free to grow
fields (an embedding vector, an internal _id) that must never appear in an
API response — keeping separate classes means that's true by construction,
not by remembering not to leak something.
"""

from pydantic import BaseModel


# --- /health ---
class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    mongo_connected: bool
    vector_index_status: str  # "ready" | "not_ready" | "unknown"


# --- /companies ---
class CompanyResponse(BaseModel):
    ticker: str
    name: str
    sector: str | None
    cik: str
    latest_filing_date: str | None


# --- /search ---
class SearchRequest(BaseModel):
    query: str
    ticker: str | None = None
    limit: int = 5


class SearchResultItem(BaseModel):
    ticker: str
    chunk_index: int
    filing_date: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


# --- /ask ---
class AskRequest(BaseModel):
    question: str
    ticker: str | None = None
    k: int = 5


class SourceResponse(BaseModel):
    ticker: str
    filing_date: str
    chunk_index: int
    score: float
    text: str
    cited: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]
    context_used: int


# --- /ingest ---
class IngestRequest(BaseModel):
    tickers: list[str]


class IngestJobCreatedResponse(BaseModel):
    job_id: str
    status: str  # "pending"


class TickerResultResponse(BaseModel):
    ticker: str
    status: str  # "pending" | "running" | "succeeded" | "failed"
    chunk_count: int | None
    error: str | None


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "succeeded" | "failed"
    tickers: list[str]
    results: list[TickerResultResponse]
    created_at: str
    updated_at: str
