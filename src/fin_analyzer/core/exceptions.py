"""Plain Python exception types for conditions the API (Phase 4) and,
later, the MCP server (Phase 5) both need to react to. Deliberately no
FastAPI import here or anywhere else in core/ — mapping these to HTTP
status codes is api/'s job, and only its job. An MCP server maps the same
exceptions to its own error convention without core/ knowing either exists.
"""


class TickerNotFound(Exception):
    """Raised when a ticker was explicitly supplied but isn't among the
    ingested companies. Raised from search()/ask() themselves (not just at
    the API boundary) — "this company isn't in the corpus" is a domain
    fact, true for the CLI and any future caller alike, not an HTTP
    concern. Does not apply to ticker=None, which means "search everything"
    and stays valid.
    """

    def __init__(self, ticker: str):
        super().__init__(f"Ticker {ticker!r} has not been ingested.")
        self.ticker = ticker


class QuotaExhausted(Exception):
    """A provider's quota is exhausted in a way that retrying in-process
    won't fix — a long-window (e.g. daily) cap, as opposed to a brief rate
    limit worth a short backoff. `retry_after_seconds` carries the real
    wait time when the provider's own response told us one, so the API
    layer can set a Retry-After header instead of the client guessing.
    """

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class IndexNotReady(Exception):
    """The Atlas Vector Search index isn't queryable right now — missing,
    still building, or the driver call itself failed. Raised by search()
    when the underlying $vectorSearch call fails, rather than letting a raw
    pymongo OperationFailure leak out of the business-logic layer.
    """


class JobNotFound(Exception):
    """No ingest job with this id — same pattern as TickerNotFound: raised
    from jobs.py itself, not left as a None-check sitting in a route
    handler, for the same reason (a future MCP tool needs this too)."""

    def __init__(self, job_id: str):
        super().__init__(f"No ingest job with id {job_id!r}.")
        self.job_id = job_id
