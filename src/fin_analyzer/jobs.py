"""The /ingest job state machine — a `jobs` collection in Mongo, not an
in-memory dict.

Why the dict version breaks, concretely:

1. Multiple worker processes. The moment this API runs with more than one
   worker (`uvicorn --workers 4`, or several instances behind a load
   balancer — the first thing anyone does to handle real concurrent load),
   each worker has its own separate memory. POST /ingest starting a job on
   worker A writes to worker A's dict; GET /ingest/{job_id} landing on
   worker B (ordinary round-robin routing) finds nothing, and the client
   gets a false 404 for a job that's actually running fine. This isn't an
   edge case — it's the default failure mode the instant you scale past
   one process.
2. Process restarts. A crash or redeploy while a job is in flight wipes
   the dict entirely. The client that started the job has no way to tell
   "it finished successfully before the restart" from "it never ran" —
   both look like a 404. Mongo survives the API process restarting.
3. Orphan detection. A job stuck in "running" with an old `started_at` and
   no progress means the process that was running it died mid-work — a
   real, detectable state. A dict can't represent this at all: it vanishes
   with the process, so there's no "stuck" state to observe, only silence.
   `started_at` is stored for exactly this reason; actually reaping stale
   jobs is future work, not built here.

Reused as-is: ingest_ticker() (Phase 1) does the real work per ticker,
completely unaware this module exists. This is only a state-tracking
wrapper around it.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pymongo.database import Database

from fin_analyzer.config import Settings
from fin_analyzer.core.exceptions import JobNotFound
from fin_analyzer.edgar_client import fetch_ticker_map
from fin_analyzer.ingest import ingest_ticker

PENDING = "pending"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"


@dataclass
class TickerResult:
    ticker: str
    status: str = PENDING
    chunk_count: int | None = None
    error: str | None = None


@dataclass
class Job:
    job_id: str
    status: str
    tickers: list[str]
    results: list[TickerResult]
    created_at: str
    updated_at: str
    error: str | None = field(default=None)


def create_job(db: Database, tickers: list[str]) -> str:
    """Inserts a pending job and returns its id. Called from the route
    handler before the response is sent — the job document exists (and is
    pollable) even before the background task has started running it.
    """
    job_id = uuid.uuid4().hex
    now = _now()
    db.jobs.insert_one(
        {
            "job_id": job_id,
            "status": PENDING,
            "tickers": tickers,
            "results": [{"ticker": t, "status": PENDING, "chunk_count": None, "error": None} for t in tickers],
            "error": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
        }
    )
    return job_id


def get_job(db: Database, job_id: str) -> Job:
    doc = db.jobs.find_one({"job_id": job_id})
    if doc is None:
        raise JobNotFound(job_id)
    return Job(
        job_id=doc["job_id"],
        status=doc["status"],
        tickers=doc["tickers"],
        results=[TickerResult(**r) for r in doc["results"]],
        error=doc.get("error"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


def run_ingest_job(job_id: str, tickers: list[str], db: Database, settings: Settings) -> None:
    """The BackgroundTasks target. Runs *after* the 202 response is already
    sent — nothing here can change what the client already received, only
    what a later GET /ingest/{job_id} sees.

    Continues past a single ticker's failure (mirrors ingest.py's CLI,
    which already does this) — one bad ticker in a batch shouldn't hide
    that the others succeeded. Overall job status is FAILED if any ticker
    failed, SUCCEEDED only if all did; per-ticker detail in `results`
    always shows exactly what happened to each one.
    """
    db.jobs.update_one({"job_id": job_id}, {"$set": {"status": RUNNING, "started_at": _now(), "updated_at": _now()}})

    try:
        ticker_map = fetch_ticker_map(settings)
    except Exception as exc:
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {"status": FAILED, "error": f"Could not fetch ticker map: {exc}", "updated_at": _now()}},
        )
        return

    any_failed = False
    for ticker in tickers:
        _set_ticker_status(db, job_id, ticker, RUNNING)
        try:
            result = ingest_ticker(ticker, ticker_map, settings, db)
            _set_ticker_result(db, job_id, ticker, SUCCEEDED, chunk_count=result["chunk_count"])
        except Exception as exc:
            any_failed = True
            _set_ticker_result(db, job_id, ticker, FAILED, error=str(exc))

    db.jobs.update_one(
        {"job_id": job_id}, {"$set": {"status": FAILED if any_failed else SUCCEEDED, "updated_at": _now()}}
    )


def _set_ticker_status(db: Database, job_id: str, ticker: str, status: str) -> None:
    db.jobs.update_one(
        {"job_id": job_id, "results.ticker": ticker},
        {"$set": {"results.$.status": status, "updated_at": _now()}},
    )


def _set_ticker_result(
    db: Database, job_id: str, ticker: str, status: str, *, chunk_count: int | None = None, error: str | None = None
) -> None:
    db.jobs.update_one(
        {"job_id": job_id, "results.ticker": ticker},
        {
            "$set": {
                "results.$.status": status,
                "results.$.chunk_count": chunk_count,
                "results.$.error": error,
                "updated_at": _now(),
            }
        },
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
