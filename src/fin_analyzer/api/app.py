"""The FastAPI app: lifespan (Mongo client setup/teardown), CORS, the
exception -> status code mapping, and per-request-ID logging. This module,
routes.py, dependencies.py, and schemas.py are the *only* places in this
codebase allowed to import fastapi — core/, providers/, and every existing
business-logic module (search.py, ask.py, jobs.py, ...) must stay free of
it, so Phase 5's MCP server can wrap the same logic without untangling
anything HTTP-specific first.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fin_analyzer.api.routes import router
from fin_analyzer.config import get_settings
from fin_analyzer.core.exceptions import IndexNotReady, JobNotFound, QuotaExhausted, TickerNotFound
from fin_analyzer.db import ensure_indexes, get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fin_analyzer.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One MongoClient for the whole process lifetime, not one per request.
    # db.get_client() opens a fresh connection every call, which is fine
    # for a short-lived CLI process but wrong for a long-running server —
    # lifespan (not the deprecated on_event hooks) is the documented place
    # to set up a resource once at startup and tear it down once at shutdown.
    settings = get_settings()
    client = get_client(settings)
    db = client[settings.mongodb_db]
    ensure_indexes(db)
    app.state.db = db
    app.state.settings = settings
    yield
    client.close()


app = FastAPI(title="FinAnalyzer API", lifespan=lifespan)

# Allow localhost only, any port — no specific frontend/port chosen yet.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """A per-request id, on both the response (X-Request-ID header, so a
    client can report it back) and every log line for that request
    (including the 500 handler below) — so a failure in the logs can be
    tied to one specific call, not just "something failed around 2:14pm".
    """
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %d (%.1fms) [%s]", request.method, request.url.path, response.status_code, duration_ms, request_id
    )
    return response


@app.exception_handler(TickerNotFound)
def handle_ticker_not_found(request: Request, exc: TickerNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(JobNotFound)
def handle_job_not_found(request: Request, exc: JobNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(QuotaExhausted)
def handle_quota_exhausted(request: Request, exc: QuotaExhausted) -> JSONResponse:
    headers = {}
    if exc.retry_after_seconds is not None:
        headers["Retry-After"] = str(int(exc.retry_after_seconds))
    return JSONResponse(status_code=429, content={"detail": str(exc)}, headers=headers)


@app.exception_handler(IndexNotReady)
def handle_index_not_ready(request: Request, exc: IndexNotReady) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    # Full traceback goes to the server log (tagged with the same request
    # id set above), never to the client — an unhandled exception can leak
    # internal details (file paths, query shapes, secrets in a stack frame)
    # that a stranger on the internet has no business seeing.
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error on %s %s [%s]", request.method, request.url.path, request_id)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(router)
