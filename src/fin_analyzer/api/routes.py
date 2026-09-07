"""All 6 routes. Deliberately thin — each handler validates nothing beyond
what its Pydantic request model already does, calls exactly one existing
business-logic function, and reshapes the result into a response schema.
No error handling here either: TickerNotFound/QuotaExhausted/IndexNotReady/
JobNotFound propagate up to the exception handlers registered in app.py,
which is where the exception -> status code mapping lives, and nowhere else.

Every handler below is plain `def`, never `async def` — do not "helpfully"
change this.

This project uses sync pymongo throughout (see CLAUDE.md — a deliberate
choice, not an oversight). An `async def` handler runs directly on the
single event loop thread; if it then makes a blocking call (any pymongo
call, or the Gemini/Groq HTTP calls inside search()/ask()), it stalls that
one thread every other concurrent request also depends on, and the whole
server serializes under load. Worse, this fails *silently* — a single
request in a manual test looks completely fine, because there's nothing
else on the event loop to be blocked by it. The problem only shows up under
concurrent load, exactly when you'd least want to discover it.

A plain `def` handler is different: FastAPI (via Starlette) runs it in a
threadpool automatically, off the event loop, so a blocking call blocks
only its own thread. That's the correct default for a sync database
driver here, not a shortcut — see CLAUDE.md's concurrency check for what
this actually buys under concurrent requests.
"""

from fastapi import APIRouter, BackgroundTasks, Depends
from pymongo.database import Database

from fin_analyzer import companies, health, jobs
from fin_analyzer.api.dependencies import get_db_dependency, get_settings_dependency
from fin_analyzer.api.schemas import (
    AskRequest,
    AskResponse,
    CompanyResponse,
    HealthResponse,
    IngestJobCreatedResponse,
    IngestJobStatusResponse,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceResponse,
    TickerResultResponse,
)
from fin_analyzer.ask import ask
from fin_analyzer.config import Settings
from fin_analyzer.search import search

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(db: Database = Depends(get_db_dependency), settings: Settings = Depends(get_settings_dependency)):
    status = health.check_health(db, settings)
    return HealthResponse(
        status="ok" if status.ok else "degraded",
        mongo_connected=status.mongo_connected,
        vector_index_status=status.vector_index_status,
    )


@router.get("/companies", response_model=list[CompanyResponse])
def get_companies(db: Database = Depends(get_db_dependency)):
    return [
        CompanyResponse(
            ticker=c.ticker, name=c.name, sector=c.sector, cik=c.cik, latest_filing_date=c.latest_filing_date
        )
        for c in companies.list_companies(db)
    ]


@router.post("/search", response_model=SearchResponse)
def post_search(
    body: SearchRequest,
    db: Database = Depends(get_db_dependency),
    settings: Settings = Depends(get_settings_dependency),
):
    chunks = search(body.query, ticker=body.ticker, limit=body.limit, db=db, settings=settings)
    results = [
        SearchResultItem(ticker=c.ticker, chunk_index=c.chunk_index, filing_date=c.filing_date, score=c.score, text=c.text)
        for c in chunks
    ]
    return SearchResponse(query=body.query, results=results)


@router.post("/ask", response_model=AskResponse)
def post_ask(
    body: AskRequest,
    db: Database = Depends(get_db_dependency),
    settings: Settings = Depends(get_settings_dependency),
):
    answer = ask(body.question, ticker=body.ticker, k=body.k, db=db, settings=settings)
    sources = [
        SourceResponse(
            ticker=s.ticker, filing_date=s.filing_date, chunk_index=s.chunk_index, score=s.score, text=s.text, cited=s.cited
        )
        for s in answer.sources
    ]
    return AskResponse(question=answer.question, answer=answer.answer, sources=sources, context_used=answer.context_used)


@router.post("/ingest", response_model=IngestJobCreatedResponse, status_code=202)
def post_ingest(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_db_dependency),
    settings: Settings = Depends(get_settings_dependency),
):
    job_id = jobs.create_job(db, body.tickers)
    background_tasks.add_task(jobs.run_ingest_job, job_id, body.tickers, db, settings)
    return IngestJobCreatedResponse(job_id=job_id, status="pending")


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
def get_ingest_job(job_id: str, db: Database = Depends(get_db_dependency)):
    job = jobs.get_job(db, job_id)
    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        tickers=job.tickers,
        results=[
            TickerResultResponse(ticker=r.ticker, status=r.status, chunk_count=r.chunk_count, error=r.error)
            for r in job.results
        ],
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
