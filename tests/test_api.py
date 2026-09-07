"""FastAPI TestClient tests against finanalyzer_test — the HTTP layer only.

search()/ask()/the SEC calls inside jobs.run_ingest_job are mocked here
(same pattern as test_search.py/test_ask.py) so these tests spend zero
Gemini/Groq/SEC quota; what's under test is routing, status codes, and
request/response shaping, not retrieval or generation quality (that's
already covered where it belongs, in test_search.py/test_ask.py).

Plain TestClient(app), not `with TestClient(app) as client:` — entering
the context manager would run app.py's lifespan, which opens a real
connection to the *production* `finanalyzer` database via get_settings()
(lifespan doesn't go through dependency_overrides). Since every route's
db/settings dependency is overridden below to point at finanalyzer_test
instead, the routes never touch what lifespan would have set up anyway —
skipping the context manager just avoids an unnecessary production Mongo
connection during tests. BackgroundTasks still run synchronously within
each request/response cycle regardless.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fin_analyzer.api.app import app
from fin_analyzer.api.dependencies import get_db_dependency, get_settings_dependency
from fin_analyzer.core.models import Answer, Source
from fin_analyzer.search import Chunk


@pytest.fixture
def client(test_db, test_settings):
    app.dependency_overrides[get_db_dependency] = lambda: test_db
    app.dependency_overrides[get_settings_dependency] = lambda: test_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def _insert_company(test_db, ticker="AAPL"):
    test_db.companies.insert_one({"ticker": ticker, "name": "Apple Inc.", "sector": "Electronic Computers", "cik": "0000320193"})
    test_db.filings.insert_one({"ticker": ticker, "form_type": "10-K", "filing_date": "2025-10-31", "accession_no": "test-acc-1", "source_url": "https://example.com"})


# --- /health ---


def test_health_happy_path(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["mongo_connected"] is True
    assert body["vector_index_status"] in {"ready", "not_ready", "unknown"}


# --- /companies ---


def test_companies_happy_path(client, test_db):
    _insert_company(test_db, "AAPL")

    response = client.get("/companies")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["latest_filing_date"] == "2025-10-31"


# --- /search ---


@patch("fin_analyzer.api.routes.search")
def test_search_happy_path(mock_search, client):
    mock_search.return_value = [
        Chunk(ticker="AAPL", chunk_index=3, text="some risk text", score=0.9, filing_date="2025-10-31")
    ]

    response = client.post("/search", json={"query": "risk", "ticker": "AAPL", "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["ticker"] == "AAPL"
    assert body["results"][0]["text"] == "some risk text"
    # Nothing in the response leaks internal-only concerns.
    assert "_id" not in body["results"][0]
    assert "embedding" not in body["results"][0]


@patch("fin_analyzer.api.routes.search")
def test_search_unknown_ticker_returns_404(mock_search, client):
    from fin_analyzer.core.exceptions import TickerNotFound

    mock_search.side_effect = TickerNotFound("TSLA")

    response = client.post("/search", json={"query": "risk", "ticker": "TSLA"})

    assert response.status_code == 404
    assert "TSLA" in response.json()["detail"]


def test_search_malformed_body_returns_422(client):
    response = client.post("/search", json={"limit": "not-a-number"})
    assert response.status_code == 422


# --- /ask ---


@patch("fin_analyzer.api.routes.ask")
def test_ask_happy_path(mock_ask, client):
    mock_ask.return_value = Answer(
        question="how does Apple describe supply chain risk?",
        answer="Apple relies on outsourcing partners [1].",
        sources=[
            Source(ticker="AAPL", filing_date="2025-10-31", chunk_index=9, score=0.87, text="full chunk text", cited=True)
        ],
        context_used=1,
    )

    response = client.post("/ask", json={"question": "how does Apple describe supply chain risk?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Apple relies on outsourcing partners [1]."
    assert body["sources"][0]["cited"] is True
    assert body["context_used"] == 1


def test_ask_malformed_body_returns_422(client):
    response = client.post("/ask", json={})  # question is required
    assert response.status_code == 422


# --- /ingest ---


@patch("fin_analyzer.jobs.fetch_ticker_map")
@patch("fin_analyzer.jobs.ingest_ticker")
def test_ingest_returns_202_with_a_resolvable_job_id(mock_ingest_ticker, mock_fetch_ticker_map, client):
    mock_fetch_ticker_map.return_value = {}
    mock_ingest_ticker.return_value = {"ticker": "AAPL", "cik": "0000320193", "filing_id": "fake", "chunk_count": 12}

    create_response = client.post("/ingest", json={"tickers": ["AAPL"]})
    assert create_response.status_code == 202
    job_id = create_response.json()["job_id"]
    assert job_id

    # BackgroundTasks run synchronously within TestClient's request cycle,
    # so by the time control returns here the job has already progressed.
    status_response = client.get(f"/ingest/{job_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "succeeded"
    assert body["results"][0] == {"ticker": "AAPL", "status": "succeeded", "chunk_count": 12, "error": None}


def test_ingest_unknown_job_id_returns_404(client):
    response = client.get("/ingest/does-not-exist")
    assert response.status_code == 404
