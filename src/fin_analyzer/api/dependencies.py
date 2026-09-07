"""FastAPI dependencies for Mongo access. The database handle comes from
`request.app.state` — set once in app.py's lifespan, not opened fresh per
request. Settings load once at import time (pydantic-settings itself does
the real .env read); wrapped as a dependency mainly so tests can override
it via `app.dependency_overrides`.
"""

from fastapi import Request
from pymongo.database import Database

from fin_analyzer.config import Settings, get_settings

_settings = get_settings()


def get_settings_dependency() -> Settings:
    return _settings


def get_db_dependency(request: Request) -> Database:
    return request.app.state.db
