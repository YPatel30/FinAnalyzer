"""App configuration, loaded from environment variables / a .env file.

Using pydantic-settings instead of plain os.getenv() calls so every setting
has a type, a default (where one makes sense), and a single place it's
declared. Nothing here should be imported by tests that want to override
values — pass a Settings(...) instance directly instead (see tests/conftest.py).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Mongo Atlas connection string, e.g. mongodb+srv://user:pass@cluster.mongodb.net
    mongodb_uri: str
    mongodb_db: str = "finanalyzer"

    # SEC requires "Name email@example.com" on every request or it returns 403.
    # This is *your* identity, not the app's — SEC uses it to contact you if
    # your scraping causes problems.
    sec_user_agent: str

    # Chunking targets, in tokens (see chunk.py).
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 150

    # Where downloaded SEC responses are cached so reruns don't re-hit the network.
    cache_dir: Path = Path(".cache")

    # Sleep after every real (non-cached) SEC request. SEC's limit is 10 req/sec;
    # 0.2s keeps us at 5 req/sec, well under it.
    request_sleep_seconds: float = 0.2

    # Gemini embeddings (Phase 2). gemini-embedding-001 is natively 3072-dim;
    # we request a truncated 768-dim vector to keep index size reasonable on
    # M0 — see embeddings.py for the manual re-normalization this requires.
    gemini_api_key: str
    embedding_dimensions: int = 768
    vector_index_name: str = "chunks_vector_index"

    # Generation (Phase 3) — on Groq, not Gemini. Provider AND model are
    # both config, never hardcoded at a call site (see providers/) — after
    # gemini-2.5-flash got discontinued for new keys mid-project, and its
    # replacement turned out to cap at 20 requests/day undocumented, a
    # provider swap needs to be a settings change, not a refactor.
    # Answer generation and the groundedness judge are deliberately
    # different models (not just different roles of the same model) — a
    # judge with no stake in having produced the answer it's checking is
    # better methodology, not just quota isolation.
    groq_api_key: str
    generation_provider: str = "groq"
    generation_model: str = "openai/gpt-oss-120b"
    judge_provider: str = "groq"
    judge_model: str = "openai/gpt-oss-20b"


def get_settings() -> Settings:
    """Small indirection so callers don't construct Settings() directly everywhere."""
    return Settings()
