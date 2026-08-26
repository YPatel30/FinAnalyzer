"""All Gemini embedding calls: batching, retry-on-429, re-normalization, and
a dimension guard. This is the one place the Gemini API gets called from —
same "single chokepoint" pattern as edgar_client.cached_get() in Phase 1.

Three things this module exists specifically to get right (getting any of
them wrong produces a system that half-works and is hard to debug):

1. task_type is asymmetric — chunks embed as RETRIEVAL_DOCUMENT, search
   queries embed as RETRIEVAL_QUERY. Callers pass task_type explicitly;
   nothing here defaults it, so a mix-up can't hide.
2. gemini-embedding-001 is natively 3072-dim and only pre-normalizes at that
   full size. Truncating to output_dimensionality=768 (via Matryoshka
   representation learning) leaves the vector's L2 norm != 1, which silently
   makes cosine similarity wrong unless normalized by hand — see _normalize().
3. Every vector's length is asserted before it's allowed anywhere near Mongo.
"""

import math
import time

from google import genai
from google.genai import errors

from fin_analyzer.config import Settings

MODEL_NAME = "gemini-embedding-001"

# Documented practical cap on texts per embed_content call for this model on
# the public Gemini Developer API (not Vertex) — batching beyond this fails.
BATCH_SIZE = 100

# The free tier's embedding rate limit is per-minute and tight enough to trip
# on a handful of consecutive batches (observed firsthand: a burst of ~4
# batch calls exhausted it). Backoff is sized to plausibly cross a 60s
# window: 5, 10, 20, 40s between the 5 attempts (~75s worst case).
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5.0
BACKOFF_MULTIPLIER = 2.0


def get_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def embed_texts(
    texts: list[str],
    task_type: str,
    settings: Settings,
    labels: list[str] | None = None,
    client: genai.Client | None = None,
) -> list[list[float] | None]:
    """Embed `texts`, returning one normalized `settings.embedding_dimensions`
    -length vector per input, in the same order. An item that fails even
    after the individual-fallback (see _embed_batch_with_fallback) comes
    back as None at that position rather than aborting the whole call —
    callers decide what to do with a None (embed.py skips and logs it).

    `labels` are purely for error messages (e.g. "AAPL chunk #12") — this
    module has no idea what a "chunk" is, it just embeds strings. `client`
    is injectable so tests can supply a fake and make zero real API calls.
    """
    if client is None:
        client = get_client(settings)
    if labels is None:
        labels = [f"text[{i}]" for i in range(len(texts))]

    vectors: list[list[float] | None] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        batch_labels = labels[start : start + BATCH_SIZE]
        vectors.extend(_embed_batch_with_fallback(client, batch, batch_labels, task_type, settings))
    return vectors


def _embed_batch_with_fallback(
    client: genai.Client, batch: list[str], batch_labels: list[str], task_type: str, settings: Settings
) -> list[list[float] | None]:
    """Try the whole batch in one call. If that fails because of one bad
    item (a content/format problem), fall back to embedding each item alone
    so the other ~99 good ones aren't lost too.

    A 429 (rate limit) is deliberately *not* handled this way: it's an
    account-level condition, not one chunk's fault, and after
    _embed_one_call's own retries are exhausted, fanning out into up to 100
    more individual calls would only hit the same limit harder — that's
    exactly what turned a transient rate limit into a much worse one the
    first time this ran for real. Let it propagate and stop the run instead;
    a rerun resumes cleanly since embedding is idempotent per-chunk.
    """
    try:
        return _embed_one_call(client, batch, task_type, settings)
    except Exception as exc:
        if isinstance(exc, errors.APIError) and exc.code == 429:
            raise
        print(f"  batch embed call failed ({exc}); falling back to {len(batch)} individual call(s)")
        results: list[list[float] | None] = []
        for text, label in zip(batch, batch_labels):
            try:
                results.extend(_embed_one_call(client, [text], task_type, settings))
            except Exception as item_exc:
                print(f"  SKIPPING {label}: {item_exc}")
                results.append(None)
        return results


def _embed_one_call(
    client: genai.Client, texts: list[str], task_type: str, settings: Settings
) -> list[list[float]]:
    """A single embed_content call (1..BATCH_SIZE texts), with retry +
    exponential backoff on 429 (rate limit) responses specifically — any
    other error is raised immediately since retrying won't fix it."""
    delay = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts,
                config={"task_type": task_type, "output_dimensionality": settings.embedding_dimensions},
            )
            vectors = [_normalize(embedding.values) for embedding in result.embeddings]
            for vector in vectors:
                _assert_dimensions(vector, settings.embedding_dimensions)
            return vectors
        except errors.APIError as exc:
            is_rate_limited = exc.code == 429
            if not is_rate_limited or attempt == MAX_RETRIES:
                raise
            print(f"    rate limited (429), retrying in {delay:.0f}s (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _assert_dimensions(vector: list[float], expected: int) -> None:
    if len(vector) != expected:
        raise ValueError(f"expected a {expected}-dim embedding, got {len(vector)}")
