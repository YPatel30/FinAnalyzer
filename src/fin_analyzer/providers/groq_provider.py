"""Groq implementation of the Provider interface.

Groq specifically because Gemini's own generation models turned out to be
unusable here in practice — see base.py's docstring. Groq's free tier is
far more generous for the two roles this project needs (a generation model,
a separate judge model), and — importantly — its quotas are inspectable
live from the API's own response headers on every call, not just something
to trust from docs or a blog post.
"""

import time

import groq

from fin_analyzer.providers.base import Provider, QuotaExhaustedError

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2.0
BACKOFF_MULTIPLIER = 2.0

# If Groq's own response says to wait longer than this, it's a long-window
# (e.g. daily) limit, not a brief one — retrying in-process won't help, so
# this is treated as QuotaExhaustedError immediately rather than sleeping
# for however long the real reset actually takes.
MAX_WORTH_RETRYING_SECONDS = 120.0


class GroqProvider(Provider):
    def __init__(self, api_key: str):
        self._client = groq.Groq(api_key=api_key)

    def generate(self, system_instruction: str, user_prompt: str, model: str, temperature: float) -> str:
        response = self._call(system_instruction, user_prompt, model, temperature)
        return response.choices[0].message.content

    def generate_structured(
        self, system_instruction: str, user_prompt: str, response_schema: type, model: str, temperature: float
    ):
        response = self._call(
            system_instruction,
            user_prompt,
            model,
            temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": _strict_schema(response_schema.model_json_schema()),
                    "strict": True,
                },
            },
        )
        return response_schema.model_validate_json(response.choices[0].message.content)

    def list_model_names(self) -> set[str]:
        return {m.id for m in self._client.models.list().data}

    def _call(self, system_instruction: str, user_prompt: str, model: str, temperature: float, **kwargs):
        delay = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    **kwargs,
                )
            except groq.RateLimitError as exc:
                retry_after = _retry_after_seconds(exc)
                is_long_wait = retry_after is not None and retry_after > MAX_WORTH_RETRYING_SECONDS
                if is_long_wait or attempt == MAX_RETRIES:
                    raise QuotaExhaustedError(str(exc)) from exc
                wait = retry_after if retry_after is not None else delay
                print(f"  rate limited, retrying in {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                delay *= BACKOFF_MULTIPLIER


def _retry_after_seconds(exc: groq.RateLimitError) -> float | None:
    """Read the wait time straight from Groq's own response header, rather
    than guessing from the error message text — this is what actually tells
    a brief per-minute limit apart from a long-window one."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _strict_schema(schema: dict) -> dict:
    """Groq's strict JSON schema mode requires `additionalProperties: false`
    on every object node — Pydantic's model_json_schema() doesn't set that
    by default (discovered via a real 400 from the API, not from docs).
    Walks the whole schema (including $defs, for nested models) rather than
    just the top level, since a nested object missing it fails the same way.
    """
    if schema.get("type") == "object":
        schema = {**schema, "additionalProperties": False}
    for key in ("properties", "$defs"):
        if key in schema:
            schema = {**schema, key: {name: _strict_schema(value) for name, value in schema[key].items()}}
    return schema
