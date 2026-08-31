"""The provider abstraction for text generation — one interface, providers
behind it. Embeddings are NOT part of this: they stay on Gemini
(embeddings.py, untouched, a separate quota that's been working fine
throughout). This is generation only — ask()'s answers and eval.py's
groundedness judge.

Built after being bitten twice by a hardcoded model name at the same call
site: gemini-2.5-flash turned out to be already discontinued for new API
keys, and its suggested replacement (gemini-3.6-flash) turned out to cap at
20 generate_content requests/day — a number that only appeared in the error
payload after hitting it, nowhere in public docs. From here on, adding a
provider should mean writing one class (below) plus one registry entry
(providers/registry.py) — never touching ask.py or eval.py.
"""

from abc import ABC, abstractmethod


class QuotaExhaustedError(Exception):
    """A provider's quota is exhausted in a way that retrying in-process
    won't fix — a long-window (e.g. daily) cap, as opposed to a brief rate
    limit worth a short backoff. Provider implementations are responsible
    for telling these apart (see groq_provider.py) and raising this only
    for the former, so callers (eval.py) can stop a loop cleanly instead of
    burning retries against a wall that won't move for hours.
    """


class Provider(ABC):
    @abstractmethod
    def generate(self, system_instruction: str, user_prompt: str, model: str, temperature: float) -> str:
        """One request/response call, returning the plain text answer."""

    @abstractmethod
    def generate_structured(
        self, system_instruction: str, user_prompt: str, response_schema: type, model: str, temperature: float
    ):
        """Like generate(), but returns a validated instance of
        response_schema (a Pydantic model class) instead of plain text."""

    @abstractmethod
    def list_model_names(self) -> set[str]:
        """Every model identifier currently available to this provider's
        API key — used at startup (providers/validate.py) to fail loudly on
        a misconfigured model name instead of discovering it mid-eval."""
