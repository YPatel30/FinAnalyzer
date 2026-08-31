"""Maps a provider name (from Settings) to a concrete Provider instance.

Adding a new provider from here on: write one class implementing Provider,
add one line to _PROVIDERS. Nothing in ask.py, eval.py, or generation.py
should ever need to change for that.
"""

from fin_analyzer.config import Settings
from fin_analyzer.providers.base import Provider
from fin_analyzer.providers.groq_provider import GroqProvider

_PROVIDERS = {
    "groq": lambda settings: GroqProvider(api_key=settings.groq_api_key),
}


def get_provider(name: str, settings: Settings) -> Provider:
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider {name!r}. Available: {sorted(_PROVIDERS)}") from None
    return factory(settings)
