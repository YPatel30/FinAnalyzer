"""Fail loudly at startup if a configured model isn't actually available to
this API key — rather than discovering that mid-eval, mid-quota-burn, which
is exactly how gemini-3.6-flash's undocumented daily cap surfaced the first
time. Called once at the top of cli_ask.py/cli_eval.py's main(), before any
retrieval or generation happens.
"""

from fin_analyzer.config import Settings
from fin_analyzer.providers.registry import get_provider


def validate_configured_models(settings: Settings) -> None:
    checked_providers: dict[str, set[str]] = {}

    for role, provider_name, model_name in [
        ("generation", settings.generation_provider, settings.generation_model),
        ("judge", settings.judge_provider, settings.judge_model),
    ]:
        if provider_name not in checked_providers:
            checked_providers[provider_name] = get_provider(provider_name, settings).list_model_names()
        available = checked_providers[provider_name]

        if model_name not in available:
            raise RuntimeError(
                f"Configured {role} model {model_name!r} (provider={provider_name!r}) is not "
                f"available to this API key. Available {provider_name} models: {sorted(available)}"
            )
