"""Role-aware entry points for text generation — 'answer' and 'judge' are
configured independently (provider + model each, see config.py), so ask.py
and eval.py call the role they need without knowing settings' field names.
All actual provider mechanics (retries, quota detection, the API call
itself) live in providers/ — this module is just the two names callers use.

temperature=0 always: this is a learning/eval project, not a product —
reproducible output while iterating on the prompt matters more than
response variety.
"""

from fin_analyzer.config import Settings
from fin_analyzer.core.models import GroundednessVerdict
from fin_analyzer.providers.registry import get_provider


def generate_answer(system_instruction: str, user_prompt: str, settings: Settings) -> str:
    provider = get_provider(settings.generation_provider, settings)
    return provider.generate(system_instruction, user_prompt, settings.generation_model, temperature=0)


def judge_groundedness(system_instruction: str, user_prompt: str, settings: Settings) -> GroundednessVerdict:
    provider = get_provider(settings.judge_provider, settings)
    return provider.generate_structured(
        system_instruction, user_prompt, GroundednessVerdict, settings.judge_model, temperature=0
    )
