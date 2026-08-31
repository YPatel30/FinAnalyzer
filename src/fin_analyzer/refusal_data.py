"""The 5 refusal-set questions: things the corpus provably cannot answer.
Spread across the 3 failure modes ask() needs to catch — see CLAUDE.md for
why these specific categories, and Phase 3's design conversation for why
these exact questions (not close paraphrases of them) live here rather than
inside core/prompts.py's system instruction: baking test questions into the
prompt would let the model pass by memorizing the test, not by actually
reasoning about missing context.
"""

REFUSAL_QUESTIONS = [
    {"category": "company not ingested", "question": "What are Tesla's main risk factors?"},
    {"category": "company not ingested", "question": "What does Amazon say about its logistics network?"},
    {"category": "forward-looking", "question": "Will Apple's margins improve next year?"},
    {"category": "forward-looking", "question": "What will Microsoft's revenue be in fiscal year 2030?"},
    {"category": "out of domain", "question": "What's the capital of France?"},
]
