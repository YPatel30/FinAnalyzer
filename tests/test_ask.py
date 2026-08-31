"""ask()'s core behavior, tested without touching Mongo, Groq, or Gemini —
search() and generate_answer() are both mocked. Citation *parsing* itself
is tested separately and directly in test_citations.py; here the concern is
ask() wiring citations to the right Source objects, and refusing correctly
when nothing was retrieved.
"""

from unittest.mock import MagicMock, patch

from fin_analyzer.ask import ask
from fin_analyzer.core.prompts import REFUSAL_MESSAGE
from fin_analyzer.search import Chunk

FAKE_CHUNKS = [
    Chunk(
        ticker="AAPL",
        chunk_index=3,
        text="Apple relies on outsourcing partners in Asia.",
        score=0.90,
        filing_date="2025-10-31",
    ),
    Chunk(
        ticker="AAPL",
        chunk_index=7,
        text="Apple faces geopolitical risk in its supply chain.",
        score=0.85,
        filing_date="2025-10-31",
    ),
]


@patch("fin_analyzer.ask.generate_answer")
@patch("fin_analyzer.ask.search")
def test_ask_returns_nonempty_sources_for_an_answerable_question(mock_search, mock_generate_answer):
    mock_search.return_value = FAKE_CHUNKS
    mock_generate_answer.return_value = "Apple relies on outsourcing partners [1] and faces geopolitical risk [2]."

    result = ask("how does Apple describe supply chain risk?", db=MagicMock(), settings=MagicMock())

    assert len(result.sources) == 2
    assert result.context_used == 2
    assert result.answer != REFUSAL_MESSAGE


@patch("fin_analyzer.ask.generate_answer")
@patch("fin_analyzer.ask.search")
def test_ask_marks_cited_sources_separately_from_merely_retrieved_ones(mock_search, mock_generate_answer):
    mock_search.return_value = FAKE_CHUNKS
    # Only cites [1] -- chunk #2 was retrieved but the model didn't use it.
    mock_generate_answer.return_value = "Apple relies on outsourcing partners in Asia [1]."

    result = ask("how does Apple describe supply chain risk?", db=MagicMock(), settings=MagicMock())

    assert result.sources[0].cited is True
    assert result.sources[1].cited is False


@patch("fin_analyzer.ask.generate_answer")
@patch("fin_analyzer.ask.search")
def test_ask_refuses_without_calling_the_model_when_nothing_is_retrieved(mock_search, mock_generate_answer):
    mock_search.return_value = []

    result = ask("what is the capital of France?", db=MagicMock(), settings=MagicMock())

    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []
    assert result.context_used == 0
    mock_generate_answer.assert_not_called()
