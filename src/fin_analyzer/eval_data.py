"""The 15 approved Phase 2 retrieval-eval questions.

Each question is grounded in a real chunk read out of the database (not
invented) — see the conversation this was drafted in for the source
chunks. `expected_phrase` is an exact substring of that chunk's `text`,
verified before this list was approved. Spread across AAPL/MSFT/GOOGL and
across Business, Risk Factors, and MD&A sections.

`ticker` here is metadata for reporting only — eval.py deliberately does
NOT filter search() by it. See eval.py's docstring for why.
"""

EVAL_QUESTIONS = [
    {
        "ticker": "AAPL",
        "section": "Business",
        "question": "What subscription services does Apple offer as part of its digital content business?",
        "expected_phrase": "Apple Fitness+",
    },
    {
        "ticker": "AAPL",
        "section": "Business",
        "question": "What are Apple's reportable geographic segments?",
        "expected_phrase": "Greater China, Japan and Rest of Asia Pacific",
    },
    {
        "ticker": "AAPL",
        "section": "Risk Factors",
        "question": "Which countries does Apple rely on most for manufacturing its products?",
        "expected_phrase": "China mainland, India, Japan, South Korea, Taiwan and Vietnam",
    },
    {
        "ticker": "AAPL",
        "section": "Risk Factors",
        "question": "How does Apple describe the importance of its company culture?",
        "expected_phrase": "distinctive and inclusive culture",
    },
    {
        "ticker": "AAPL",
        "section": "MD&A",
        "question": "What is Apple's estimated maximum one-day loss from foreign currency risk?",
        "expected_phrase": "$590 million",
    },
    {
        "ticker": "MSFT",
        "section": "Business",
        "question": "What economies of scale benefit Microsoft's cloud business?",
        "expected_phrase": "three economies of scale",
    },
    {
        "ticker": "MSFT",
        "section": "Business",
        "question": "What are Microsoft's three reportable segments?",
        "expected_phrase": "Productivity and Business Processes, Intelligent Cloud, and More Personal Computing",
    },
    {
        "ticker": "MSFT",
        "section": "Risk Factors",
        "question": "What costs create uncertainty for Microsoft's AI products?",
        "expected_phrase": "model training and inference costs",
    },
    {
        "ticker": "MSFT",
        "section": "Risk Factors",
        "question": "What pandemic-related risk does Microsoft disclose?",
        "expected_phrase": "regional epidemics or a global pandemic",
    },
    {
        "ticker": "MSFT",
        "section": "MD&A",
        "question": "How much did Microsoft's revenue grow in fiscal year 2026?",
        "expected_phrase": "Revenue increased $50.1 billion or 18%",
    },
    {
        "ticker": "GOOGL",
        "section": "Business",
        "question": "What is Google's corporate mission statement?",
        "expected_phrase": "organize the world’s information and make it universally accessible and useful",
    },
    {
        "ticker": "GOOGL",
        "section": "Business",
        "question": "What are Google's reportable segments?",
        "expected_phrase": "comprises two segments: Google Services and Google Cloud",
    },
    {
        "ticker": "GOOGL",
        "section": "Risk Factors",
        "question": "What areas does Alphabet invest in through its Other Bets segment?",
        "expected_phrase": "life sciences and transportation",
    },
    {
        "ticker": "GOOGL",
        "section": "Risk Factors",
        "question": "What financing risk does Alphabet describe?",
        "expected_phrase": "access future financing",
    },
    {
        "ticker": "GOOGL",
        "section": "MD&A",
        "question": "What was Alphabet's operating cash flow in 2025?",
        "expected_phrase": "$164.7 billion",
    },
]
