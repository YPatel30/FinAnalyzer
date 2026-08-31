"""core.citations pure-function tests — no mocking needed, no network."""

from fin_analyzer.core.citations import parse_cited_numbers


def test_parses_adjacent_bracket_citations():
    assert parse_cited_numbers("Apple attributes this to shortages [2][4].") == {2, 4}


def test_parses_a_single_citation():
    assert parse_cited_numbers("Revenue grew [1].") == {1}


def test_no_citations_returns_an_empty_set():
    assert parse_cited_numbers("No citations here.") == set()


def test_repeated_citations_are_deduplicated():
    assert parse_cited_numbers("[1] again, still [1], and also [2].") == {1, 2}


def test_fullwidth_brackets_are_also_recognized():
    # Observed in practice from a real model response — it clearly meant to
    # cite source 1, just used fullwidth brackets instead of ASCII ones.
    assert parse_cited_numbers("...essential to creating this environment【1】.") == {1}
