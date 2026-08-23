"""Extraction quality: junk (tables, nav, inline-XBRL) dropped, prose kept
and not fragmented by inline formatting tags. Uses a small embedded HTML
snippet shaped like real 10-K markup rather than a live filing, so this test
has no network dependency and runs fast.
"""

from fin_analyzer.extract import extract_text

SAMPLE_HTML = """
<html><body>
<table>
  <tr><td>Item 1</td><td>Business</td><td>5</td></tr>
  <tr><td>Item 1A</td><td>Risk Factors</td><td>12</td></tr>
</table>
<div>PART I</div>
<p>
  The Company designs, manufactures and markets <span>smartphones</span>,
  personal computers, tablets, wearables and <b>accessories</b>, and sells a
  variety of related services to customers worldwide.
</p>
<p>12</p>
<script>var trackingPixel = 1;</script>
<p>
  Net sales increased during the period, driven primarily by growth in
  Services revenue and continued demand for the Company's flagship products
  across all reportable geographic segments.
</p>
</body></html>
"""


def test_table_rows_are_dropped():
    text = extract_text(SAMPLE_HTML)
    assert "Risk Factors" not in text


def test_page_number_and_junk_lines_are_dropped():
    text = extract_text(SAMPLE_HTML)
    assert "PART I" not in text
    assert "trackingPixel" not in text


def test_prose_survives_and_is_not_fragmented_by_inline_tags():
    text = extract_text(SAMPLE_HTML)
    # If <span>/<b> weren't handled, this sentence would come out split into
    # "The Company designs, manufactures and markets" / "smartphones" / "..."
    assert "smartphones" in text
    assert "personal computers, tablets, wearables and accessories" in text
    assert "Net sales increased during the period" in text
