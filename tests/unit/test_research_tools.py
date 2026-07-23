"""Tests for the tool-augmented niche research tools (CL-2czc).

Covers: HTML→text, anchored excerpt, SEC filing excerpt (submissions →
newest 10-K → doc → excerpt, with fail-soft branches), the per-name
grounding block, and enrich() over a set of ideas. Injected transports —
no live SEC / yfinance / network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.events.research_tools import (
    ResearchTools,
    anchored_excerpt,
    html_to_text,
)


class _FakeUniverse:
    def __init__(self, ciks: dict[str, int]) -> None:
        self._ciks = ciks

    def get_cik(self, ticker: str) -> int | None:
        return self._ciks.get((ticker or "").upper())


_SUBMISSIONS = json.dumps(
    {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K", "10-Q"],
                "accessionNumber": [
                    "0000000000-26-000001",
                    "0001801368-26-000012",
                    "0001801368-26-000030",
                ],
                "primaryDocument": ["ev.htm", "mp-10k.htm", "mp-10q.htm"],
            }
        },
    }
)

_DOC_HTML = (
    "<html><head><style>.x{color:red}</style></head><body>"
    "<h1>Item&nbsp;1A. Risk Factors</h1>"
    "<p>We depend on a single customer, Acme Corp, for approximately 40% of "
    "our consolidated revenue, and the loss of this customer would materially "
    "harm our results. We source neodymium and other rare-earth feedstock from "
    "a single supplier, Beta Mining, and any disruption to that relationship "
    "could halt production. We also face intense competition from larger, "
    "better-capitalized producers and from substitute materials.</p>"
    "<script>ignored()</script></body></html>"
)


def _fake_sec(submissions: str = _SUBMISSIONS, doc: str = _DOC_HTML):
    def _get(url: str, headers: dict[str, str]) -> str:
        assert "User-Agent" in headers
        return submissions if "submissions" in url else doc

    return _get


# --------------------------------------------------------------------------- #
# html_to_text / anchored_excerpt
# --------------------------------------------------------------------------- #


def test_html_to_text_strips_and_unescapes():
    out = html_to_text(_DOC_HTML)
    assert "Acme Corp" in out and "Beta Mining" in out
    assert "<" not in out and ">" not in out
    assert "ignored()" not in out  # script content dropped
    assert "color:red" not in out  # style dropped


def test_anchored_excerpt_starts_at_anchor():
    text = "boilerplate cover page " * 20 + "RISK FACTORS here is the meat"
    exc = anchored_excerpt(text, 100)
    assert exc is not None and exc.lower().startswith("risk factors")


def test_anchored_excerpt_none_when_too_short():
    assert anchored_excerpt("tiny", 100) is None


def test_anchored_excerpt_falls_back_to_head():
    text = "x" * 500  # no anchor present
    exc = anchored_excerpt(text, 100)
    assert exc == "x" * 100


def test_anchored_excerpt_skips_table_of_contents():
    # First "risk factors" is a TOC line (page numbers); the real section is
    # later and reads like prose — the excerpt must land on the prose.
    toc = "Risk Factors 10 Item 1B Unresolved Staff Comments 27 Properties 28 "
    body = (
        "Risk Factors you should carefully consider the following risks before "
        "investing as we depend heavily on our largest customer and a single "
        "rare earth supplier for critical feedstock and operations."
    )
    text = toc + body
    exc = anchored_excerpt(text, 200)
    assert exc is not None
    assert "you should carefully consider" in exc  # landed on the real section


# --------------------------------------------------------------------------- #
# sec_excerpt
# --------------------------------------------------------------------------- #


def test_sec_excerpt_picks_10k_and_extracts():
    tools = ResearchTools(sec_http_get=_fake_sec(), profile_fn=lambda t: None)
    exc = tools.sec_excerpt(1801368)
    assert exc is not None
    assert exc.startswith("[10-K]")
    assert "Acme Corp" in exc and "Beta Mining" in exc


def test_sec_excerpt_falls_back_to_10q():
    subs = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q"],
                    "accessionNumber": ["a-1", "b-2"],
                    "primaryDocument": ["x.htm", "q.htm"],
                }
            }
        }
    )
    tools = ResearchTools(sec_http_get=_fake_sec(submissions=subs))
    assert tools.sec_excerpt(123)[:6] == "[10-Q]"


def test_sec_excerpt_none_when_no_periodic_filing():
    subs = json.dumps(
        {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "accessionNumber": ["a-1"],
                    "primaryDocument": ["x.htm"],
                }
            }
        }
    )
    tools = ResearchTools(sec_http_get=_fake_sec(submissions=subs))
    assert tools.sec_excerpt(123) is None


def test_sec_excerpt_fail_soft_on_bad_json():
    tools = ResearchTools(sec_http_get=lambda u, h: "not json")
    assert tools.sec_excerpt(123) is None


def test_sec_excerpt_fail_soft_on_fetch_error():
    def boom(url, headers):
        raise RuntimeError("sec down")

    assert ResearchTools(sec_http_get=boom).sec_excerpt(123) is None


# --------------------------------------------------------------------------- #
# ground_one / enrich
# --------------------------------------------------------------------------- #


def test_ground_one_combines_profile_and_filing():
    tools = ResearchTools(
        sec_http_get=_fake_sec(),
        profile_fn=lambda t: {
            "sector": "Materials",
            "industry": "Mining",
            "summary": "Rare-earth producer.",
        },
    )
    block = tools.ground_one("MP", "MP Materials", _FakeUniverse({"MP": 1801368}))
    assert block is not None
    assert "MP Materials (MP)" in block
    assert "Materials / Mining" in block
    assert "Rare-earth producer." in block
    assert "Acme Corp" in block  # from the SEC excerpt


def test_ground_one_none_when_no_data():
    # No profile, no CIK → nothing but the header → None.
    tools = ResearchTools(sec_http_get=_fake_sec(), profile_fn=lambda t: None)
    assert tools.ground_one("ZZZ", "Zed", _FakeUniverse({})) is None


def test_enrich_skips_tickerless_and_returns_blocks():
    tools = ResearchTools(
        sec_http_get=_fake_sec(),
        profile_fn=lambda t: {"sector": "Materials", "industry": "Mining", "summary": "x"},
    )
    ideas = [
        SimpleNamespace(ticker="MP", company_name="MP Materials"),
        SimpleNamespace(ticker="", company_name="No Ticker Co"),  # skipped
    ]
    blocks = tools.enrich(ideas, _FakeUniverse({"MP": 1801368}))
    assert len(blocks) == 1
    assert "MP Materials (MP)" in blocks[0]
