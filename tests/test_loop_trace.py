"""Tests for the trace helpers in agent/loop.py.

These exist because of a real run. A search for a person returned twenty
results, six were opened, three of those refused the request, and the answer
said nothing was found. Every number in the trace looked healthy and none of
them said the useful thing, which was: the page most likely to hold the answer
refused us, fourteen results were never opened, and the name asked about
appeared nowhere in what could be read.

Each helper here exists to make one of those legible. They are pure functions,
so none of this needs a network, a key or a model - the heavy imports are
stubbed before agent.loop is imported.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

# Stub what agent.loop pulls in transitively. Same rule as the other test
# modules: no API keys, no network, no model download.
for _name in ("tavily", "trafilatura", "tiktoken", "sentence_transformers", "groq"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["tavily"].TavilyClient = object                    # type: ignore[attr-defined]
sys.modules["sentence_transformers"].SentenceTransformer = object  # type: ignore[attr-defined]
sys.modules["groq"].Groq = object                              # type: ignore[attr-defined]
if "google" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.genai")
    _gg.Client = object                                        # type: ignore[attr-defined]
    _gt = types.ModuleType("google.genai.types")
    _gg.types = _gt                                            # type: ignore[attr-defined]
    _g.genai = _gg                                             # type: ignore[attr-defined]
    sys.modules["google"], sys.modules["google.genai"] = _g, _gg
    sys.modules["google.genai.types"] = _gt

from agent import loop                                          # noqa: E402


@dataclass
class R:
    url: str
    title: str = ""
    snippet: str = ""
    score: float | None = None


@dataclass
class P:
    text: str
    error: str | None = None
    domain: str = "example.com"
    url: str = "https://example.com/a"
    title: str = ""


# ------------------------------------------------------------------- terms

def test_query_terms_drops_noise_words():
    # "who", "is" and "and" are not things a source can fail to cover.
    assert loop._query_terms("Who is Somanorani Ningthoujam and what is her role?") == [
        "somanorani", "ningthoujam", "her", "role",
    ]


def test_query_terms_are_unique_and_keep_order():
    assert loop._query_terms("beaver dams and beaver ponds") == ["beaver", "dams", "ponds"]


def test_term_coverage_separates_the_name_that_matched_from_the_one_that_did_not():
    # The exact case this was written for. Both pages are about a Ningthoujam;
    # neither is about Somanorani. A total result count cannot show that.
    pages = [
        P("Ningthoujam Babysana, a 14-year-old student, died in 2019."),
        P("Dr Ningthoujam Shovarani works in environmental biotechnology."),
    ]
    cov = {c["term"]: c for c in loop._term_coverage("somanorani ningthoujam", pages)}

    assert cov["ningthoujam"]["pages"] == 2
    assert cov["somanorani"]["pages"] == 0
    assert cov["somanorani"]["hits"] == 0
    assert cov["ningthoujam"]["of_pages"] == 2


def test_term_coverage_ignores_pages_that_could_not_be_read():
    # A refused or empty page is not evidence that a term is absent, so it must
    # not be counted in the denominator.
    pages = [P("somanorani appears here"), P("", error="403"), P("")]
    cov = loop._term_coverage("somanorani", pages)[0]
    assert cov["of_pages"] == 1
    assert cov["pages"] == 1


def test_term_coverage_is_empty_when_nothing_was_readable():
    assert loop._term_coverage("anything", [P("", error="403")])[0]["of_pages"] == 0


# ------------------------------------------------------------------ dedupe

def test_dedupe_records_which_query_found_each_result():
    a, b = R("https://x.com/1"), R("https://x.com/2")
    results, found_by, stats = loop._dedupe_results([
        ("biography", [a, b]),
        ("career", [a]),
    ])

    assert len(results) == 2
    assert found_by["https://x.com/1"] == ["biography", "career"]
    assert found_by["https://x.com/2"] == ["biography"]


def test_dedupe_reports_new_results_per_query_not_just_totals():
    # The second query returning 3 results that were all already seen is a
    # different event from it returning 3 new ones, and the old trace showed
    # both as the same number.
    a, b, c = R("https://x.com/1"), R("https://x.com/2"), R("https://x.com/3")
    _, _, stats = loop._dedupe_results([("q1", [a, b, c]), ("q2", [a, b, c])])

    assert stats[0] == {"query": "q1", "results": 3, "new": 3}
    assert stats[1] == {"query": "q2", "results": 3, "new": 0}


def test_dedupe_reports_a_query_that_found_nothing():
    _, _, stats = loop._dedupe_results([("q1", [R("https://x.com/1")]), ("q2", [])])
    assert stats[1] == {"query": "q2", "results": 0, "new": 0}


def test_dedupe_skips_results_with_no_url():
    results, _, _ = loop._dedupe_results([("q", [R(""), R("https://x.com/1")])])
    assert [r.url for r in results] == ["https://x.com/1"]


# -------------------------------------------------------------------- pick

def test_ranking_puts_unscored_results_last():
    ranked = loop._rank_results([R("u1"), R("u2", score=0.9), R("u3", score=0.4)])
    assert [r.url for r in ranked] == ["u2", "u3", "u1"]


def test_pick_records_what_the_cap_left_behind():
    # The failure this is for: 20 results, 6 opened, and a trace that mentioned
    # only the 6, so the run looked like the search had found nothing more.
    ranked = [R(f"https://d{i}.com/a", score=1 - i / 100) for i in range(10)]
    urls, skipped = loop._pick_urls_to_fetch(ranked, cap=3)

    assert len(urls) == 3
    assert len(skipped) == 7
    assert all(s["reason"] == "page cap reached" for s in skipped)
    assert skipped[0]["rank"] == 4          # rank is search position, not skip order


def test_pick_records_the_domain_rule_separately_from_the_cap():
    # Two different reasons a page was not opened. Collapsing them would make a
    # site that was deliberately limited look like one that ran out of budget.
    ranked = [
        R("https://a.com/1", score=0.9), R("https://a.com/2", score=0.8),
        R("https://a.com/3", score=0.7), R("https://b.com/1", score=0.6),
    ]
    urls, skipped = loop._pick_urls_to_fetch(ranked, cap=6)

    assert urls == ["https://a.com/1", "https://a.com/2", "https://b.com/1"]
    assert [s["reason"] for s in skipped] == ["2 already picked from this domain"]
    assert skipped[0]["rank"] == 3


def test_pick_keeps_rank_meaning_search_position():
    # A 403 from the top-ranked result and a 403 from the twentieth are not the
    # same event, and rank is the only thing that tells them apart.
    ranked = [R(f"https://d{i}.com/a", score=1 - i / 100) for i in range(5)]
    _, skipped = loop._pick_urls_to_fetch(ranked, cap=2)
    assert [s["rank"] for s in skipped] == [3, 4, 5]


@pytest.mark.parametrize("cap", [0, 1, 99])
def test_pick_never_exceeds_the_cap(cap):
    ranked = [R(f"https://d{i}.com/a", score=0.5) for i in range(10)]
    urls, skipped = loop._pick_urls_to_fetch(ranked, cap=cap)
    assert len(urls) <= cap
    assert len(urls) + len(skipped) == 10
