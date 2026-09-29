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

import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass

import pytest

# Stub what agent.loop pulls in transitively, but ONLY where the real thing is
# not installed.
#
# The version of this that stubbed unconditionally worked here and would have
# broken CI. pytest imports test modules alphabetically, this file sorts before
# test_selector.py, and a stub left in sys.modules is what the next module
# imports. So test_selector would have received a fake SentenceTransformer in
# the one environment where the real one exists.
_FAKE = {
    "tavily": {"TavilyClient": object},
    "trafilatura": {},
    "tiktoken": {},
    "sentence_transformers": {"SentenceTransformer": object},
    "groq": {"Groq": object},
}

for _name, _attrs in _FAKE.items():
    if _name in sys.modules or importlib.util.find_spec(_name) is not None:
        continue                                   # the real package is available
    _mod = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_mod, _k, _v)
    sys.modules[_name] = _mod

if "google.genai" not in sys.modules and importlib.util.find_spec("google.genai") is None:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.genai")
    _gg.Client = object                                        # type: ignore[attr-defined]
    _gt = types.ModuleType("google.genai.types")
    _gg.types = _gt                                            # type: ignore[attr-defined]
    _g.genai = _gg                                             # type: ignore[attr-defined]
    sys.modules.setdefault("google", _g)
    sys.modules["google.genai"] = _gg
    sys.modules["google.genai.types"] = _gt

# Load agent/loop.py from its path under a private name, rather than
# `from agent import loop`.
#
# tests/test_api.py replaces sys.modules["agent.loop"] with a fake so it can
# drive the SSE layer without an agent behind it, and pytest imports test
# modules alphabetically, so test_api runs first and that fake is what
# `from agent import loop` returns here. CI caught it; running these files in a
# different order locally did not.
#
# Loading by path sidesteps the question entirely: this module gets the real
# file whatever any other test has done to sys.modules, and puts nothing back
# that could affect a later one.
_spec = importlib.util.spec_from_file_location(
    "_loop_under_test",
    pathlib.Path(__file__).resolve().parent.parent / "agent" / "loop.py",
)
assert _spec and _spec.loader
loop = importlib.util.module_from_spec(_spec)                   # noqa: E402
_spec.loader.exec_module(loop)


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


# ----------------------------------------------------------- digging deeper

def _pending(*specs):
    """specs are (domain, n) -> n urls on that domain, in the given order."""
    out = []
    rank = 1
    for domain, n in specs:
        for i in range(n):
            out.append({"url": f"https://{domain}/p{i}", "domain": domain,
                        "rank": rank, "reason": "page cap reached"})
            rank += 1
    return out


def test_next_batch_takes_the_cap_and_returns_the_rest():
    batch, leftover = loop._next_batch(_pending(("a.com", 1), ("b.com", 1), ("c.com", 1),
                                                ("d.com", 1), ("e.com", 1)), cap=3)
    assert len(batch) == 3
    assert len(leftover) == 2


def test_next_batch_keeps_the_domain_rule_the_first_pass_used():
    # A dig that ignored it would open six more pages from the one site that
    # already dominated the search, which is the failure the rule prevents and
    # is no less a failure the second time round.
    batch, leftover = loop._next_batch(_pending(("a.com", 5), ("b.com", 1)), cap=6)
    assert batch == ["https://a.com/p0", "https://a.com/p1", "https://b.com/p0"]
    assert len(leftover) == 3


def test_next_batch_preserves_order_so_higher_ranked_results_go_first():
    batch, _ = loop._next_batch(_pending(("a.com", 1), ("b.com", 1), ("c.com", 1)), cap=2)
    assert batch == ["https://a.com/p0", "https://b.com/p0"]


def test_next_batch_skips_entries_with_no_url():
    batch, leftover = loop._next_batch([{"domain": "a.com"}, {"url": "https://b.com/x"}])
    assert batch == ["https://b.com/x"]
    assert leftover == []


def test_next_batch_infers_the_domain_when_it_was_not_stored():
    # pending_urls rows written by an older version have no domain field, and
    # the rule still has to apply to them.
    pending = [{"url": "https://a.com/1"}, {"url": "https://a.com/2"},
               {"url": "https://a.com/3"}]
    batch, leftover = loop._next_batch(pending, cap=6)
    assert len(batch) == 2
    assert len(leftover) == 1


def test_next_batch_on_an_empty_list_is_empty():
    assert loop._next_batch([]) == ([], [])
