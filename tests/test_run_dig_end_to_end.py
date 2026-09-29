"""End-to-end tests for run() and dig(), over a real SQLite database.

Every other test in this suite checks one function in isolation. These check
the seams between them, because both bugs this project actually shipped lived
in a seam rather than in a function.

The first: `select_done` carried `chars`, a length, and the front end declared
`text: string`. Each side was correct on its own and the app went blank at the
fourth stage.

The second is what these tests guard. `run()` stops at MAX_PAGES_TO_FETCH and
writes what it never opened into `turns.pending_urls_json`; `dig()` reads that
column back and continues from it. Nothing between the two is exercised by a
unit test. If `save_turn` stopped persisting snippet text tomorrow, `dig()`
would rebuild every prior Snippet with an empty string, re-pack, and answer
from half the evidence with nothing anywhere reporting a problem.

So the database here is real: a temporary file, the real schema, the real JSON
round trip. What is faked is only the outside world - the planner's LLM call,
Tavily, the network fetch, the embedding model, the tokenizer and the answer
stream. No key, no socket, no 80MB download.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

# Loaded by path, for the reason tests/test_loop_trace.py documents: test_api
# replaces sys.modules["agent.loop"] with a fake and pytest imports test
# modules alphabetically, so `from agent import loop` would hand this file that
# fake. Everything else agent/loop.py imports is the real module, which is the
# point - only the outside world is replaced, in the fixture below.
_spec = importlib.util.spec_from_file_location(
    "_loop_end_to_end",
    pathlib.Path(__file__).resolve().parent.parent / "agent" / "loop.py",
)
assert _spec and _spec.loader
loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop)

SearchResult = loop.websearch.SearchResult
FetchedPage = loop.webfetch.FetchedPage

QUERY = "How do beavers build dams"
# _query_terms drops "how" as a stop word and "do" as too short.
QUERY_TERMS = ["beavers", "build", "dams"]


# --------------------------------------------------------------- the fake world

# d1.com gets three results and everyone else two. That asymmetry is the whole
# fixture: with two per domain, the third d1 result is held back by the domain
# rule rather than by the page cap, which is the only way the same domain lands
# in both the first batch and the dig. Without it there is nothing to check the
# cap against once the two batches are merged.
LAYOUT = [("d1.com", 3), ("d2.com", 2), ("d3.com", 2),
          ("d4.com", 2), ("d5.com", 2), ("d6.com", 2)]

ALL_URLS = [f"https://{d}/{i}" for d, n in LAYOUT for i in range(n)]   # 13, in rank order

HELD_BY_DOMAIN_RULE = "https://d1.com/2"


def _domain(url: str) -> str:
    return url.split("//", 1)[1].split("/", 1)[0]


def _body(url: str, terms: list[str]) -> str:
    """One paragraph, comfortably over the hundred-character floor the chunker
    uses to discard boilerplate, so each page contributes exactly one snippet
    and the counts in these tests stay predictable."""
    return (f"{' '.join(terms)} is the subject here. The page at {url} covers "
            "it in enough words to survive the chunker's minimum paragraph "
            "length, which exists to throw away navigation text.")


class World:
    """The outside world, faked, keeping a record of what was asked of it.

    The records are what make "a dig does not search again" checkable. An
    assertion about cost has to be an assertion about a call that did not
    happen; nothing in the event stream can stand in for it.
    """

    def __init__(self):
        self.searches: list[str] = []
        self.fetched: list[list[str]] = []
        self.streamed: list[list[dict]] = []

        self.plan_queries = ["beaver dam construction", "beaver lodge materials"]
        # [1] and [2] are real; [99] is not, and must not survive to either the
        # stream's last event or the database. The em dash is here because the
        # loop replaces it despite the system prompt telling the model not to
        # use one, and a rule nothing exercises is a rule that quietly rots.
        self.answer = "Beavers build dams [1] and lodges [2] — from wood [99]."

        self.plan_error: Exception | None = None
        self.search_errors: dict[str, Exception] = {}
        self.broken: set[str] = set()     # urls fetch will report as failed
        self.silent: set[str] = set()     # urls whose text omits "dams"

    # -- planner
    def plan(self, user_query, rolling_summary=None, recent_turns=None):
        if self.plan_error:
            raise self.plan_error
        return loop.planner_mod.SearchPlan(queries=list(self.plan_queries),
                                           strategy="split into two sub-questions")

    # -- search: the two sub-queries overlap on three URLs, so deduplication
    #    and the "which query found this" attribution are both exercised.
    def search(self, query, max_results=5):
        self.searches.append(query)
        if query in self.search_errors:
            raise self.search_errors[query]
        urls = ALL_URLS[:7] if query == self.plan_queries[0] else ALL_URLS[4:]
        return [
            SearchResult(title=f"title {u}", url=u, snippet=f"snippet {u}",
                         score=1.0 - ALL_URLS.index(u) / 100)
            for u in urls
        ]

    # -- fetch
    def fetch_many(self, urls, **kw):
        self.fetched.append(list(urls))
        out = []
        for u in urls:
            if u in self.broken:
                out.append(FetchedPage(url=u, title="", text="", domain=_domain(u),
                                       retrieved_at="2026-09-30", error="403 Forbidden"))
                continue
            terms = [t for t in QUERY_TERMS if not (u in self.silent and t == "dams")]
            out.append(FetchedPage(url=u, title=f"title {u}", text=_body(u, terms),
                                   domain=_domain(u), retrieved_at="2026-09-30"))
        return out

    # -- answer
    def stream(self, messages, **kw):
        self.streamed.append(messages)
        for word in self.answer.split(" "):
            yield word + " "


class _FakeModel:
    """Scores snippets by their position in the batch, highest first.

    The real embedder is an 80MB download and what is under test here is the
    plumbing, not the maths. Position scoring keeps the ranking deterministic,
    which is what lets these tests say anything exact about what got packed.
    """

    def encode(self, texts, **kw):
        if len(texts) == 1:                      # the query
            return np.array([[1.0, 0.0]])
        n = len(texts)
        return np.array([[1.0 - i / (n + 1), i / (n + 1)] for i in range(n)])


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World()

    # A real database in a temporary file. connect() reads DB_PATH on every
    # call, so replacing the module attribute is enough and nothing here can
    # touch ./data/research.db.
    monkeypatch.setattr(loop.db, "DB_PATH", tmp_path / "research.db")
    loop.db.init_db()

    monkeypatch.setattr(loop.planner_mod, "plan", w.plan)
    monkeypatch.setattr(loop.websearch, "search", w.search)
    monkeypatch.setattr(loop.webfetch, "fetch_many", w.fetch_many)
    monkeypatch.setattr(loop.llm, "stream", w.stream)
    monkeypatch.setattr(loop.selector_mod, "_embedder", lambda: _FakeModel())
    monkeypatch.setattr(loop.selector_mod, "count_tokens", lambda t: len(t) // 4)
    return w


# ------------------------------------------------------------------- helpers

def one(events: list[dict], kind: str) -> dict:
    hits = [e for e in events if e["type"] == kind]
    assert len(hits) == 1, f"expected exactly one {kind!r}, got {len(hits)}"
    return hits[0]


def stages(events: list[dict]) -> list[str]:
    return [e["type"] for e in events if e["type"] != "answer_token"]


@pytest.fixture
def after_run(world):
    """A completed run, and the session it ran in. The starting point for dig."""
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))
    return sid, events


# =============================================================== run(), streaming

def test_the_stage_events_arrive_once_each_in_the_documented_order(after_run):
    # This sequence is a contract, not an implementation detail: the front end
    # advances its stage machine on it and renders nothing until it sees them.
    _sid, events = after_run
    assert stages(events) == [
        "plan", "search_done", "fetch_done", "select_done", "answer_done",
    ]


def test_every_token_is_streamed_before_the_answer_is_declared_done(after_run):
    _sid, events = after_run
    kinds = [e["type"] for e in events]
    assert kinds.index("answer_done") == len(kinds) - 1
    assert kinds.count("answer_token") > 1, "a single chunk is not a stream"


def test_the_streamed_tokens_join_back_into_what_the_provider_sent(world, after_run):
    # If they did not, the citation mapping would be running over different
    # text than the reader saw.
    _sid, events = after_run
    streamed = "".join(e["text"] for e in events if e["type"] == "answer_token")
    assert streamed.strip() == world.answer


# =============================================================== run(), searching

def test_a_result_two_sub_queries_found_is_carried_once_and_says_who_found_it(after_run):
    _sid, events = after_run
    done = one(events, "search_done")

    assert len(done["results"]) == len(ALL_URLS), "deduplication changed the count"
    both = [r for r in done["results"] if len(r["found_by"]) == 2]
    assert len(both) == 3, "the overlap between the two sub-queries was lost"


def test_each_sub_query_reports_what_it_added_rather_than_only_what_it_returned(after_run):
    # A sub-question that returns ten results and adds none is the interesting
    # case, and a single total hides it completely.
    _sid, events = after_run
    q1, q2 = one(events, "search_done")["queries"]

    assert (q1["results"], q1["new"]) == (7, 7)
    assert (q2["results"], q2["new"]) == (9, 6)


# ================================================================ run(), fetching

def test_the_page_cap_and_the_domain_rule_are_reported_as_separate_reasons(after_run):
    """Both hold results back and they mean opposite things.

    The cap means there was more worth reading and the run ran out of budget,
    which is a reason to dig. The domain rule means the search was dominated by
    one site, which is a reason to doubt the answer's independence. Collapsing
    them into "skipped: 7" throws the difference away.
    """
    _sid, events = after_run
    skipped = one(events, "fetch_done")["skipped"]
    reasons: dict[str, int] = {}
    for s in skipped:
        reasons[s["reason"]] = reasons.get(s["reason"], 0) + 1

    assert reasons == {"2 already picked from this domain": 1, "page cap reached": 6}
    held = next(s for s in skipped if s["reason"].startswith("2 already"))
    assert (held["domain"], held["url"]) == ("d1.com", HELD_BY_DOMAIN_RULE)


def test_the_run_opens_the_page_cap_and_no_more(world, after_run):
    _sid, events = after_run
    done = one(events, "fetch_done")
    expected = [u for u in ALL_URLS if u != HELD_BY_DOMAIN_RULE][:loop.MAX_PAGES_TO_FETCH]

    assert len(done["pages"]) == loop.MAX_PAGES_TO_FETCH
    assert done["considered"] == len(ALL_URLS)
    assert world.fetched == [expected]


def test_every_fetched_page_carries_the_rank_it_was_chosen_from(after_run):
    # A 403 from the top-ranked result and a 403 from the twentieth are not the
    # same event, and only the rank says which one happened.
    _sid, events = after_run
    ranks = [p["rank"] for p in one(events, "fetch_done")["pages"]]
    assert ranks == [1, 2, 4, 5, 6, 7], "rank 3 was held back by the domain rule"


def test_a_page_that_refused_the_request_is_reported_and_yields_no_snippet(world):
    world.broken = {ALL_URLS[0]}
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    failed = [p for p in one(events, "fetch_done")["pages"] if not p["ok"]]
    assert len(failed) == 1
    assert failed[0]["error"] == "403 Forbidden"
    assert len(one(events, "select_done")["snippets"]) == 5, "the failed page was mined anyway"


def test_term_coverage_counts_the_question_words_in_what_could_actually_be_read(world):
    """The failure this whole trace was rebuilt for.

    A search can return pages that all match one word of the question and none
    that match another, and every count in the old trace looked healthy while
    the answer was built from sources that were not about the subject at all.
    """
    world.silent = {ALL_URLS[0], ALL_URLS[1]}
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    coverage = {c["term"]: c for c in one(events, "fetch_done")["term_coverage"]}
    assert set(coverage) == set(QUERY_TERMS)
    assert coverage["beavers"]["pages"] == 6
    assert coverage["dams"]["pages"] == 4, "two pages never mentioned it"
    assert coverage["dams"]["of_pages"] == 6


def test_term_coverage_is_measured_over_readable_pages_only(world):
    # A page that 403'd has no text, so counting it in the denominator would
    # report a coverage gap where there is only a fetch failure.
    world.broken = {ALL_URLS[0]}
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    assert all(c["of_pages"] == 5 for c in one(events, "fetch_done")["term_coverage"])


# ================================================================= run(), citing

def test_a_citation_marker_pointing_at_nothing_never_reaches_the_reader(after_run):
    """The guard that makes the whole project's claim true.

    [99] is in the text the provider streamed. If it survived, the answer would
    carry a reference the reader cannot follow, and the system would be
    inventing sources - the one failure this design exists to prevent.
    """
    _sid, events = after_run
    done = one(events, "answer_done")

    assert "[99]" not in done["answer"]
    assert "[1]" in done["answer"] and "[2]" in done["answer"]
    assert [c["marker"] for c in done["citations"]] == [1, 2]


def test_the_stripped_marker_never_reaches_the_database_either(after_run):
    # The stream and the stored turn are written from the same cleaned string,
    # and a second cleaning path would be a second place to get it wrong.
    sid, _events = after_run
    turn = loop.db.get_turns(sid)[-1]
    assert "[99]" not in turn["final_answer"]
    assert [c["marker"] for c in turn["citations"]] == [1, 2]


def test_every_citation_resolves_to_a_snippet_that_was_actually_selected(after_run):
    _sid, events = after_run
    chosen = {s["url"] for s in one(events, "select_done")["snippets"]}
    for c in one(events, "answer_done")["citations"]:
        assert c["url"] in chosen


def test_the_em_dash_the_model_keeps_using_is_replaced(after_run):
    _sid, events = after_run
    assert "—" not in one(events, "answer_done")["answer"]


# =============================================================== run(), persisting

def test_the_stored_turn_is_the_one_that_was_streamed(after_run):
    sid, events = after_run
    done = one(events, "answer_done")
    turn = loop.db.get_turns(sid)[-1]

    assert turn["query"] == QUERY
    assert turn["final_answer"] == done["answer"]
    assert turn["citations"] == done["citations"]
    assert len(turn["snippets"]) == len(one(events, "select_done")["snippets"])


def test_every_stage_reports_how_long_it_took_in_the_event_and_in_the_row(after_run):
    sid, events = after_run
    done = one(events, "answer_done")
    turn = loop.db.get_turns(sid)[-1]

    assert set(done["stage_latencies"]) == {"plan", "search", "fetch", "select", "answer"}
    assert set(turn["stage_latencies"]) == set(done["stage_latencies"])
    assert turn["latency_ms"] == done["latency_ms"]


def test_the_snippet_text_is_stored_and_not_only_its_length(after_run):
    """The one that would catch a silent halving of dig()'s evidence.

    dig() rebuilds every prior Snippet from this column. If it held lengths and
    not text - which is exactly the shape the select_done event carries, and
    exactly the confusion that blanked the front end once already - dig would
    re-pack empty strings, answer from the new batch alone, and report nothing
    wrong.
    """
    sid, _events = after_run
    stored = loop.db.get_turns(sid)[-1]["snippets"]
    assert stored, "nothing was selected"
    assert all(s["text"].strip() for s in stored)
    assert all(isinstance(s["score"], float) for s in stored)


def test_what_was_never_opened_is_stored_so_a_later_dig_can_use_it(after_run):
    sid, events = after_run
    assert loop.db.get_turns(sid)[-1]["pending_urls"] == one(events, "fetch_done")["skipped"]


def test_both_sides_of_the_exchange_are_recorded_as_messages(after_run):
    sid, events = after_run
    msgs = loop.db.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == QUERY
    assert msgs[1]["content"] == one(events, "answer_done")["answer"]


# =============================================================== run(), degrading

def test_an_unknown_session_yields_one_error_and_writes_nothing(world):
    events = list(loop.run("no-such-session", QUERY))
    assert len(events) == 1
    assert events[0]["type"] == "error" and events[0]["stage"] == "session"
    assert world.searches == [], "it searched on behalf of a session that does not exist"


def test_a_planner_failure_falls_back_to_the_raw_query_and_still_answers(world):
    # Losing the planner costs breadth, not the run. Refusing to proceed would
    # turn one degraded provider into a total outage.
    world.plan_error = RuntimeError("planner returned unparseable json")
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    err = one(events, "error")
    assert err["stage"] == "plan"
    assert one(events, "plan")["queries"] == [QUERY]
    assert one(events, "answer_done")["answer"]
    assert loop.db.get_turns(sid), "a degraded run still owes a stored turn"


def test_one_failed_sub_query_does_not_take_the_others_with_it(world):
    world.search_errors = {world.plan_queries[1]: RuntimeError("tavily 502")}
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    err = one(events, "error")
    assert err["stage"] == "search" and "tavily 502" in err["message"]

    q1, q2 = one(events, "search_done")["queries"]
    assert q1["results"] == 7
    assert (q2["results"], q2["new"]) == (0, 0), "a failed query must report zero, not vanish"
    assert one(events, "answer_done")["answer"]


def test_a_search_that_returns_nothing_at_all_still_completes_the_stages(world):
    world.search_errors = {q: RuntimeError("tavily down") for q in world.plan_queries}
    sid = loop.db.create_session("t")
    events = list(loop.run(sid, QUERY))

    assert stages(events)[-4:] == ["search_done", "fetch_done", "select_done", "answer_done"]
    assert one(events, "fetch_done")["pages"] == []
    assert one(events, "select_done")["snippets"] == []
    # With no snippets, every marker the model emits points at nothing.
    assert one(events, "answer_done")["citations"] == []
    assert world.fetched == [], "it fetched from an empty result set"


# ========================================================================= dig()

def test_a_dig_opens_the_unread_results_without_searching_again(world, after_run):
    """The property that makes dig cheap enough to offer as a button.

    The results are already ranked and already stored, so continuing is a fetch
    and a re-rank. A dig that searched again would bill a second time for
    results the run already had.
    """
    sid, _events = after_run
    world.searches.clear()
    events = list(loop.dig(sid))

    assert world.searches == [], "the dig went back to the search provider"
    assert one(events, "dig_start")["opening"] == 6


def test_a_dig_obeys_the_same_caps_and_leaves_the_remainder_for_the_next_one(after_run):
    # Same caps on purpose: a dig that ignored the domain rule would open six
    # more pages from the site that already dominated the search.
    sid, _events = after_run
    events = list(loop.dig(sid))

    start = one(events, "dig_start")
    assert (start["opening"], start["remaining_after"]) == (6, 1)
    assert len(one(events, "fetch_done")["skipped"]) == 1
    assert loop.db.get_turns(sid)[-1]["pending_urls"] == one(events, "fetch_done")["skipped"]


def test_a_dig_never_reopens_a_page_the_run_already_read(world, after_run):
    """Guards a filter whose absence would be invisible.

    Re-fetching a page costs a request and produces a duplicate snippet that
    competes with the new evidence for the same eight slots, so the dig would
    appear to work while adding less than it claimed. The pending list is
    rewritten here to hold an already-opened URL because no current code path
    produces one - which is precisely why the filter needs holding down.
    """
    sid, _events = after_run
    opened = world.fetched[0][0]
    turn = loop.db.get_turns(sid)[-1]
    with loop.db.connect() as conn:
        conn.execute(
            "UPDATE turns SET pending_urls_json = ? WHERE id = ?",
            (json.dumps([{"url": opened, "domain": "d1.com", "rank": 1,
                          "reason": "page cap reached"}] + turn["pending_urls"]),
             turn["id"]),
        )

    world.fetched.clear()
    list(loop.dig(sid))
    assert opened not in world.fetched[0]


def test_prior_evidence_survives_the_database_and_is_repacked_with_the_new(after_run):
    """The seam. Nothing else in the suite crosses it.

    The prior snippets in this list came out of SQLite as JSON and were rebuilt
    into Snippet objects. If any part of that round trip lost the text, they
    would still be here and still be counted, and contribute nothing.
    """
    sid, run_events = after_run
    prior_urls = {s["url"] for s in one(run_events, "select_done")["snippets"]}

    events = list(loop.dig(sid))
    merged = one(events, "select_done")["snippets"]

    assert len(merged) == loop.selector_mod.MAX_SNIPPETS_PER_TURN
    assert any(s["url"] in prior_urls for s in merged), "the earlier evidence was dropped"
    assert any(s["url"] not in prior_urls for s in merged), "the dig added nothing"
    assert all(s["chars"] > 0 for s in merged), "a snippet survived with no text"


def test_the_domain_cap_holds_across_both_batches_rather_than_within_each(after_run):
    """Why dig re-packs instead of appending.

    Three d1.com snippets exist by now: two the run kept and one the dig just
    opened. Appending would leave three from one domain in the context and
    quietly undo the diversity rule the first pass enforced.
    """
    sid, _events = after_run
    merged = one(list(loop.dig(sid)), "select_done")["snippets"]

    assert len([s for s in merged if s["domain"] == "d1.com"]) == 2


def test_a_dig_marks_which_evidence_it_actually_added(after_run):
    # Without the flag the UI re-presents the same snippets as if they were new,
    # and a dig that found nothing looks identical to one that found plenty.
    sid, run_events = after_run
    prior_urls = {s["url"] for s in one(run_events, "select_done")["snippets"]}

    merged = one(list(loop.dig(sid)), "select_done")["snippets"]
    for s in merged:
        assert s["is_new"] is (s["url"] not in prior_urls)


def test_a_dig_writes_its_own_turn_so_the_chain_can_continue(after_run):
    sid, _events = after_run
    list(loop.dig(sid))

    turns = loop.db.get_turns(sid)
    assert len(turns) == 2
    assert turns[1]["query"] == QUERY, "a dig answers the original question"
    assert turns[1]["plan"] == "dig: opened 6 more of 7 unread results"
    assert turns[1]["search_queries"] == turns[0]["search_queries"]


def test_a_second_dig_opens_the_last_result_and_a_third_has_nothing_left(after_run):
    sid, _events = after_run
    list(loop.dig(sid))

    second = list(loop.dig(sid))
    assert one(second, "dig_start") == {"type": "dig_start", "query": QUERY,
                                        "opening": 1, "remaining_after": 0}

    third = list(loop.dig(sid))
    assert len(third) == 1
    assert "already been opened" in third[0]["message"]


def test_digging_a_session_that_has_never_run_says_so(world):
    sid = loop.db.create_session("t")
    events = list(loop.dig(sid))

    assert len(events) == 1
    assert events[0]["type"] == "error" and events[0]["stage"] == "dig"
    assert world.fetched == []


def test_digging_an_unknown_session_errors_the_way_a_run_does(world):
    events = list(loop.dig("no-such-session"))
    assert len(events) == 1
    assert events[0]["type"] == "error" and events[0]["stage"] == "session"
