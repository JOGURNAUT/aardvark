"""The agent loop- streaming generator that orchestrates the 5 stages.

run(session_id, user_query) yields events at every stage transition + per-token
chunks during the answer stream. Persists turn to SQLite at the end.

Event shapes:
  {"type": "plan",         "queries": [...], "strategy": "..."}
  {"type": "search_done",  "results": [...], "queries": [{query, results, new}]}
  {"type": "fetch_done",   "pages": [...], "considered": n, "skipped": [...],
                           "term_coverage": [{term, pages, hits, of_pages}]}
  {"type": "select_done",  "snippets": [...]}
  {"type": "answer_token", "text": "..."}
  {"type": "answer_done",  "answer", "citations", "latency_ms", "provider"}
  {"type": "error",        "stage": "...", "message": "..."}

No chain-of-thought is ever streamed- only operational progress + final tokens.
"""
from __future__ import annotations
import re
import time
from typing import Iterator
from urllib.parse import urlparse

from config import MAX_PAGES_TO_FETCH, TURNS_BEFORE_SUMMARY
from storage import db
from tools import search as websearch
from tools import fetch as webfetch
from llm import client as llm
from agent import planner as planner_mod
from agent import selector as selector_mod
from agent import context_builder as ctx_mod
from agent import citations as cite_mod


#Stop words for the term-coverage check. Short on purpose: this is not trying
#to be a tokenizer, it is trying to stop "what", "is" and "the" being reported
#as terms the sources failed to cover.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "is", "it", "its", "many", "much",
    "of", "on", "or", "that", "the", "their", "there", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}


def _query_terms(query: str) -> list[str]:
    """The words worth checking the fetched text for."""
    words = re.findall(r"[\w'-]+", query.lower())
    return [w for w in dict.fromkeys(words) if len(w) > 2 and w not in _STOP]


def _term_coverage(query: str, pages: list[webfetch.FetchedPage]) -> list[dict]:
    """How many readable pages each query term actually appears in.

    Answers a question the trace could not answer before. Not "did we find
    anything" but "did the thing you asked about appear at all". A search for a
    person can return twenty pages that all match the surname and none that
    match the given name, and every count in the old trace looked healthy while
    the answer was built on people who were not the one asked about.

    Counted over fetched text, not search snippets: a snippet is chosen by the
    search engine to look relevant and the page behind it often is not.
    """
    readable = [p for p in pages if p.error is None and (p.text or "")]
    blobs = [(p.text or "").lower() for p in readable]
    out = []
    for term in _query_terms(query):
        out.append({
            "term": term,
            "pages": sum(1 for b in blobs if term in b),
            "hits": sum(b.count(term) for b in blobs),
            "of_pages": len(blobs),
        })
    return out


def _dedupe_results(pairs: list[tuple[str, list[websearch.SearchResult]]]):
    """Dedupe by URL across plan queries, keeping which query found each one.

    Returns (results, per_query_stats). The attribution is the point: without
    it the trace can say "20 unique results" and cannot say which of the five
    sub-questions produced nothing, which is the only thing worth knowing when
    an answer comes back empty.
    """
    seen: dict[str, websearch.SearchResult] = {}
    found_by: dict[str, list[str]] = {}
    stats: list[dict] = []

    for query, results in pairs:
        new = 0
        for r in results:
            if not r.url:
                continue
            if r.url not in seen:
                seen[r.url] = r
                found_by[r.url] = []
                new += 1
            if query not in found_by[r.url]:
                found_by[r.url].append(query)
        stats.append({"query": query, "results": len(results), "new": new})

    out = list(seen.values())
    return out, found_by, stats


def _rank_results(results: list[websearch.SearchResult]) -> list[websearch.SearchResult]:
    """Search order: score descending, unscored last. Shared by the picker and
    the rank reported for each fetched page, so a page's rank is the position
    it was actually chosen from rather than a second, separate ordering."""
    return sorted(results, key=lambda r: (r.score is None, -(r.score or 0)))


def _pick_urls_to_fetch(ranked: list[websearch.SearchResult],
                        cap: int = MAX_PAGES_TO_FETCH) -> tuple[list[str], list[dict]]:
    """Pick up to `cap` URLs, diversifying domains. Returns (urls, skipped).

    `skipped` is everything the cap and the domain rule left behind, with the
    reason. That list used to be thrown away, which is how a run could open 6
    of 20 results and report only the 6, leaving the trace looking like the
    search had found nothing more.
    """
    picked: list[str] = []
    skipped: list[dict] = []
    per_domain: dict[str, int] = {}

    for rank, r in enumerate(ranked, start=1):
        d = urlparse(r.url).netloc.replace("www.", "")
        if len(picked) >= cap:
            skipped.append({"url": r.url, "domain": d, "rank": rank,
                            "reason": "page cap reached"})
            continue
        if per_domain.get(d, 0) >= 2:  #at most 2 URLs per domain at fetch stage
            skipped.append({"url": r.url, "domain": d, "rank": rank,
                            "reason": "2 already picked from this domain"})
            continue
        picked.append(r.url)
        per_domain[d] = per_domain.get(d, 0) + 1
    return picked, skipped


def run(session_id: str, user_query: str) -> Iterator[dict]:
    """Main entry. Yields events as the agent works; persists at the end."""
    t0 = time.time()
    last_t = t0
    stage_latencies: dict[str, float] = {}   #per-stage seconds, persisted in save_turn

    #LOAD context- session + rolling summary + prior turns
    session = db.get_session(session_id)
    if not session:
        yield {"type": "error", "stage": "session", "message": f"unknown session_id {session_id}"}
        return
    rolling_summary = session.get("rolling_summary", "") or ""
    prior_turns = db.get_turns(session_id)

    #PLAN- LLM reformulates user query into 3-5 search queries
    try:
        plan_obj = planner_mod.plan(
            user_query,
            rolling_summary=rolling_summary or None,
            recent_turns=prior_turns or None,
        )
    except Exception as e:
        yield {"type": "error", "stage": "plan", "message": str(e)}
        plan_obj = planner_mod.SearchPlan(queries=[user_query], strategy="(planner failed, using raw query)")

    stage_latencies["plan"] = time.time() - last_t
    last_t = time.time()
    yield {"type": "plan", "queries": plan_obj.queries, "strategy": plan_obj.strategy}

    #SEARCH- one Tavily call per plan query, sequential
    per_query: list[tuple[str, list[websearch.SearchResult]]] = []
    for q in plan_obj.queries:
        try:
            per_query.append((q, websearch.search(q)))
        except Exception as e:
            per_query.append((q, []))
            yield {"type": "error", "stage": "search", "message": f"{q}: {e}"}

    all_results, found_by, query_stats = _dedupe_results(per_query)
    ranked = _rank_results(all_results)
    rank_of = {r.url: i for i, r in enumerate(ranked, start=1)}

    stage_latencies["search"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "search_done",
        #Per query, so a sub-question that returned nothing is visible instead
        #of being averaged away into the total.
        "queries": query_stats,
        "results": [
            {"title": r.title, "url": r.url, "rank": rank_of.get(r.url),
             "domain": urlparse(r.url).netloc.replace("www.", ""),
             "found_by": found_by.get(r.url, []),
             "snippet": r.snippet[:200]}
            for r in ranked
        ],
    }

    #FETCH- parallel via ThreadPool, 8s timeout per URL, trafilatura for main content
    urls, skipped = _pick_urls_to_fetch(ranked)
    pages: list[webfetch.FetchedPage] = []
    if urls:
        pages = webfetch.fetch_many(urls)
    stage_latencies["fetch"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "fetch_done",
        "pages": [
            {"url": p.url, "ok": p.error is None, "chars": len(p.text or ""),
             "error": p.error, "title": p.title, "domain": p.domain,
             #Rank is what makes a refusal legible: a 403 from the top-ranked
             #result and a 403 from the twentieth are not the same event.
             "rank": rank_of.get(p.url),
             "found_by": found_by.get(p.url, [])}
            for p in pages
        ],
        "considered": len(ranked),
        "skipped": skipped,
        #Which words of the question actually appear in what came back.
        "term_coverage": _term_coverage(user_query, pages),
    }

    #SELECT- chunk + embed + budget-and-diversity-aware pick
    snippets = selector_mod.select(pages, user_query)
    stage_latencies["select"] = time.time() - last_t
    last_t = time.time()
    yield {
        "type": "select_done",
        "snippets": [
            {"idx": i + 1, "title": s.title, "domain": s.domain,
             "chars": len(s.text), "score": round(s.score, 3), "url": s.url}
            for i, s in enumerate(snippets)
        ],
    }

    #CONTEXT- summarize older history if conversation has grown beyond threshold
    if len(prior_turns) > TURNS_BEFORE_SUMMARY:
        try:
            new_summary = ctx_mod.summarize_history(prior_turns)
            if new_summary:
                rolling_summary = new_summary
                db.update_rolling_summary(session_id, rolling_summary)
        except Exception as e:
            yield {"type": "error", "stage": "summarize", "message": str(e)}

    messages = ctx_mod.build_messages(
        user_query=user_query,
        snippets=snippets,
        rolling_summary=rolling_summary,
        recent_turns=prior_turns,
    )

    #ANSWER- stream tokens, BOTH yield to UI AND accumulate for citation post-process
    chunks: list[str] = []
    provider_used = "unknown"
    try:
        for chunk in llm.stream(messages, temperature=0.2, max_tokens=900):
            chunks.append(chunk)
            yield {"type": "answer_token", "text": chunk}
    except Exception as e:
        yield {"type": "error", "stage": "answer", "message": str(e)}
        #fall through to persist what we got

    raw_answer = "".join(chunks)
    stage_latencies["answer"] = time.time() - last_t

    #POST-PROCESS CITATIONS- regex extract [n] markers, drop hallucinated
    cleaned_answer, citations = cite_mod.extract_citations(raw_answer, snippets)
    #strip em/en dashes that the model sometimes uses despite the system prompt
    cleaned_answer = cleaned_answer.replace("—", ",").replace("–", "-")
    latency_ms = int((time.time() - t0) * 1000)

    yield {
        "type": "answer_done",
        "answer": cleaned_answer,
        "citations": cite_mod.citations_to_dicts(citations),
        "latency_ms": latency_ms,
        "stage_latencies": stage_latencies,
        "provider": provider_used,
    }

    #PERSIST- chat messages + full audit trail to SQLite
    db.add_message(session_id, "user", user_query)
    db.add_message(session_id, "assistant", cleaned_answer)
    db.save_turn(
        session_id=session_id,
        query=user_query,
        plan=plan_obj.strategy,
        search_queries=plan_obj.queries,
        urls_opened=[
            {"url": p.url, "title": p.title, "domain": p.domain,
             "retrieved_at": p.retrieved_at, "ok": p.error is None}
            for p in pages
        ],
        snippets=[
            {"idx": i + 1, "title": s.title, "domain": s.domain,
             "url": s.url, "score": s.score, "text": s.text}
            for i, s in enumerate(snippets)
        ],
        final_answer=cleaned_answer,
        citations=cite_mod.citations_to_dicts(citations),
        latency_ms=latency_ms,
        stage_latencies=stage_latencies,
    )
