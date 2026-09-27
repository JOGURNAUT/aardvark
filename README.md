# Beaver: Deep Research Agent

A streaming web-research agent that plans, searches, fetches, selects context, and answers with grounded citations. Built without LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack - the agent loop and orchestration are hand-rolled Python.

---

## Demo video

**https://www.loom.com/share/7dee1696b51e44c79b61c4d0a91b7742**

(Walkthrough of the Streamlit UI showing live query, streaming progress events, citations, confidence badge, and per-stage latency.)

---

## Quick start

```bash
# 1. clone / unzip and cd into project root
python -m venv .venv
.venv\Scripts\activate              # on Windows
# source .venv/bin/activate         # on macOS / Linux

# 2. install deps (~3 min - pulls torch for embeddings)
pip install -r requirements.txt

# 3. set API keys
cp .env.example .env
# then edit .env and paste your keys for:
#   GROQ_API_KEY     (console.groq.com - free)
#   GEMINI_API_KEY   (aistudio.google.com/apikey - free)
#   TAVILY_API_KEY   (app.tavily.com - 1000 req/mo free)

# 4. sanity check (storage + search + both LLMs)
python smoke_test.py

# 5. launch the UI
streamlit run ui/app.py
# opens http://localhost:8501

# CLI alternative
python cli.py

# Run the evaluation harness
python eval/run_eval.py            # all 17 questions, ~10 min
python eval/run_eval.py --limit 2  # quick smoke

# Unit tests (no API keys, no network, ~1s)
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

---

## Web front end (React + TypeScript)

A third interface over the same `agent/loop.py`, after Streamlit and Gradio.

```bash
# terminal 1 - the API
pip install -r requirements-api.txt
uvicorn api.server:app --port 8000 --reload

# terminal 2 - the dev server (proxies /api to 8000)
cd web && npm install && npm run dev      # http://localhost:5173
```

In the container there is no second process: `npm run build` output is copied
into the image and FastAPI serves it alongside the API on one port.

### Why it exists

The loop has always claimed to be decoupled from its UI. Two Python front ends
assert that; one in another language tests it. Writing this surfaced two places
where the claim was softer than stated:

- `answer_done` re-emits the whole answer after the citation guard has run, so
  a consumer that appends tokens and then appends the final answer prints it
  twice. Streamlit never hit this because it rebuilds from state on each rerun.
  **Not fixed in the loop** — the contract is reasonable and changing it would
  break the other two front ends. The reducer in `App.tsx` replaces rather than
  appends, and says why. It is a documented sharp edge, not a repaired one.
- The stream had no way to say "this run was abandoned". A turn with no
  `answer_done` meant either a crash or a closed tab, and nothing distinguished
  them in the audit trail. **Partly addressed**: `api/server.py` now logs the
  two cases differently. The SQLite audit trail still cannot tell them apart.

### Three decisions worth the space

**SSE over `fetch`, not `EventSource`.** `EventSource` is the obvious tool and
the wrong one: it only issues GET, so the query rides in the URL, and closing it
does not reliably reach the server. `fetch` gives a request body and an
`AbortController`. The cost is reimplementing the wire format in `web/src/api.ts`,
including the fact that a chunk boundary can land in the middle of a JSON object.

**Cancellation is deterministic, and getting there took a rewrite.** The first
version was a sync generator handed to `StreamingResponse`. Starlette drives
that with `iterate_in_threadpool`, which stops pulling when the client goes but
never calls `.close()`, so `GeneratorExit` arrives whenever garbage collection
decides, and the provider response stays open until then. The stream is now an
async generator that drives the sync one by hand and closes it in a `finally`.
Starlette does call `aclose()` on disconnect, and `finally` runs on
cancellation. `tests/test_api.py` asserts it at the generator level rather than
through `TestClient`, which buffers the whole body and cannot express a
disconnect at all.

**Events are forwarded, not mapped.** The server re-serialises whatever the loop
yields. A new event type reaches the browser without a change to `api/server.py`,
and `web/src/App.tsx` counts the ones it does not recognise instead of dropping
them, so drift between the two shows up in the UI rather than as silence.

### Known limitations

- `web/src/types.ts` is hand-written against a docstring in `loop.py`. It can
  drift; the unknown-event counter is what makes a drift visible, not a fix.
- No frontend test runner. `reduce()` in `App.tsx` is exported and pure
  specifically so one can be added without restructuring anything.
- One session per page load, created on mount. There is no session list in this
  UI yet, though `/api/sessions` returns one.

## Part 1 - Design note

### Target user & problem

Analysts, journalists, researchers, and operators who currently keep 10+ browser tabs open trying to triangulate a fact and end up pasting blocks of text into ChatGPT. They need: (a) **fresh** information from the live web, (b) **citations they can audit**, (c) **explicit uncertainty** when evidence is weak, and (d) the ability to keep **session context** across follow-up questions.

The system is designed for the workflow that consumer chat tools don't serve well: questions where the *provenance of the answer matters as much as the answer itself*.

### Definition of "deep research" used here

A query is treated as deep research when the agent:

1. Reformulates the user question into multiple complementary search queries (not just the raw text).
2. Reads the actual page content, not only search-result snippets.
3. Selects evidence by relevance, recency, and source diversity - never letting one domain dominate.
4. Grounds every claim in a numbered snippet from the evidence pool. URLs are never typed by the LLM; they are mapped from snippet metadata after generation.
5. Surfaces disagreement when sources conflict, and refuses to answer when evidence is insufficient.

The agent is intentionally **two LLM calls** (planner + answer), not a tool-use loop. The reasoning is that a deterministic select-then-answer pipeline lets us audit exactly what evidence the model saw, which makes both grounding guarantees and evaluation possible.

### Success metrics (5 chosen)

| Metric | Why it captures research quality |
|---|---|
| **Faithfulness** (LLM-judge) | Fraction of factual claims in the answer that are supported by the retrieved snippets. This is the direct test for hallucination. |
| **Citation precision** (LLM-judge) | For each citation event in the answer, does the cited snippet actually support the adjacent claim? Catches "right answer, wrong source" failures. |
| **Refusal correctness** (binary + LLM-verify) | On questions where evidence is genuinely insufficient (e.g., future revenue), does the agent admit uncertainty instead of confabulating? |
| **Conflict handling** (binary + LLM-verify) | When retrieved sources disagree, does the answer explicitly flag the disagreement and cite both sides? |
| **Source diversity** (unique domains per answer) | Operational counter that proves the selector's per-domain cap is doing real work. |

Latency, gold-fact coverage, and multi-source coverage are logged alongside as secondary signals.

### Data flow

```
User query
   │
   ▼
[ PLAN ]      ── LLM call #1 (planner). Output: 3-5 search queries + strategy.
   │             Event: "plan"
   ▼
[ SEARCH ]    ── Tavily, one call per plan query. Dedupe by URL.
   │             Event: "search_done"
   ▼
[ FETCH ]     ── ThreadPool, top-N URLs (domain-diversified), 8s timeout,
   │             trafilatura main-content extraction.
   │             Event: "fetch_done"
   ▼
[ SELECT ]    ── Paragraph-first chunking → MiniLM-L6 cosine similarity →
   │             apply gentle recency multiplier (publish_date from trafilatura) →
   │             pick top-K within token budget, max 2 snippets per domain.
   │             Event: "select_done"
   ▼
[ ANSWER ]    ── LLM call #2, STREAMING. Prompt = rolling summary + recent
   │             turns + numbered snippets + strict citation rules.
   │             Per-token events: "answer_token".
   ▼
[ POST ]      ── Regex extracts [n] markers → map to snippet metadata →
   │             drop any [n] out of range (defense against hallucinated URLs).
   │             Event: "answer_done" with citations + latency_ms.
   ▼
[ PERSIST ]   ── Save full turn (plan, queries, URLs, snippets, answer,
                 citations, latency) to SQLite. Trigger rolling summary if
                 conversation > N turns.
```

Streaming uses Python generators that the UI (Streamlit) and CLI both consume. No chain-of-thought is ever streamed - only operational progress + the final answer tokens.

### Risks and limitations

| Risk | Current mitigation | What still breaks |
|---|---|---|
| LLM hallucinated URL | LLM only emits `[n]` markers; URLs come from snippet metadata via post-processing. | If the LLM cites a snippet that does support a *related* fact but not exactly the claim, we catch it in citation-precision eval but not at runtime. |
| Low-quality sources | Tavily relevance score + max-2-per-domain diversity cap. | Tavily can still return SEO content farms. No domain whitelist or quality scoring yet. |
| Conflicting sources missed | Selector pulls from multiple domains; system prompt instructs the answerer to flag disagreement. | If search returns only consensus content (e.g., Mt Everest height post-2020), the agent has nothing to flag. |
| Context length blowup | Token-budget cap on selected snippets; rolling summary kicks in after `TURNS_BEFORE_SUMMARY` turns. | Summary itself is single-shot; long sessions could compound paraphrase loss. |
| Search/LLM rate limits | Provider fallback (Groq → Gemini). Tavily has no fallback yet. | Sustained Tavily 429 would degrade quality of search; Serper would be the cheapest add. |
| Slow page fetches | 8s per-URL timeout, parallel via thread pool. | A single slow URL on the critical path still adds latency to that turn. |

### Two future improvements

1. **Iterative evidence loop.** When the answer LLM signals low confidence or missing evidence on a sub-claim, automatically issue follow-up search queries scoped to that gap, rather than returning the partial answer. This converts the current one-shot pipeline into a bounded multi-hop loop without sacrificing the determinism of single-pass selection.

2. **Per-claim citation post-checking at runtime.** Today citation precision is only measured offline by the evaluator. The same judge prompt could run as a streaming sidecar: each answer paragraph is checked against its cited snippet before being shown to the user. Failures are silently re-prompted or rendered with a "low-confidence" badge. Costs roughly 1.5× the answer call but raises the floor on trust.

---

## Part 2 - Technical implementation

### Project layout

```
agent/                 # the hand-rolled agent loop
  planner.py           # Stage 1: reformulate user Q -> 3-5 search queries
  selector.py          # Stage 4: chunk + embed + rank + budget-aware select
  context_builder.py   # Build LLM messages (system + history + snippets)
  citations.py         # Stage 5b: post-process [n] markers, drop hallucinated
  loop.py              # The streaming generator that orchestrates 1-5
llm/
  client.py            # Unified Groq+Gemini client w/ fallback (complete & stream)
tools/
  search.py            # Tavily wrapper
  fetch.py             # httpx + trafilatura, parallel via ThreadPoolExecutor
storage/
  db.py                # SQLite: sessions, messages, turns
ui/
  app.py               # Streamlit chat + live progress + session switcher
eval/
  dataset/questions.json   # 17 questions across 6 categories
  judge.py             # LLM-as-judge primitives (Gemini judges Groq outputs)
  metrics.py           # score_turn() + aggregate()
  run_eval.py          # runner; writes results.json, summary.json, report.md
tests/
  test_citations.py    # marker mapping + hallucinated-marker stripping
  test_selector.py     # chunking, recency, cosine, budget walk (embedder stubbed)
cli.py                 # CLI entrypoint (mirrors UI behavior)
config.py              # env-driven config (tokens, timeouts, model names)
smoke_test.py          # 4-check sanity script
```

### Key design choices and why

| Choice | Rationale |
|---|---|
| **No framework** | Explicit agent loop is auditable. Every prompt, every retry, every fallback is in the call stack and the SQLite log. With LangGraph this would be hidden in an abstraction. |
| **Two LLM calls, not tool-use loop** | Deterministic select-then-answer pipeline. Lets us inspect exactly what evidence the answerer saw, which is what makes faithfulness and citation-precision evaluation possible. |
| **Local embeddings (sentence-transformers MiniLM-L6)** | 80MB one-time download, ~50ms per encode. Free, deterministic, reproducible for eval. Avoids a vector DB while still getting semantic rerank. |
| **Numbered citation markers, URLs post-processed** | The LLM is structurally incapable of typing a URL - it only emits `[1]`, `[2]`. URLs come from snippet metadata after generation. Hallucinated `[99]` is dropped. |
| **Provider fallback (Groq → Gemini)** | Production realism. Groq's free tier rate-limits; Gemini covers gracefully. The agent loop never knows which provider answered. |
| **SQLite, three tables** | `messages` is the human-facing transcript. `turns` is the auditable agent trail (plan, search queries, URLs opened, snippets used, citations, latency). Separation lets the UI render a clean chat while the evaluator and any debugging tool have the full record. |
| **Generators for streaming** | Single agent function. Same generator drives the CLI (`for ev in run(...)`), Streamlit (`for ev in run(...)`), and a FastAPI `StreamingResponse` if we wanted to expose it. No threading or callbacks. |
| **Cross-model LLM-as-judge** | Gemini judges Groq's outputs (and vice versa). Avoids the self-grading bias that plagues same-model evaluation. |

---

## Example conversations

### Single-turn, multi-fact question
```
> What is Sarvam AI, where are they based, and who founded them?

[PLAN] Diverse queries covering origin, location, and leadership.
   1. "Sarvam AI company overview"
   2. "Sarvam AI headquarters location India"
   3. "Sarvam AI founders Vivek Raghavan Pratyush Kumar"
[SEARCH] 12 unique results across queries
[FETCH] 6/6 pages extracted (sarvam.ai, careers.kula.ai, billionhopes.ai, ...)
[SELECT] 4 snippets chosen (4 unique domains, 3.2K chars)
[ANSWER] Sarvam AI is an AI company that aims to build "AI for all from India"
with a focus on developing a full-stack AI platform [3]. They are based in
India, with locations in Bengaluru and Delhi [5]. Sarvam AI was founded by
Dr. Vivek Raghavan and Dr. Pratyush Kumar in August 2023 [1,2]. Dr. Vivek
Raghavan has experience building India's digital public infrastructure,
while Dr. Pratyush Kumar has led India's open-source AI efforts across
Indian languages [1].
[CITATIONS] 4 sources (sarvam.ai, billionhopes.ai, careers.kula.ai)
```

### Multi-turn follow-up - context resolution
```
> Have they raised any funding? Who are their investors?

[PLAN] Follow-up about funding for the subject of prior turn (Sarvam AI).
   1. "Sarvam AI funding round investors"
   2. "Sarvam AI Series A 2025"
   3. "Lightspeed Khosla Sarvam investment"
[SEARCH] 9 unique results
[FETCH] 4/5 pages extracted (1 timeout, gracefully skipped)
[SELECT] 1 snippet chosen (linkedin.com)
[ANSWER] Sarvam AI is reportedly in talks to raise 250 million in a new
funding round, which could value the company at around 1.5 billion [1]. They
have already been backed by investors such as Lightspeed and Khosla
Ventures [1]. Additionally, potential participation from Nvidia, Accel, and
HCLTech is also mentioned in the funding round [1]. However, it is not
confirmed if they have successfully raised the funding, as the development
has not yet been independently verified [1].
[CITATIONS] 1 source (linkedin.com)
```

Note: "they" was resolved to Sarvam AI because the planner sees prior-turn Q+A.
Note: the closing sentence is the system prompt's uncertainty rule firing - evidence was weak (one LinkedIn post), so the agent said so explicitly.

### Insufficient-evidence question - correct refusal
```
> What will OpenAI's revenue be in Q3 of 2027?

[ANSWER] We cannot confidently predict OpenAI's revenue in Q3 of 2027 [1,6].
The available snippets provide annual revenue projections, such as reaching
$100B in Annual Recurring Revenue by mid-2027 [1,6], but do not provide
quarterly breakdowns. To answer this question, additional research on
OpenAI's quarterly financial reports or more detailed revenue projections
would be necessary.
```

### Conflicting-sources question - disagreement surfaced
```
> What is the population of Mumbai, India?

[ANSWER] The population of Mumbai, India is estimated to be around
21,782,818 [1], 27,719,000 [2], or 22,539,000 [3] in 2026. The discrepancy
in population estimates may be due to different sources and methods used
for projections. Source [1] reports a population of 21,782,818, while
source [2] reports a higher estimate of 27,719,000, and source [3] reports
22,539,000. Additional research would be necessary to determine the most
accurate estimate.
```

---

## Part 3 - Evaluation

### Methodology

17 questions across 6 categories (3 each, except 2 multi-turn). Each question is tagged with an `expected_behavior` (e.g., `must_refuse`, `must_flag_conflict`, `must_cite_at_least_2_sources`), and only metrics relevant to those tags are computed. A factual lookup is not scored on "refusal correctness"; a future-revenue question is not scored on "faithfulness". This avoids penalizing the agent on metrics that don't apply.

**Judges.** Gemini-2.5-flash judges Groq-Llama outputs, always at temperature 0. Cross-model judging reduces the self-grading bias that plagues same-model evals. When a judge fails to return parseable JSON, the metric is recorded as `null` (skipped in aggregates) rather than `0` (which would conflate failures with bad answers).

**Reproducibility.** All raw judge responses are logged. Every run writes `results.json` (per-question), `summary.json` (aggregates by category), and `report.md` (human-readable) into a timestamped folder under `eval/results/`.

### Dataset categories

| Category | n | What it tests |
|---|---|---|
| `factual_single_hop` | 3 | Baseline grounding for unambiguous facts |
| `multi_hop` | 3 | Whether the planner reformulates effectively |
| `comparison` | 3 | Source diversity + faithfulness on contrastive Qs |
| `insufficient_evidence` | 3 | Refusal behavior on questions with no public answer |
| `conflicting_sources` | 3 | Whether the agent flags disagreement and cites both sides |
| `multi_turn` | 2 | Whether a follow-up resolves against the prior turn. The follow-up runs in the *same* session, so this measures context carry-over rather than a cold question |

### Results

Headline numbers from a **15-question run**, taken before the two multi-turn
questions were added. The dataset is now 17 questions across 6 categories, so these
figures do not cover the multi-turn category. Re-run with `python eval/run_eval.py`
to regenerate against the current set.

| Metric | Overall | Reading |
|---|---|---|
| `avg_relevance` | **0.677** | answer addresses the question (judge-scored, weakest on refusal questions - see note) |
| `avg_faithfulness` | **0.859** | claims grounded in retrieved snippets |
| `avg_citation_precision` | **0.876** | cited snippets actually support the claim attached |
| `avg_refusal_score` | **0.875** | correctly refused on insufficient-evidence questions |
| `avg_conflict_score` | **0.333** | flagged disagreement when sources conflicted (see note) |
| `avg_multi_source` | **1.0** | hit ≥2 unique domains on questions tagged for it |
| `avg_gold_fact_coverage` | **0.708** | named entities from gold appeared in answers |
| `avg_unique_domains_per_answer` | **2.2** | source diversity |
| `avg_latency_ms` | **13,764** | ~14s end-to-end per turn |

#### By category

| Category | n | relevance | faithfulness | citation_prec | refusal | conflict |
|---|---|---|---|---|---|---|
| `factual_single_hop` | 3 | **1.0** | **1.0** | **1.0** | - | - |
| `multi_hop` | 3 | 0.933 | 0.833 | 0.777 | - | - |
| `comparison` | 3 | **1.0** | 0.8 | 0.8 | - | - |
| `insufficient_evidence` | 3 | 0.0* | - | - | **0.875** | - |
| `conflicting_sources` | 3 | 0.333 | 0.71 | - | - | 0.333 |

*the relevance judge gives near-zero to "I cannot answer" responses by design - the answer doesn't *address* the question even though refusing is correct behavior. The refusal score on the same row (0.875) is the metric that actually evaluates these turns.

### Key findings

- **Factual single-hop is perfect** (1.0 / 1.0 / 1.0 across relevance, faithfulness, citation precision). On well-defined factual questions the entire pipeline lands cleanly.
- **Refusal works (0.875)** after the judge-scoring fix (see below). All 3 insufficient-evidence questions produced answers that explicitly admitted uncertainty.
- **Multi-source diversity is 1.0** - every question tagged for ≥2 unique domains hit that target. Confirms the selector's `max_per_domain=2` cap is doing real work.
- **Comparison faithfulness improved from 0.546 → 0.8** between the first and second eval runs after I tightened the answer system prompt to be more explicit about snippet-only grounding. Real measured improvement from a single prompt change.
- **Conflict detection is the weak spot (0.333)** - only 1 of 3 conflict questions cleanly flagged disagreement on this run (Mumbai population). Mt Everest and ChatGPT carbon got consensus-leaning snippets from Tavily and the agent didn't surface a real range to disagree on. Search-side limitation, not agent-logic.
- **Citation precision 0.876** (vs 0.99 on the previous run) reflects real run-to-run variance: Tavily returns different snippets each call, and on this run the multi-hop and comparison answers occasionally cited the right family of snippets but with a slightly mismatched claim-to-snippet boundary. The post-processing guarantee (no hallucinated URLs) still holds; the LLM only emits numeric markers.
- **Judge-scoring fix had a measurable effect** on this run. The first eval recorded score=0 whenever the judge LLM returned unparseable JSON, conflating "judge crashed" with "agent did terribly". Fixing to record `null` and skip in aggregates (`eval/judge.py`) lifted relevance from 0.64 → 0.677 and refusal from 0.5 → 0.875 without changing the agent at all. The improvement was the measurement, not the model.

### Qualitative observations (out-of-band testing)

Beyond the 15-question scored eval, I ran a small targeted qualitative probe across multi-hop, conflict, refusal, and multi-turn behaviors. Each query was run in a fresh session and the answer + sources observed manually.

| Query | Tests | Result |
|---|---|---|
| *"Who founded Anthropic and what were their roles at OpenAI before?"* | Multi-hop + planner reformulation | **Partial.** Got founders (Dario & Daniela Amodei) but missed their OpenAI roles - Tavily returned tangential LinkedIn posts about ex-OpenAI staff. The agent cited a snippet about Jan Leike when answering the role question - a faithfulness wobble (cited because the system prompt demands citations, but the snippet didn't answer the question). |
| *"What is the population of Tokyo?"* | Conflict handling | **Strong.** Explicitly flagged city vs metro definitional difference, cited 3 different domains (worldpopulationreview, nippon.com, macrotrends), gave both numbers. Textbook conflict handling per spec. |
| *"What will Anthropic's revenue be in 2028?"* | Refusal on insufficient evidence | **Strong.** "Snippets do not contain enough information to answer confidently", noted what IS available (the $30B 2026 target), suggested follow-up research. |
| *"Who is the current CEO of OpenAI?" → "When did they take over?"* | Multi-turn context resolution | **Partial.** Pronoun "they" was correctly resolved to Sam Altman (no clarification asked). But the agent refused to answer the date question even though Altman's 2019 start is widely documented - Tavily returned career articles, no snippet stated the specific date. |

**Three cross-cutting findings (and what I did with each):**

1. **Found and shipped a fix: duplicate citations from one URL.** When trafilatura extracted one page and the selector chunked it into 2-3 snippets, all chunks could appear as separate citations in the Sources panel pointing to the same URL. Fixed by URL-deduping the citation list after the marker-to-metadata mapping (`agent/citations.py`). Inline markers in the answer text are preserved; only the displayed Sources list is deduped.

2. **Over-grounding pattern (3 cases observed).** The system intentionally biases toward strict snippet-grounding. On widely-documented public facts where Tavily returned tangential snippets (Modi's wife, Altman's 2019 start, founders' prior OpenAI roles), the agent refused or partialed. This is the safe failure mode (no hallucination) but reduces usefulness on common-knowledge questions. Documented under Limitations with the proposed v2 fix.

3. **Session contamination of the planner.** When a session had accumulated unusual prior content (several hallucination-trick questions), the planner produced off-target queries for an unrelated next question. Asking the same question in a fresh session worked correctly. Documented under Limitations.

### Multi-turn evidence

Multi-turn is now in the harness: two questions carry a `follow_up`, and
`run_eval.py` runs it in the **same session** so context carry-over is what is being
scored. The published numbers above predate those questions. Multi-turn behaviour is
additionally evidenced by:
- The qualitative table above (Sam Altman → "when did they take over?" - pronoun resolved correctly)
- The Sarvam-AI example conversation in the "Example conversations" section earlier in this README (multi-turn "Have they raised funding?" correctly inferred Sarvam from prior turn)
- The agent loop's documented context-passing path (`agent/planner.py` receives last 2 prior turns; full conversation is available to the answer LLM via `agent/context_builder.py`)

Adding multi-turn sequences to the automated harness is listed under Future Improvements.

---

## Limitations

- **Over-grounding on common facts when search misses.** The system intentionally biases toward strict snippet-grounding to prevent URL hallucination. A consequence found during my own testing: on widely-documented public facts where Tavily returns tangential results (e.g., asking about a public figure's spouse after a session of hallucination-trick questions), the agent refuses rather than answering from prior knowledge. The clean v2 fix is a dual-mode answer that says *"no snippet evidence found, but this is widely-documented public information…"* - preserving the URL-hallucination guarantee while restoring usefulness on common-knowledge questions.
- **Session contamination of the planner.** The planner sees the last 2 turns' Q+A to resolve pronouns. When a session has accumulated unusual prior content (e.g., several hallucination-trick questions), the planner can produce off-target search queries for an unrelated next question. A fresh session usually answers the same question correctly. Mitigation would be a planner that classifies whether the new question is topically related to recent turns before pulling them in as context.
- **Recency signal is metadata-only.** The selector now uses a gentle recency multiplier (`score = cosine * (0.7 + 0.3 * recency_factor)`) where `recency_factor` decays linearly from 1.0 today to 0.3 floor at ~2 years. Pages with no extractable publish date get a neutral 0.5. The publish date is pulled from trafilatura's metadata, which is missing on ~20-30% of real pages. A v2 improvement would add a URL-pattern fallback (e.g., `/2024/07/...`) and a body-text date regex as a last resort. The current implementation is gentle by design: a strong-cosine old article still beats a weak-cosine fresh one.
- **Multi-turn is measured but not yet reported.** Two `multi_turn` questions were added to the dataset and `run_eval.py` runs each follow-up in the same session, which closes the gap this section previously described. The headline table above still comes from the earlier 15-question run, so the category has no published score yet.
- **Single search provider** (Tavily). No automatic fallback to Serper or Brave on 429.
- **No retrieval cache.** Repeated queries pay full search + fetch cost.
- **Rolling summary is single-shot.** Long sessions could compound paraphrase loss; a tiered summary would help.
- **No per-domain authority weighting.** A Reddit comment and a peer-reviewed paper get equal ranking weight pre-citation.
- **Latency is dominated by serial page fetch.** Parallel fetch helps but a slow URL still stalls its slot.
- **Eval dataset is small (17)** and Western/English biased. Production deployment would need a domain-specific eval set and adversarial cases (prompt injection in fetched pages).

## Future improvements

Beyond the two highlighted in the design note:

- **Recency weighting in the selector.** Boost snippets whose source page has a recent `last-modified` header or a date in the URL. Important for fast-moving topics.
- **Domain whitelist / blacklist per-session.** Lets a legal-research session restrict to .gov/.edu; lets a market-research session block social media.
- **Cost tracking.** Tokens in/out per turn already loggable from the LLM client; surface in the UI for cost-awareness.
- **Tool-use mode toggle.** Keep the deterministic pipeline as default but expose an opt-in mode where the answer LLM can issue follow-up searches mid-generation for hardest questions.

---

## Submission checklist

- Working Streamlit UI: `streamlit run ui/app.py`
- Working CLI: `python cli.py`
- Web research via Tavily
- Persistent sessions with conversation + turn history (SQLite)
- Citation-grounded answers (URL/title/domain rendered in UI Sources expander)
- Streaming intermediate updates (5 stages + per-token answer)
- Evaluation harness (`python eval/run_eval.py`)
- This README with design note, setup, examples, methodology, findings, limitations

## Assumptions

- Free-tier API limits are sufficient for demo: Groq (~30 req/min), Gemini (15 req/min flash), Tavily (1000 req/mo). The Groq model is set by `GROQ_MODEL` rather than hardcoded, because Groq retires models: the original `llama-3.3-70b-versatile` now 404s and the default is `openai/gpt-oss-120b`.
- Today's "correct answer" for any web-research question may not be tomorrow's. The eval methodology accommodates this by using LLM-judges over evidence rather than gold-string matching.
- The agent is built for English queries on the public web. Localization, paywalled content, and authenticated sources are out of scope for this submission.
