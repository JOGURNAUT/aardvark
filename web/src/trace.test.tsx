import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { FetchNote, WeakEvidence, reduce } from "./App";
import type { RunState } from "./App";
import type { AgentEvent, FetchedPage, Snippet } from "./types";

// Written against one real run. A search for a person returned 20 results, 6
// were opened, 3 of those refused, and the answer said nothing was found. The
// trace showed "20 unique results", "6 pages", and the sentence
// "0 of 6 pages were dry digs". Every number was correct and none of them said
// the useful thing.

const base = (over: Partial<RunState> = {}): RunState => ({
  stages: { plan: "done", search: "done", fetch: "done", select: "idle", answer: "idle" },
  plan: [],
  strategy: "",
  searchResults: 20,
  searchQueries: [],
  pages: [],
  considered: 20,
  skipped: [],
  termCoverage: [],
  pagesFetched: 0,
  snippets: [],
  answer: "",
  citations: [],
  latencyMs: null,
  stageLatencies: {},
  errors: [],
  unknownEvents: 0,
  ...over,
});

const page = (over: Partial<FetchedPage> = {}): FetchedPage => ({
  url: "https://example.com/a",
  title: "",
  domain: "example.com",
  ok: true,
  chars: 1200,
  ...over,
});

const render = (r: RunState) => {
  const dry = r.pages.filter((p: FetchedPage) => p.ok !== false && p.chars === 0);
  const failed = r.pages.filter((p: FetchedPage) => p.ok === false);
  return renderToStaticMarkup(<FetchNote run={r} dry={dry} failed={failed} />);
};

// --------------------------------------------------------------- the bug

describe("FetchNote", () => {
  it("never announces a count of zero", () => {
    // The exact sentence that shipped: "0 of 6 pages were dry digs". The old
    // note rendered on any failure but always led with the dry count.
    const html = render(base({
      considered: 6,
      pages: [
        page({ domain: "a.com" }),
        page({ domain: "in.linkedin.com", ok: false, chars: 0, rank: 2 }),
      ],
    }));
    expect(html).not.toContain("0 page");
    expect(html).not.toMatch(/\b0 of\b/);
    expect(html).not.toContain("came back dry");
  });

  it("renders nothing at all when nothing went wrong", () => {
    const html = render(base({ considered: 2, pages: [page(), page()] }));
    expect(html).toBe("");
  });
});

// ------------------------------------------------------- blocked sources

describe("refusals", () => {
  it("names the blocked domain, its rank, and links to it", () => {
    // The user's complaint: the LinkedIn profile exists, the tool said nothing
    // was found. The honest output names the block and hands over the URL.
    const html = render(base({
      considered: 6,
      pages: [
        page(),
        page({ domain: "in.linkedin.com", url: "https://in.linkedin.com/in/x",
               ok: false, chars: 0, rank: 2 }),
      ],
    }));
    expect(html).toContain("in.linkedin.com");
    expect(html).toContain("#2");
    expect(html).toContain('href="https://in.linkedin.com/in/x"');
    expect(html).toContain("refused the request");
  });

  it("calls out a refusal from the top of the ranking", () => {
    const html = render(base({
      considered: 6,
      pages: [page(), page({ domain: "x.com", ok: false, chars: 0, rank: 1 })],
    }));
    expect(html).toContain("search result #1");
  });

  it("does not make that claim for a refusal from far down the ranking", () => {
    // A 403 from result #17 is not evidence that the answer was behind it.
    const html = render(base({
      considered: 20,
      pages: [page(), page({ domain: "x.com", ok: false, chars: 0, rank: 17 })],
    }));
    expect(html).toContain("refused the request");
    expect(html).not.toContain("search result #17");
  });
});

// ------------------------------------------------------ unopened results

describe("results never opened", () => {
  it("reports the ones the page cap left behind", () => {
    // 20 results, 6 opened. The old trace mentioned only the 6, so the run
    // looked like the search had found nothing more.
    const html = render(base({
      considered: 20,
      pages: [page(), page({ ok: false, chars: 0 })],
      skipped: [{ url: "u", domain: "d.com", rank: 7, reason: "page cap reached" }],
    }));
    expect(html).toContain("18 more results were never opened");
    expect(html).toContain("page cap");
  });

  it("mentions the domain rule when that is what applied", () => {
    const html = render(base({
      considered: 10,
      pages: [page({ ok: false, chars: 0 })],
      skipped: [{ url: "u", domain: "d.com", rank: 3,
                  reason: "2 already picked from this domain" }],
    }));
    expect(html).toContain("two-per-domain");
  });
});

// -------------------------------------------------------- term coverage

describe("term coverage", () => {
  it("says when the thing asked about appears nowhere", () => {
    // The whole point. Every page matched the surname; none matched the name.
    const html = render(base({
      considered: 6,
      pages: [page(), page({ domain: "b.com" }), page({ domain: "c.com" })],
      termCoverage: [
        { term: "somanorani", pages: 0, hits: 0, of_pages: 3 },
        { term: "ningthoujam", pages: 3, hits: 14, of_pages: 3 },
      ],
    }));
    expect(html).toContain("somanorani");
    expect(html).toContain("appears nowhere");
    expect(html).not.toContain('"ningthoujam" appears nowhere');
  });

  it("stays quiet when nothing could be read, since absence proves nothing", () => {
    const html = render(base({
      considered: 3,
      pages: [page({ ok: false, chars: 0 })],
      termCoverage: [{ term: "somanorani", pages: 0, hits: 0, of_pages: 0 }],
    }));
    expect(html).not.toContain("appears nowhere");
  });
});

// ------------------------------------------------------- weak evidence

describe("WeakEvidence", () => {
  const snips = (...scores: number[]): Snippet[] =>
    scores.map((score) => ({ title: "t", domain: "d.com", url: "u", score }));

  it("warns when the best match is poor", () => {
    // The selector has no score floor, so a full-looking evidence list can be
    // built entirely from near-noise.
    const html = renderToStaticMarkup(<WeakEvidence snippets={snips(0.25, 0.22, 0.07)} />);
    expect(html).toContain("0.25");
    expect(html).toContain("weak evidence");
  });

  it("stays quiet when the evidence is good", () => {
    expect(renderToStaticMarkup(<WeakEvidence snippets={snips(0.82, 0.4)} />)).toBe("");
  });

  it("judges on the best snippet, not the average", () => {
    // One strong match plus three weak ones is a usable answer.
    expect(renderToStaticMarkup(<WeakEvidence snippets={snips(0.9, 0.05, 0.05, 0.05)} />)).toBe("");
  });

  it("stays quiet when nothing carries a score", () => {
    const unscored = [{ title: "t", domain: "d", url: "u" }] as Snippet[];
    expect(renderToStaticMarkup(<WeakEvidence snippets={unscored} />)).toBe("");
  });
});

// ------------------------------------------------------------- reducer

describe("reduce keeps the new trace fields", () => {
  const empty = base({ searchResults: 0, considered: 0 });

  it("keeps per-query search stats", () => {
    const ev: AgentEvent = {
      type: "search_done",
      results: [{ url: "a", title: "" }],
      queries: [
        { query: "biography", results: 8, new: 8 },
        { query: "social media", results: 0, new: 0 },
      ],
    };
    const r = reduce(empty, ev);
    expect(r.searchQueries).toHaveLength(2);
    expect(r.searchQueries[1]).toEqual({ query: "social media", results: 0, new: 0 });
  });

  it("keeps what the fetch stage left behind", () => {
    const ev: AgentEvent = {
      type: "fetch_done",
      pages: [page()],
      considered: 20,
      skipped: [{ url: "u", domain: "d", rank: 7, reason: "page cap reached" }],
      term_coverage: [{ term: "x", pages: 0, hits: 0, of_pages: 1 }],
    };
    const r = reduce(empty, ev);
    expect(r.considered).toBe(20);
    expect(r.skipped).toHaveLength(1);
    expect(r.termCoverage[0].term).toBe("x");
  });

  it("falls back to the page count when the server sends no `considered`", () => {
    // The field is new; an older server must not make the UI claim that 0
    // results were considered and then report a negative number unopened.
    const r = reduce(empty, { type: "fetch_done", pages: [page(), page()] } as AgentEvent);
    expect(r.considered).toBe(2);
    expect(r.skipped).toEqual([]);
  });
});

// -------------------------------------------------------------- digging

describe("dig", () => {
  const withUnopened = () => base({
    considered: 20,
    pages: [page(), page({ ok: false, chars: 0 })],
    skipped: [{ url: "u", domain: "d.com", rank: 7, reason: "page cap reached" }],
  });

  it("offers a dig only when something was left unopened", () => {
    const r = withUnopened();
    const dry = r.pages.filter((p: FetchedPage) => p.ok !== false && p.chars === 0);
    const failed = r.pages.filter((p: FetchedPage) => p.ok === false);
    const html = renderToStaticMarkup(
      <FetchNote run={r} dry={dry} failed={failed} onDig={() => {}} />,
    );
    expect(html).toContain("Dig deeper");
  });

  it("caps the offer at one batch rather than promising all 18", () => {
    // 18 are unopened; a dig opens at most MAX_PAGES_TO_FETCH of them.
    const r = withUnopened();
    const html = renderToStaticMarkup(
      <FetchNote run={r} dry={[]} failed={[]} onDig={() => {}} />,
    );
    expect(html).toContain("open 6 more");
    expect(html).not.toContain("open 18 more");
  });

  it("shows no button when the caller offers no handler", () => {
    const html = renderToStaticMarkup(<FetchNote run={withUnopened()} dry={[]} failed={[]} />);
    expect(html).toContain("never opened");
    expect(html).not.toContain("Dig deeper");
  });

  it("disables the button while a dig is running", () => {
    const html = renderToStaticMarkup(
      <FetchNote run={withUnopened()} dry={[]} failed={[]} onDig={() => {}} digging />,
    );
    expect(html).toContain("disabled");
    expect(html).toContain("Digging");
  });

  it("restarts the stage machine at fetch, since a dig does not re-plan", () => {
    // The results were already ranked and stored by the turn being continued,
    // so plan and search are done before this starts.
    const r = reduce(base(), { type: "dig_start", query: "q", opening: 6, remaining_after: 8 });
    expect(r.stages.plan).toBe("done");
    expect(r.stages.search).toBe("done");
    expect(r.stages.fetch).toBe("running");
    expect(r.stages.answer).toBe("idle");
  });

  it("clears the previous turn's pages so the trace is not double-counted", () => {
    const started = base({ pages: [page(), page()], answer: "old answer" });
    const r = reduce(started, { type: "dig_start", query: "q", opening: 6, remaining_after: 8 });
    expect(r.pages).toEqual([]);
    expect(r.answer).toBe("");
  });
});
