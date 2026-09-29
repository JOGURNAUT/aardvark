import { useCallback, useEffect, useRef, useState } from "react";
import { ask, createSession } from "./api";
import { ErrorBoundary } from "./ErrorBoundary";
import { STAGES } from "./types";
import type {
  AgentEvent, Citation, FetchedPage, QueryStat, Snippet,
  SkippedResult, Stage, TermCoverage,
} from "./types";

type StageState = "idle" | "running" | "done";

export interface RunState {
  stages: Record<Stage, StageState>;
  plan: string[];
  strategy: string;
  searchResults: number;
  searchQueries: QueryStat[];
  pages: FetchedPage[];
  considered: number;
  skipped: SkippedResult[];
  termCoverage: TermCoverage[];
  pagesFetched: number;
  snippets: Snippet[];
  answer: string;
  citations: Citation[];
  latencyMs: number | null;
  stageLatencies: Record<string, number>;
  errors: { stage: string; message: string }[];
  unknownEvents: number;
}

const emptyRun = (): RunState => ({
  stages: { plan: "idle", search: "idle", fetch: "idle", select: "idle", answer: "idle" },
  plan: [],
  strategy: "",
  searchResults: 0,
  searchQueries: [],
  pages: [],
  considered: 0,
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
});

// The stage names are the loop's; the second line is the design's. Both are
// shown because "fetch" says what the code is doing and "hauling logs" says
// how long to expect to wait.
const NICK: Record<Stage, string> = {
  plan: "sniffing the ground",
  search: "finding the mounds",
  fetch: "digging in",
  select: "sifting the soil",
  answer: "bringing it up",
};

const EXAMPLES = [
  {
    q: "How did the EU AI Act's definition of a high-risk system change between the 2021 proposal and the final text?",
    why: "multi-hop · needs two documents",
  },
  {
    q: "What was the exact internal headcount of OpenAI's alignment team in March 2023?",
    why: "insufficient evidence · should refuse rather than guess",
  },
  {
    q: "How many people died in the 1970 Ancash earthquake?",
    why: "sources disagree · should say so and cite both",
  },
];

// A dry dig: a page that loaded and gave up nothing. The server sends ok and
// chars separately, so this is derived rather than reported, and it is the
// single most useful line in the trace, because it is the usual reason an
// answer comes out thin.
type PageKind = "ok" | "dry" | "failed";

function kindOf(p: FetchedPage): PageKind {
  if (p.ok === false) return "failed";
  if (p.chars != null && p.chars === 0) return "dry";
  return "ok";
}

const BAR_MAX = 128;

function barWidth(chars: number | undefined): number {
  if (!chars || chars <= 0) return 0;
  // Log scale: a 40k-character page should not make a 2k one invisible.
  return Math.max(3, Math.round((Math.log10(chars) / Math.log10(20000)) * BAR_MAX));
}

function splitUrl(p: FetchedPage): { host: string; path: string } {
  try {
    const u = new URL(p.url);
    return { host: p.domain || u.hostname.replace(/^www\./, ""), path: u.pathname + u.search };
  } catch {
    return { host: p.domain || p.url, path: "" };
  }
}

// Five strata with a burrow cutting down through them, the last two greying
// out while a run is live so the tunnel reads as still being dug. The burrow
// is stroked in the background colour, so it punches through the lines rather
// than drawing over them.
const Logo = ({ size = 30, live = false }: { size?: number; live?: boolean }) => (
  <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true">
    <line x1="4" y1="6" x2="44" y2="6" stroke="#C6E15B" strokeWidth="3" strokeLinecap="round" />
    <line x1="4" y1="15" x2="44" y2="15" stroke="#C6E15B" strokeWidth="3" strokeLinecap="round" />
    <line className={live ? "bv-pulse" : undefined}
          x1="4" y1="24" x2="44" y2="24" stroke="#C6E15B" strokeWidth="3" strokeLinecap="round" />
    <line x1="4" y1="33" x2="44" y2="33" stroke={live ? "#3A3734" : "#C6E15B"}
          strokeWidth="3" strokeLinecap="round" />
    <line x1="4" y1="42" x2="44" y2="42" stroke={live ? "#3A3734" : "#C6E15B"}
          strokeWidth="3" strokeLinecap="round" />
    <path d="M15 0 C15 18, 34 16, 31 39" stroke="#121110" strokeWidth="7" strokeLinecap="round" />
    <circle cx="31" cy="40" r="4.5" fill="#C6E15B" />
  </svg>
);

/**
 * Renders [1] and [2,3] markers as links into the evidence list.
 *
 * The server already strips markers that point at nothing, so any marker here
 * is known to resolve. One that does not renders in the failure colour rather
 * than as a dead link, because a broken citation should look broken.
 */
function AnswerText({ text, citations, streaming }:
                    { text: string; citations: Citation[]; streaming: boolean }) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);

  return (
    <div className={streaming ? "answer streaming" : "answer"}>
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
        if (!m) return <span key={i}>{part}</span>;
        return (
          <span key={i}>
            {m[1].split(",").map((raw) => {
              const n = parseInt(raw.trim(), 10);
              const c = byMarker.get(n);
              return c ? (
                <a key={n} className="cite" href={`#ev-${n}`} title={`${c.title} - ${c.domain}`}>
                  {n}
                </a>
              ) : (
                <span key={n} className="cite dead">{n}</span>
              );
            })}
          </span>
        );
      })}
    </div>
  );
}

/** The evidence list. Exported so it can be render-tested against a payload
 *  shaped like the wire, which is where the black-screen crash lived: the
 *  reducer never touched the missing field, only the render did. */
export function Evidence({ snippets }: { snippets: Snippet[] }) {
  const domains = new Set(snippets.map((s) => s.domain).filter(Boolean));
  return (
    <section className="evidence" aria-labelledby="evidence-h">
      <div className="evidence-head">
        <h2 id="evidence-h">Evidence</h2>
        <span className="mono">
          {snippets.length} snippets · {domains.size} domains
        </span>
      </div>
      <ol>
        {snippets.map((s, i) => {
          const n = s.idx ?? i + 1;
          return (
            <li key={i} id={`ev-${n}`}>
              <span className="n">{n}</span>
              <div className="body">
                <div className="meta">
                  <span>{s.domain}</span>
                  {s.score != null && (
                    <span className="score">
                      <span className="bar">
                        <i style={{ width: `${Math.round(s.score * 64)}px` }} />
                      </span>
                      <b>{s.score.toFixed(2)}</b>
                    </span>
                  )}
                  {s.chars != null && <span>{s.chars.toLocaleString("en-US")} chars</span>}
                </div>
                <a className="title" href={s.url} target="_blank" rel="noreferrer">
                  {s.title || s.url}
                </a>
                {s.text && <p className="snip">{s.text.slice(0, 260)}</p>}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [asked, setAsked] = useState("");
  const [run, setRun] = useState<RunState>(emptyRun);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [bootError, setBootError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    createSession("web").then(setSessionId).catch((e) => setBootError(String(e)));
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  // A research turn takes tens of seconds and fetch is the long silent one.
  // Without a clock the page looks hung exactly where it is working hardest.
  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 250);
    return () => clearInterval(id);
  }, [busy]);

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const submit = useCallback(async (override?: string) => {
    const q = (override ?? query).trim();
    if (!sessionId || !q || busy) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setStarted(true);
    setAsked(q);
    setElapsed(0);
    setRun({ ...emptyRun(), stages: { ...emptyRun().stages, plan: "running" } });

    try {
      for await (const ev of ask(sessionId, q, controller.signal)) {
        setRun((r) => reduce(r, ev));
      }
    } catch (e) {
      const stopped = e instanceof DOMException && e.name === "AbortError";
      setRun((r) => ({
        ...r,
        errors: [...r.errors, { stage: "client", message: stopped ? "stopped" : String(e) }],
      }));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [sessionId, query, busy]);

  const reset = () => {
    abortRef.current?.abort();
    setStarted(false);
    setAsked("");
    setQuery("");
    setRun(emptyRun());
  };

  const activeStage = STAGES.find((s) => run.stages[s] === "running");

  return (
    <div className="app">
      <div className="col">
        <header className="bar">
          <span className="brand">
            <Logo live={busy} />
            <span className="word">Aardvark</span>
          </span>
          {busy ? (
            <button className="pill-ghost" onClick={stop}>Stop</button>
          ) : started ? (
            <button className="pill-ghost" onClick={reset}>New question</button>
          ) : (
            <span className="mono">deep research agent</span>
          )}
        </header>

        {bootError && (
          <div className="error">
            <span className="stage">session</span>
            <span className="msg">{bootError}</span>
          </div>
        )}

        {!started ? (
          <main className="ask">
            <h1 className="hero">
              One question.
              <br />
              <em>Dug all the way down.</em>
            </h1>

            <div className="field">
              <label className="eyebrow" htmlFor="q">Your question</label>
              <textarea
                id="q"
                rows={3}
                value={query}
                placeholder="What do you want to know? Ask it the way you'd ask a librarian."
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void submit();
                }}
                disabled={!sessionId}
              />
              <div className="field-foot">
                <span className="mono">plan → search → fetch → select → answer · 20–60 s</span>
                <button className="pill" onClick={() => void submit()}
                        disabled={!sessionId || !query.trim()}>
                  Research
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                    <path d="M2 7 H12 M8 3 L12 7 L8 11" stroke="#121110" strokeWidth="1.8" />
                  </svg>
                </button>
              </div>
            </div>

            <p className="standfirst">
              Aardvark shows every query it runs and every page it reads, including the dry
              digs that give up no text, so you can see what the answer stands on.
            </p>

            <div className="suggest">
              {EXAMPLES.map((ex) => (
                <button key={ex.q} onClick={() => void submit(ex.q)} disabled={!sessionId}>
                  {ex.q}
                  <span className="why">{ex.why}</span>
                </button>
              ))}
            </div>
          </main>
        ) : (
          <main>
            <section className="question">
              <div className="eyebrow">Question</div>
              <h1>{asked}</h1>
              <div className="mono">
                {run.latencyMs !== null
                  ? `Answered in ${(run.latencyMs / 1000).toFixed(1)} s · ${run.citations.length} sources cited`
                  : `${elapsed} s elapsed · usually 20–60 s`}
              </div>
            </section>

            {busy && activeStage && (
              <div className="status">
                <Logo size={56} live />
                <div>
                  <div className="now">
                    <em>{NICK[activeStage]}</em>
                    {activeStage === "fetch" && run.pagesFetched > 0
                      ? ` — reading ${run.pagesFetched} pages`
                      : ""}
                  </div>
                  <div className="sub">
                    Stage {STAGES.indexOf(activeStage) + 1} of 5 · {elapsed} s elapsed
                  </div>
                </div>
              </div>
            )}

            <ErrorBoundary label="trace">
              <Trace run={run} />
            </ErrorBoundary>

            {run.errors.map((e, i) => (
              <div className="error" key={i}>
                <span className="stage">{e.stage}</span>
                <span className="msg">{e.message}</span>
              </div>
            ))}

            {run.answer && (
              <ErrorBoundary label="answer">
                <WeakEvidence snippets={run.snippets} />
                <section aria-labelledby="answer-h">
                  <div className="answer-head">
                    <h2 id="answer-h">Answer</h2>
                    <button className="textbtn"
                            onClick={() => void navigator.clipboard?.writeText(run.answer)}>
                      Copy
                    </button>
                  </div>
                  <AnswerText text={run.answer} citations={run.citations}
                              streaming={busy && run.stages.answer === "running"} />
                </section>
              </ErrorBoundary>
            )}

            {run.snippets.length > 0 && (
              <ErrorBoundary label="evidence">
                <Evidence snippets={run.snippets} />
              </ErrorBoundary>
            )}

            {run.latencyMs !== null && (
              <footer className="note-foot">
                Aardvark builds each answer only from the snippets listed above.
                {Object.entries(run.stageLatencies).map(([s, secs]) => (
                  <span key={s}> · {s} {Math.round(secs * 1000)} ms</span>
                ))}
                {run.unknownEvents > 0 && ` · ${run.unknownEvents} unrecognised events`}
              </footer>
            )}
          </main>
        )}
      </div>
    </div>
  );
}

/**
 * The five stages as a timeline, each filling in with what it actually found.
 *
 * The bead column is the only progress indicator; everything to the right of
 * it is evidence. A page that returned 200 and no extractable text shows as
 * "hollow", which is derived here rather than reported, because the server
 * sends `ok` and `chars` separately and neither alone says it.
 */
/**
 * What the fetch stage could not do, in the order it matters.
 *
 * The version this replaces led with the count of dry pages and rendered
 * whenever anything had gone wrong, so a run with three refusals and no dry
 * pages announced "0 of 6 pages were dry digs". It also never mentioned the
 * results that were never opened at all, which on the run that prompted this
 * was fourteen of twenty.
 *
 * Every sentence is emitted only when its own count is non-zero, and the
 * refusals carry their search rank, because a refusal from the top result and
 * one from the twentieth are not the same event.
 */
export function FetchNote({ run, dry, failed }:
                   { run: RunState; dry: FetchedPage[]; failed: FetchedPage[] }) {
  const readable = run.pages.length - dry.length - failed.length;
  const unopened = Math.max(0, run.considered - run.pages.length);
  const topBlocked = [...failed].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))[0];
  const missing = run.termCoverage.filter((t) => t.pages === 0 && t.of_pages > 0);

  if (!dry.length && !failed.length && !unopened && !missing.length) return null;

  return (
    <div className="note">
      <svg width="16" height="16" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="9" r="7" stroke="#C6E15B" strokeWidth="2" />
      </svg>
      <div className="note-body">
        {missing.length > 0 && (
          <p>
            <strong>
              {missing.map((t) => `"${t.term}"`).join(" and ")} appears nowhere in
              the {missing[0].of_pages} page{missing[0].of_pages === 1 ? "" : "s"} that
              could be read.
            </strong>{" "}
            Everything below matched on the other words only, so read it as a
            near miss rather than an answer.
          </p>
        )}

        {failed.length > 0 && (
          <p>
            <strong>
              {failed.length} page{failed.length === 1 ? "" : "s"} refused the request
            </strong>
            {": "}
            {failed.map((p, i) => (
              <span key={p.url}>
                {i > 0 && ", "}
                <a href={p.url} target="_blank" rel="noreferrer">{p.domain || p.url}</a>
                {p.rank != null && <span className="rank"> #{p.rank}</span>}
              </span>
            ))}
            {topBlocked?.rank != null && topBlocked.rank <= 3 && (
              <>
                {" "}
                <strong>One was search result #{topBlocked.rank}</strong>, so the
                page most likely to hold the answer is the one that could not be
                read. The link above opens it directly.
              </>
            )}
          </p>
        )}

        {dry.length > 0 && (
          <p>
            <strong>{dry.length} page{dry.length === 1 ? "" : "s"} came back dry</strong>
            {" — loaded, but gave up no readable text: "}
            {dry.map((p) => p.domain || p.url).join(", ")}.
          </p>
        )}

        {unopened > 0 && (
          <p>
            <strong>{unopened} more results were never opened.</strong>{" "}
            {run.skipped.some((k) => k.reason.includes("domain"))
              ? "Some hit the two-per-domain rule; the rest hit the page cap."
              : "They were past the page cap."}
          </p>
        )}

        <p className="rests">
          This answer rests on {readable} page{readable === 1 ? "" : "s"}.
        </p>
      </div>
    </div>
  );
}

// Below this, the best thing retrieved is not a match for the question, it is
// merely the least bad thing available.
const WEAK = 0.35;

/**
 * Says out loud when the evidence is thin.
 *
 * The selector has no score floor: it keeps the top N whatever they scored. So
 * a question with no good match still produces a full-looking evidence list,
 * and the page reads exactly as confidently whether the best snippet scored
 * 0.9 or 0.07.
 */
export function WeakEvidence({ snippets }: { snippets: Snippet[] }) {
  const scored = snippets.map((s) => s.score).filter((x): x is number => x != null);
  if (!scored.length) return null;
  const top = Math.max(...scored);
  if (top >= WEAK) return null;

  return (
    <div className="note weak">
      <svg width="16" height="16" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path d="M9 3 L16 15 H2 Z" stroke="#F07A93" strokeWidth="1.8" />
      </svg>
      <div className="note-body">
        <p>
          <strong>Best snippet scored {top.toFixed(2)}.</strong> Nothing retrieved
          is a close match for the question, so this answer is built on weak
          evidence. The selector keeps the best it has; it does not require the
          best to be good.
        </p>
      </div>
    </div>
  );
}

function Trace({ run }: { run: RunState }) {
  const dry = run.pages.filter((p) => kindOf(p) === "dry");
  const failed = run.pages.filter((p) => kindOf(p) === "failed");

  return (
    <ol className="steps">
      {STAGES.map((s, i) => {
        const state = run.stages[s];
        const secs = run.stageLatencies[s];
        return (
          <li className={`step ${state}`} key={s}>
            <div className="rail">
              <span className="bead">
                {state === "done" && (
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
                    <path d="M3 7.2 L6 10 L11 4" stroke="#121110" strokeWidth="2" />
                  </svg>
                )}
                {state === "running" && (
                  <svg className="bv-spin" width="14" height="14" viewBox="0 0 14 14"
                       fill="none" aria-hidden="true">
                    <path d="M7 1.5 A5.5 5.5 0 0 1 12.5 7" stroke="#C6E15B" strokeWidth="2" />
                  </svg>
                )}
              </span>
              {i < STAGES.length - 1 && <span className="wire" />}
            </div>

            <div className="step-body">
              <div className="step-head">
                <span className="left">
                  <span className="name">{s}</span>
                  <span className="nick">{NICK[s]}</span>
                </span>
                <span className="t">
                  {secs != null
                    ? `${secs.toFixed(1)} s`
                    : state === "running"
                      ? "working"
                      : state === "idle"
                        ? "waiting"
                        : ""}
                </span>
              </div>

              {s === "plan" && run.plan.length > 0 && (
                <ol className="subqs">
                  {run.plan.map((q) => <li key={q}>{q}</li>)}
                </ol>
              )}

              {s === "search" && run.searchQueries.length > 0 && (
                <div className="qtable">
                  {run.searchQueries.map((q) => (
                    <div className={q.results === 0 ? "qline empty" : "qline"} key={q.query}>
                      <span className="q">{q.query}</span>
                      <span className="n">
                        {q.results === 0
                          ? "nothing"
                          : `${q.results} results · ${q.new} new`}
                      </span>
                    </div>
                  ))}
                  <div className="qline total">
                    <span className="q">{run.searchResults} unique after de-duplicating by URL</span>
                  </div>
                </div>
              )}

              {s === "fetch" && run.pages.length > 0 && (
                <>
                  <div className="ptable">
                    <div className="thead">
                      <span>Status</span>
                      <span>Page</span>
                      <span>Text extracted</span>
                      <span className="r">Chars</span>
                    </div>
                    {run.pages.map((p, k) => {
                      const kind = kindOf(p);
                      const { host, path } = splitUrl(p);
                      return (
                        <div className={`prow ${kind}`} key={k}>
                          <span className="st">
                            <Dot kind={kind} />
                            {kind === "failed" ? "403" : kind === "dry" ? "dry" : "200"}
                          </span>
                          <span className="where">
                            <a href={p.url} target="_blank" rel="noreferrer">{host}</a>
                            <span className="path">{path}</span>
                          </span>
                          <span className="bar">
                            <i style={{ width: `${barWidth(p.chars)}px` }} />
                          </span>
                          <span className="chars">
                            {kind === "failed"
                              ? "refused"
                              : p.chars != null
                                ? p.chars.toLocaleString("en-US")
                                : ""}
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <FetchNote run={run} dry={dry} failed={failed} />
                </>
              )}

              {s === "select" && run.snippets.length > 0 && (
                <div className="qrow">
                  <span>{run.snippets.length} snippets kept</span>
                  <span className="mono">at most 2 per domain</span>
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function Dot({ kind }: { kind: PageKind }) {
  if (kind === "failed") {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
        <path d="M1.5 1.5 L8.5 8.5 M8.5 1.5 L1.5 8.5" stroke="#F07A93" strokeWidth="1.8" />
      </svg>
    );
  }
  if (kind === "dry") {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
        <circle cx="5" cy="5" r="3.8" stroke="#C6E15B" strokeWidth="1.6" />
      </svg>
    );
  }
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
      <circle cx="5" cy="5" r="4" fill="#ADA9A0" />
    </svg>
  );
}

/** Pure reducer, so the event handling is testable without a browser. */
export function reduce(r: RunState, ev: AgentEvent): RunState {
  switch (ev.type) {
    case "plan":
      return {
        ...r,
        plan: ev.queries,
        strategy: ev.strategy,
        stages: { ...r.stages, plan: "done", search: "running" },
      };
    case "search_done":
      return {
        ...r,
        searchResults: ev.results.length,
        searchQueries: ev.queries ?? [],
        stages: { ...r.stages, search: "done", fetch: "running" },
      };
    case "fetch_done":
      return {
        ...r,
        pages: ev.pages,
        pagesFetched: ev.pages.length,
        considered: ev.considered ?? ev.pages.length,
        skipped: ev.skipped ?? [],
        termCoverage: ev.term_coverage ?? [],
        stages: { ...r.stages, fetch: "done", select: "running" },
      };
    case "select_done":
      return {
        ...r,
        snippets: ev.snippets,
        stages: { ...r.stages, select: "done", answer: "running" },
      };
    case "answer_token":
      return { ...r, answer: r.answer + ev.text };
    case "answer_done":
      return {
        ...r,
        // The server re-emits the full answer after stripping markers that point
        // at nothing, so this replaces the accumulated tokens rather than
        // appending to them. Streamed text is a preview; this is the record.
        answer: ev.answer,
        citations: ev.citations,
        latencyMs: ev.latency_ms,
        stageLatencies: ev.stage_latencies ?? {},
        stages: { ...r.stages, answer: "done" },
      };
    case "error":
      return { ...r, errors: [...r.errors, { stage: ev.stage, message: ev.message }] };
    default:
      // An event the loop added and this file has not caught up with. Counted
      // and shown, because a UI that silently drops events it does not know
      // about is how a front end and its backend drift apart unnoticed.
      return { ...r, unknownEvents: r.unknownEvents + 1 };
  }
}
