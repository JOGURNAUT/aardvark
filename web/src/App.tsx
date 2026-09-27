import { useCallback, useEffect, useRef, useState } from "react";
import { ask, createSession } from "./api";
import { ErrorBoundary } from "./ErrorBoundary";
import { STAGES } from "./types";
import type { AgentEvent, Citation, FetchedPage, Snippet, Stage } from "./types";

type StageState = "idle" | "running" | "done";

interface RunState {
  stages: Record<Stage, StageState>;
  plan: string[];
  strategy: string;
  searchResults: number;
  pages: FetchedPage[];
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
  pages: [],
  pagesFetched: 0,
  snippets: [],
  answer: "",
  citations: [],
  latencyMs: null,
  stageLatencies: {},
  errors: [],
  unknownEvents: 0,
});

// Each of these exercises a different one of the six evaluation categories, so
// the empty state doubles as the demo. The second one is the interesting one:
// the honest answer is that it cannot be answered from what is findable.
const EXAMPLES = [
  {
    q: "How did the EU AI Act's definition of a high-risk system change between the 2021 proposal and the final text?",
    why: "multi-hop, needs two documents",
  },
  {
    q: "What was the exact internal headcount of OpenAI's alignment team in March 2023?",
    why: "insufficient evidence, should refuse rather than guess",
  },
  {
    q: "How many people died in the 1970 Ancash earthquake?",
    why: "sources disagree, should say so and cite both",
  },
];

/**
 * Renders [1] and [2,3] markers as links into the snippet list.
 *
 * Deliberately done here and not in the model's output: the server already
 * strips markers that point at nothing, so any marker reaching this point is
 * known to resolve. If one does not, it renders as plain text rather than a
 * dead link, because a broken citation should look broken.
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
        const markers = m[1].split(",").map((n) => parseInt(n.trim(), 10));
        return (
          <sup key={i} className="cites">
            [
            {markers.map((n, j) => {
              const c = byMarker.get(n);
              return (
                <span key={n}>
                  {j > 0 && ","}
                  {c ? (
                    <a href={c.url} target="_blank" rel="noreferrer"
                       title={`${c.title} - ${c.domain}`}>
                      {n}
                    </a>
                  ) : (
                    <span className="dead-cite">{n}</span>
                  )}
                </span>
              );
            })}
            ]
          </sup>
        );
      })}
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [run, setRun] = useState<RunState>(emptyRun);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [bootError, setBootError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const boxRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    createSession("web").then(setSessionId).catch((e) => setBootError(String(e)));
  }, []);

  // Abort any in-flight run when the component goes away, so a navigation does
  // not leave the server streaming tokens into a closed connection.
  useEffect(() => () => abortRef.current?.abort(), []);

  // A research turn takes tens of seconds. Without a clock the page looks hung
  // during fetch, which is the longest stage and the one with nothing to show.
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

  const useExample = (q: string) => {
    setQuery(q);
    boxRef.current?.focus();
    void submit(q);
  };

  return (
    <div className="app">
      <header>
        <h1>Beaver</h1>
        <span className="sub">deep research agent</span>
        <span className="spacer" />
        {busy && <span className="sub">{elapsed}s</span>}
      </header>

      {bootError && (
        <div className="error">
          <span className="stage">session</span>
          <span className="msg">{bootError}</span>
        </div>
      )}

      <div className="composer">
        <textarea
          ref={boxRef}
          value={query}
          placeholder="Ask something that needs more than one source."
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void submit();
          }}
          disabled={!sessionId}
        />
        <div className="actions">
          <button className="btn-primary" onClick={() => void submit()}
                  disabled={busy || !sessionId || !query.trim()}>
            {busy ? "Researching" : "Ask"}
          </button>
          {busy && <button className="btn-ghost" onClick={stop}>Stop</button>}
          <span className="spacer" />
          <span className="hint">Ctrl + Enter</span>
        </div>
      </div>

      {!started && (
        <section className="examples">
          <h2>Try one</h2>
          {EXAMPLES.map((ex) => (
            <button key={ex.q} onClick={() => useExample(ex.q)} disabled={!sessionId}>
              {ex.q}
              <span className="why">{ex.why}</span>
            </button>
          ))}
        </section>
      )}

      {started && (
        <ErrorBoundary label="trace">
          <StageTrail run={run} />
          <Trace run={run} />
        </ErrorBoundary>
      )}

      {run.errors.map((e, i) => (
        <div className="error" key={i}>
          <span className="stage">{e.stage}</span>
          <span className="msg">{e.message}</span>
        </div>
      ))}

      {run.answer && (
        <ErrorBoundary label="answer">
          <AnswerText text={run.answer} citations={run.citations}
                      streaming={busy && run.stages.answer === "running"} />
        </ErrorBoundary>
      )}

      {run.latencyMs !== null && (
        <div className="latency">
          <span className="total">{(run.latencyMs / 1000).toFixed(1)}s total</span>
          {Object.entries(run.stageLatencies).map(([s, secs]) => (
            <span key={s} className="chip">{s} {Math.round(secs * 1000)}ms</span>
          ))}
          {run.unknownEvents > 0 && (
            <span className="chip warn">{run.unknownEvents} unrecognised events</span>
          )}
        </div>
      )}

      {run.snippets.length > 0 && (
        <ErrorBoundary label="evidence">
          <Evidence snippets={run.snippets} />
        </ErrorBoundary>
      )}
    </div>
  );
}

/**
 * What the agent actually did, filling in live.
 *
 * This is the part worth watching, and it is the part a progress bar throws
 * away. The five dots above say which stage is running; this says which
 * queries were chosen, which pages came back empty, and what each snippet
 * scored. A fetch that returns 0 characters is the single most common reason
 * an answer is thin, and it is invisible unless the per-page result is shown.
 */
function Trace({ run }: { run: RunState }) {
  const nothingYet = !run.strategy && run.pages.length === 0 && run.snippets.length === 0;
  if (nothingYet) return null;

  return (
    <section className="trace">
      {run.strategy && (
        <div className="trace-group">
          <span className="trace-stage">plan</span>
          <div className="trace-body">
            <p className="strategy">{run.strategy}</p>
            <div className="chips">
              {run.plan.map((q) => <span className="chip-q" key={q}>{q}</span>)}
            </div>
          </div>
        </div>
      )}

      {run.searchResults > 0 && (
        <div className="trace-group">
          <span className="trace-stage">search</span>
          <div className="trace-body">
            <p className="muted">{run.searchResults} unique results after de-duplicating by URL</p>
          </div>
        </div>
      )}

      {run.pages.length > 0 && (
        <div className="trace-group">
          <span className="trace-stage">fetch</span>
          <div className="trace-body">
            <p className="muted">
              {run.pages.filter((p) => p.ok !== false).length} of {run.pages.length} extracted
            </p>
            {run.pages.map((p, i) => (
              <div className={p.ok === false ? "row bad" : "row"} key={i}>
                <span className="mark">{p.ok === false ? "fail" : "ok"}</span>
                <span className="meta">{p.chars != null ? `${p.chars} chars` : ""}</span>
                <a href={p.url} target="_blank" rel="noreferrer">{p.domain || p.url}</a>
              </div>
            ))}
          </div>
        </div>
      )}

      {run.snippets.length > 0 && (
        <div className="trace-group">
          <span className="trace-stage">select</span>
          <div className="trace-body">
            <p className="muted">
              {run.snippets.length} snippets kept, at most 2 per domain
            </p>
            {run.snippets.map((s, i) => (
              <div className="row" key={i}>
                <span className="mark idx">[{s.idx ?? i + 1}]</span>
                <span className="meta">{s.score != null ? s.score.toFixed(3) : ""}</span>
                <span className="title">{s.title || s.url}</span>
                <span className="domain">{s.domain}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

/** The evidence list. Exported so it can be render-tested against a payload
 *  shaped like the wire, which is where the black-screen crash lived: the
 *  reducer never touched the missing field, only the render did. */
export function Evidence({ snippets }: { snippets: Snippet[] }) {
  return (
      <section className="snippets">
        <h2>Evidence &middot; {snippets.length} snippets</h2>
        {snippets.map((s, i) => (
          <article key={i}>
            <span className="num">{i + 1}</span>
            <span className="head">
              <a href={s.url} target="_blank" rel="noreferrer">{s.title || s.url}</a>
              <span className="domain">{s.domain}</span>
            </span>
            {/* select_done sends `chars`, a length, not the snippet text.
                Showing the length is honest; the text is not on the wire. */}
            <p>
              {s.chars != null && `${s.chars.toLocaleString()} characters`}
              {s.score != null && ` · relevance ${s.score.toFixed(3)}`}
              {s.text && ` · ${s.text.slice(0, 220)}`}
            </p>
          </article>
        ))}
      </section>
  );
}

function StageTrail({ run }: { run: RunState }) {
  return (
    <ol className="trail">
      {STAGES.map((s) => (
        <li key={s} className={run.stages[s]}>
          <span className="dot" />
          <span className="name">{s}</span>
          {s === "plan" && run.plan.length > 0 && (
            <span className="detail">{run.plan.length}</span>
          )}
          {s === "fetch" && run.pagesFetched > 0 && (
            <span className="detail">{run.pagesFetched}</span>
          )}
          {s === "select" && run.snippets.length > 0 && (
            <span className="detail">{run.snippets.length}</span>
          )}
        </li>
      ))}
    </ol>
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
        stages: { ...r.stages, search: "done", fetch: "running" },
      };
    case "fetch_done":
      return {
        ...r,
        pages: ev.pages,
        pagesFetched: ev.pages.length,
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
