import { useCallback, useEffect, useRef, useState } from "react";
import { ask, createSession } from "./api";
import { STAGES } from "./types";
import type { AgentEvent, Citation, Snippet, Stage } from "./types";

type StageState = "idle" | "running" | "done";

interface RunState {
  stages: Record<Stage, StageState>;
  plan: string[];
  strategy: string;
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
  pagesFetched: 0,
  snippets: [],
  answer: "",
  citations: [],
  latencyMs: null,
  stageLatencies: {},
  errors: [],
  unknownEvents: 0,
});

/**
 * Renders [1] and [2,3] markers as links into the snippet list.
 *
 * Deliberately done here and not in the model's output: the server already
 * strips markers that point at nothing, so any marker reaching this point is
 * known to resolve. If one does not, it renders as plain text rather than a
 * dead link, because a broken citation should look broken.
 */
function AnswerText({ text, citations }: { text: string; citations: Citation[] }) {
  const byMarker = new Map(citations.map((c) => [c.marker, c]));
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);

  return (
    <p className="answer">
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
                    <a href={c.url} target="_blank" rel="noreferrer" title={`${c.title} - ${c.domain}`}>
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
    </p>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [run, setRun] = useState<RunState>(emptyRun);
  const [busy, setBusy] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    createSession("web")
      .then(setSessionId)
      .catch((e) => setBootError(String(e)));
  }, []);

  // Abort any in-flight run when the component goes away, so a navigation does
  // not leave the server streaming tokens into a closed connection.
  useEffect(() => () => abortRef.current?.abort(), []);

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const submit = useCallback(async () => {
    if (!sessionId || !query.trim() || busy) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setRun({ ...emptyRun(), stages: { ...emptyRun().stages, plan: "running" } });

    try {
      for await (const ev of ask(sessionId, query, controller.signal)) {
        setRun((r) => reduce(r, ev));
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setRun((r) => ({ ...r, errors: [...r.errors, { stage: "client", message: "stopped" }] }));
      } else {
        setRun((r) => ({ ...r, errors: [...r.errors, { stage: "client", message: String(e) }] }));
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }, [sessionId, query, busy]);

  return (
    <div className="app">
      <header>
        <h1>Beaver</h1>
        <span className="sub">deep research agent</span>
      </header>

      {bootError && <div className="error">could not start a session: {bootError}</div>}

      <div className="composer">
        <textarea
          value={query}
          placeholder="Ask something that needs more than one source."
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
          }}
          disabled={!sessionId}
        />
        <div className="actions">
          <button onClick={submit} disabled={busy || !sessionId || !query.trim()}>
            {busy ? "Researching" : "Ask"}
          </button>
          {busy && (
            <button className="stop" onClick={stop}>
              Stop
            </button>
          )}
          <span className="hint">Ctrl+Enter</span>
        </div>
      </div>

      <StageTrail run={run} />

      {run.errors.map((e, i) => (
        <div className="error" key={i}>
          {e.stage}: {e.message}
        </div>
      ))}

      {run.answer && <AnswerText text={run.answer} citations={run.citations} />}

      {run.latencyMs !== null && (
        <div className="latency">
          {run.latencyMs} ms total
          {Object.entries(run.stageLatencies).map(([s, secs]) => (
            <span key={s} className="chip">
              {s} {Math.round(secs * 1000)}ms
            </span>
          ))}
          {run.unknownEvents > 0 && (
            <span className="chip warn">{run.unknownEvents} unrecognised events</span>
          )}
        </div>
      )}

      {run.snippets.length > 0 && (
        <section className="snippets">
          <h2>Evidence ({run.snippets.length})</h2>
          {run.snippets.map((s, i) => (
            <article key={i}>
              <a href={s.url} target="_blank" rel="noreferrer">
                [{i + 1}] {s.title || s.url}
              </a>
              <span className="domain">{s.domain}</span>
              <p>{s.text.slice(0, 280)}</p>
            </article>
          ))}
        </section>
      )}
    </div>
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
            <span className="detail">{run.plan.length} queries</span>
          )}
          {s === "fetch" && run.pagesFetched > 0 && (
            <span className="detail">{run.pagesFetched} pages</span>
          )}
          {s === "select" && run.snippets.length > 0 && (
            <span className="detail">{run.snippets.length} snippets</span>
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
      return { ...r, stages: { ...r.stages, search: "done", fetch: "running" } };
    case "fetch_done":
      return {
        ...r,
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
