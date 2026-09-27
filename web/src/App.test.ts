import { describe, expect, it } from "vitest";
import { reduce } from "./App";
import type { AgentEvent } from "./types";

// `reduce` is exported and pure for exactly this reason: the stage machine and
// the answer accumulation are the parts that go wrong, and neither of them
// needs a DOM to check.

const empty = () =>
  reduce(
    {
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
    },
    { type: "plan", queries: [], strategy: "" },
  );

function play(events: AgentEvent[]) {
  return events.reduce(reduce, empty());
}

describe("stage machine", () => {
  it("advances each stage as its event arrives", () => {
    const r = play([
      { type: "search_done", results: [] },
      { type: "fetch_done", pages: [{ url: "u", title: "t", domain: "d" }] },
      { type: "select_done", snippets: [{ title: "t", domain: "d", url: "u", text: "x" }] },
    ]);
    expect(r.stages.search).toBe("done");
    expect(r.stages.fetch).toBe("done");
    expect(r.stages.select).toBe("done");
    expect(r.stages.answer).toBe("running");
  });

  it("counts pages and snippets for the trail", () => {
    const r = play([
      { type: "fetch_done", pages: [{ url: "a", title: "", domain: "" }, { url: "b", title: "", domain: "" }] },
      { type: "select_done", snippets: [{ title: "", domain: "", url: "", text: "" }] },
    ]);
    expect(r.pagesFetched).toBe(2);
    expect(r.snippets).toHaveLength(1);
  });
});

describe("answer accumulation", () => {
  it("appends streamed tokens in order", () => {
    const r = play([
      { type: "answer_token", text: "one " },
      { type: "answer_token", text: "two" },
    ]);
    expect(r.answer).toBe("one two");
  });

  it("replaces the streamed text with the final answer, not appends it", () => {
    // The loop re-emits the whole answer in answer_done after the citation
    // guard has stripped markers pointing at nothing. Appending here prints
    // the answer twice, which is the first bug this front end hit.
    const r = play([
      { type: "answer_token", text: "draft with [9] " },
      {
        type: "answer_done",
        answer: "draft with ",
        citations: [],
        latency_ms: 5,
        stage_latencies: { plan: 0.2 },
        provider: "groq",
      },
    ]);
    expect(r.answer).toBe("draft with ");
    expect(r.latencyMs).toBe(5);
    expect(r.stageLatencies).toEqual({ plan: 0.2 });
  });

  it("tolerates answer_done without stage_latencies", () => {
    const r = play([
      {
        type: "answer_done",
        answer: "a",
        citations: [],
        latency_ms: 1,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        stage_latencies: undefined as any,
        provider: "p",
      },
    ]);
    expect(r.stageLatencies).toEqual({});
  });
});

describe("failure handling", () => {
  it("collects errors without discarding the run", () => {
    const r = play([
      { type: "answer_token", text: "partial" },
      { type: "error", stage: "fetch", message: "timeout" },
    ]);
    expect(r.errors).toEqual([{ stage: "fetch", message: "timeout" }]);
    expect(r.answer).toBe("partial");
  });

  it("counts events it does not recognise instead of dropping them", () => {
    // The loop owns the event shapes and types.ts is hand-written against a
    // docstring. This counter is what makes the drift visible rather than
    // silent, so it is worth a test of its own.
    const r = play([{ type: "rerank_done", kept: 3 } as unknown as AgentEvent]);
    expect(r.unknownEvents).toBe(1);
  });
});
