import { describe, expect, it, vi, afterEach } from "vitest";
import { ask, HttpError } from "./api";
import type { AgentEvent } from "./types";

// The SSE wire format is reimplemented here rather than inherited from
// EventSource, so the framing is ours to get wrong. These tests exist for the
// one failure that is invisible in a browser until it is not: a chunk boundary
// landing inside a frame.

function streamOf(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

function mockFetch(res: Response) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res));
}

async function collect(signal = new AbortController().signal): Promise<AgentEvent[]> {
  const out: AgentEvent[] = [];
  for await (const ev of ask("sid", "q", signal)) out.push(ev);
  return out;
}

afterEach(() => vi.unstubAllGlobals());

describe("SSE framing", () => {
  it("parses one event per frame", async () => {
    mockFetch(
      streamOf([
        'data: {"type":"plan","queries":["a"],"strategy":"s"}\n\n',
        'data: {"type":"answer_token","text":"hi"}\n\n',
      ]),
    );
    const events = await collect();
    expect(events.map((e) => e.type)).toEqual(["plan", "answer_token"]);
  });

  it("survives a chunk boundary inside a JSON payload", async () => {
    // TCP does not respect our frame boundaries. This is the case that works
    // on localhost, where everything arrives in one chunk, and breaks behind a
    // proxy or on a slow connection.
    mockFetch(streamOf(['data: {"type":"answer_to', 'ken","text":"split"}\n\n']));
    const events = await collect();
    expect(events).toEqual([{ type: "answer_token", text: "split" }]);
  });

  it("survives a chunk boundary between the two terminating newlines", async () => {
    mockFetch(streamOf(['data: {"type":"answer_token","text":"a"}\n', '\ndata: {"type":"answer_token","text":"b"}\n\n']));
    const events = await collect();
    expect(events.map((e) => (e as { text: string }).text)).toEqual(["a", "b"]);
  });

  it("ignores keep-alive comments and blank frames", async () => {
    mockFetch(streamOf([": keep-alive\n\n", 'data: {"type":"answer_token","text":"x"}\n\n', "\n\n"]));
    const events = await collect();
    expect(events).toHaveLength(1);
  });

  it("reports an unparseable frame instead of killing the run", async () => {
    // A bad frame is a bug on the wire. It is not a reason to discard an answer
    // that is otherwise still arriving.
    mockFetch(
      streamOf([
        "data: {not json}\n\n",
        'data: {"type":"answer_token","text":"still here"}\n\n',
      ]),
    );
    const events = await collect();
    expect(events[0].type).toBe("error");
    expect(events[1]).toEqual({ type: "answer_token", text: "still here" });
  });

  it("does not emit a trailing partial frame", async () => {
    // A stream cut mid-frame must not produce a half-parsed event.
    mockFetch(streamOf(['data: {"type":"answer_token","text":"ok"}\n\n', 'data: {"type":"answer_to']));
    const events = await collect();
    expect(events).toHaveLength(1);
  });
});

describe("errors", () => {
  it("throws HttpError on a non-2xx response", async () => {
    mockFetch(new Response("unknown session_id", { status: 404 }));
    await expect(collect()).rejects.toBeInstanceOf(HttpError);
  });

  it("throws when the response has no body to stream", async () => {
    mockFetch(new Response(null, { status: 200 }));
    await expect(collect()).rejects.toThrow(/no body/);
  });
});
