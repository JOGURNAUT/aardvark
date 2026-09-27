import type { AgentEvent } from "./types";

// Server-Sent Events over fetch, not EventSource.
//
// EventSource is the obvious tool and the wrong one. It only issues GET, so the
// query would have to ride in the URL, and closing it does not reliably reach
// the server. fetch gives a POST body and, more importantly, an AbortSignal
// that drops the TCP connection, which is what makes the server stop pulling
// paid tokens from the provider.
//
// The cost of not using EventSource is that its framing has to be reimplemented
// here: reconnection, last-event-id and the wire format. Only the wire format is
// needed, because an abandoned research run should not silently resume.

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function createSession(title?: string): Promise<string> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title ?? null }),
  });
  if (!res.ok) throw new HttpError(res.status, await res.text());
  const data = (await res.json()) as { session_id: string };
  return data.session_id;
}

export async function getTurns(sessionId: string): Promise<unknown[]> {
  const res = await fetch(`/api/sessions/${sessionId}/turns`);
  if (!res.ok) throw new HttpError(res.status, await res.text());
  return res.json();
}

/**
 * Streams agent events for one question.
 *
 * Yields parsed events as they arrive. Throws on a non-2xx response. An abort
 * surfaces as a DOMException named "AbortError", which the caller is expected
 * to treat as a user action rather than a failure.
 */
export async function* ask(
  sessionId: string,
  query: string,
  signal: AbortSignal,
): AsyncGenerator<AgentEvent> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query }),
    signal,
  });

  if (!res.ok) throw new HttpError(res.status, await res.text());
  if (!res.body) throw new Error("response has no body to stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // A chunk boundary can land anywhere, including inside a JSON object or
      // between the two newlines that terminate an event. Everything after the
      // last complete separator stays in the buffer for the next read.
      buffer += decoder.decode(value, { stream: true });

      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        try {
          yield JSON.parse(payload) as AgentEvent;
        } catch {
          // A frame that will not parse is a bug on the wire, not a reason to
          // kill a run that is otherwise producing an answer. Surface it as an
          // error event and keep reading.
          yield {
            type: "error",
            stage: "client",
            message: `unparseable event frame: ${payload.slice(0, 120)}`,
          };
        }
      }
    }
  } finally {
    // Releasing the lock lets the browser tear the connection down promptly on
    // abort instead of waiting for GC.
    reader.releaseLock();
  }
}
