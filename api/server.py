"""HTTP front end over the same agent generator the Streamlit UI drives.

This is the third interface over `agent/loop.py`, after `ui/app.py` (Streamlit)
and `ui/gradio_app.py` (Gradio). The loop's contract has always been that it
yields typed events and knows nothing about who consumes them. Two Python UIs
assert that; a front end in another language tests it. Writing this one found
two places where the loop had leaked assumptions about its caller, both noted
in the README.

Streaming shape
---------------
Events go out as Server-Sent Events, one JSON object per `data:` line, in the
same shape `loop.run()` yields them. No translation layer: if a new event type
is added to the loop, it reaches the browser without a change here. That is the
point of forwarding rather than mapping.

Why POST and not EventSource
----------------------------
`EventSource` is the obvious choice for SSE and it is the wrong one here. It
only issues GET, so the query has to travel in the URL, and it has no abort
that reaches the server. The browser uses `fetch` with a `ReadableStream` body
instead, which gives both a request body and an `AbortController`.

Cancellation
------------
Aborting matters more than it looks. The answer stage streams tokens from a
paid API; a user who closes the tab mid-run should stop costing money on the
next token, not after the whole answer finishes.

The obvious implementation is a sync generator, and it does not actually do
this. Starlette hands a sync iterator to `iterate_in_threadpool`, which pulls
one item at a time and simply stops pulling when the task is cancelled. It
never calls `.close()` on the underlying generator, so `GeneratorExit` arrives
whenever garbage collection gets round to it, which on a stream holding an open
provider response is exactly the wrong time for it to be non-deterministic.

So the stream is an async generator that drives the sync one by hand and closes
it in a `finally`. Starlette does call `aclose()` on an async generator when the
client disconnects, and `finally` runs on cancellation, so the close is
deterministic: it propagates into `loop.run()`, which closes `llm.stream()`,
which closes the provider response. `tests/test_api.py` asserts it.

The same `finally` records that the run ended without an answer, because an
abandoned run and a failed one are otherwise indistinguishable in the audit
trail: both are a turn with no `answer_done`.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from typing import AsyncIterator, Iterator

import anyio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import loop as agent_loop
from storage import db

log = logging.getLogger(__name__)

app = FastAPI(title="Aardvark", docs_url="/api/docs", openapi_url="/api/openapi.json")

# The Vite dev server runs on a different origin. In the container the built
# assets are served from this same app, so this only ever matters in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Liveness and readiness are different questions and this app is a good example
# of why. The embedding model is a lazy global in agent/selector.py, loaded on
# first use inside the select stage, so a container that has started is not a
# container that can answer: the first real query pays a ten-to-twenty second
# model load. On Azure that shows up as the first request timing out and the
# second succeeding.
#
# So the process starts, says it is alive immediately, and warms the model on a
# background thread. /api/ready only goes green once that finishes, which is
# what an orchestrator should gate traffic on.
_ready: dict[str, object] = {"model": False, "error": None}


def _warm() -> None:
    try:
        from agent import selector
        selector._embedder()
        _ready["model"] = True
        log.info("embedding model warm")
    except Exception as e:                      # noqa: BLE001
        _ready["error"] = str(e)
        log.exception("warmup failed")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    # Daemon, so a warmup still running does not hold up a shutdown.
    threading.Thread(target=_warm, name="warmup", daemon=True).start()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class NewSession(BaseModel):
    title: str | None = None


class AskRequest(BaseModel):
    session_id: str
    query: str = Field(min_length=1, max_length=4000)


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return db.list_sessions()


@app.post("/api/sessions")
def create_session(body: NewSession) -> dict:
    session_id = db.create_session(body.title)
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> list[dict]:
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")
    return db.get_messages(session_id)


@app.get("/api/sessions/{session_id}/turns")
def get_turns(session_id: str) -> list[dict]:
    """The audit trail: plan, queries, URLs opened, snippets, citations and
    per-stage latencies for every turn. The UI's trace panel reads this, and it
    is also what makes a past answer reviewable rather than just readable."""
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")
    return db.get_turns(session_id)


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------

def _sse(event: dict) -> str:
    # `json.dumps` with no newlines, because a raw newline inside a data line
    # terminates the event early and the browser silently drops the remainder.
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


_DONE = object()


def _next(it: Iterator[dict]) -> object:
    """One step of the agent, run off the event loop. Returns _DONE at the end
    rather than raising, because StopIteration does not survive the hop."""
    return next(it, _DONE)


async def _event_stream(gen: Iterator[dict], label: str) -> AsyncIterator[str]:
    """Drive a sync agent generator and frame it as SSE.

    Takes the generator rather than the arguments to build one, so /api/ask and
    /api/dig share the cancellation path and the logging instead of each
    growing its own copy.
    """
    t0 = time.time()
    completed = False
    events = 0
    try:
        while True:
            event = await anyio.to_thread.run_sync(_next, gen)
            if event is _DONE:
                break
            assert isinstance(event, dict)
            events += 1
            if event.get("type") == "answer_done":
                completed = True
            yield _sse(event)
    except Exception as e:                      # noqa: BLE001 - surfaced to the UI
        log.exception("stream failed")
        yield _sse({"type": "error", "stage": "server", "message": str(e)})
    finally:
        # Runs on normal completion, on an exception, and on cancellation when
        # the client disconnects. This close is the whole point of driving the
        # generator by hand instead of handing it to Starlette.
        gen.close()
        if not completed:
            log.info("%s ended after %d events in %.1fs without answer_done",
                     label, events, time.time() - t0)


@app.post("/api/ask")
def ask(body: AskRequest) -> StreamingResponse:
    if not db.get_session(body.session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")
    return StreamingResponse(
        _event_stream(agent_loop.run(body.session_id, body.query), "ask"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx and several Azure front ends buffer proxied responses by
            # default, which turns a token stream into one delivery at the end.
            "X-Accel-Buffering": "no",
        },
    )


class DigRequest(BaseModel):
    session_id: str


@app.post("/api/dig")
def dig(body: DigRequest) -> StreamingResponse:
    """Open the results the last turn never reached, and answer again.

    No query in the body on purpose: a dig continues the question that was
    already asked, using search results that are already ranked and already
    stored. Letting the client pass a new query here would make it a second
    search wearing a cheaper name.
    """
    if not db.get_session(body.session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")
    return StreamingResponse(
        _event_stream(agent_loop.dig(body.session_id), "dig"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health() -> dict:
    """Liveness: the process is running and can serve HTTP. Deliberately does
    not check the model, because restarting a pod that is merely still warming
    up turns a slow start into a crash loop."""
    return {"ok": True}


@app.get("/api/ready")
def ready() -> JSONResponse:
    """Readiness: this replica can actually answer a research question.

    503 until the embedding model is loaded, so an orchestrator holds traffic
    back rather than sending the first user a request that will hang on a model
    download. A warmup that failed stays 503 and says why.
    """
    ok = bool(_ready["model"])
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ready": ok, "model": _ready["model"], "error": _ready["error"]},
    )


# ---------------------------------------------------------------------------
# Static assets (the built React app), mounted last so /api wins every time
# ---------------------------------------------------------------------------

_WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

if _WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        """Every unmatched path returns index.html so client-side routing works
        on a hard refresh. /api routes are declared above and are matched first."""
        return FileResponse(_WEB_DIST / "index.html")
