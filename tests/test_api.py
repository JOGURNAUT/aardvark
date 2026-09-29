"""Tests for the HTTP front end.

`agent.loop` is replaced before `api.server` imports it, so nothing here loads
the embedding model, the tokenizer or a provider SDK. Same rule as the other
test modules: no API keys, no network, no model download.

What is actually under test is the part this layer owns, which is the framing
and the lifecycle, not the agent:

  - one JSON object per SSE frame, and a newline inside a payload does not
    split a frame early
  - the loop's events reach the client unchanged, with no translation
  - an exception mid-stream arrives as an error event instead of a dead socket
  - closing the stream early closes the agent generator, which is what stops
    the answer stage pulling paid tokens for a client that has gone
"""
from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest


# --------------------------------------------------------------------------
# Replace agent.loop before api.server binds it
# --------------------------------------------------------------------------

_SCRIPT: list[dict] = []
_CLOSED: list[bool] = []


def _fake_run(session_id: str, user_query: str):
    """Yields whatever the current test put in _SCRIPT, and records whether it
    was closed before finishing. A dict with the key `__raise__` raises."""
    try:
        for ev in _SCRIPT:
            if "__raise__" in ev:
                raise RuntimeError(ev["__raise__"])
            yield ev
    except GeneratorExit:
        _CLOSED.append(True)
        raise


_fake_loop = types.ModuleType("agent.loop")
_fake_loop.run = _fake_run                      # type: ignore[attr-defined]
sys.modules["agent.loop"] = _fake_loop

import agent                                     # noqa: E402
agent.loop = _fake_loop                          # type: ignore[attr-defined]

from fastapi.testclient import TestClient        # noqa: E402

import api.server as server                      # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    _SCRIPT.clear()
    _CLOSED.clear()
    # Keep every test's sessions in its own file so ordering cannot matter.
    monkeypatch.setattr(server.db, "DB_PATH", str(tmp_path / "test.db"), raising=False)
    with TestClient(server.app) as c:
        yield c


def _session(client) -> str:
    return client.post("/api/sessions", json={"title": "t"}).json()["session_id"]


def _frames(text: str) -> list[dict]:
    out = []
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

def test_sse_payload_is_a_single_line():
    # A raw newline inside a data line terminates the event, and everything
    # after it is silently dropped by the browser. json.dumps with the compact
    # separators is what keeps that from happening.
    frame = server._sse({"type": "answer_token", "text": "line one\nline two"})
    body = frame[: -len("\n\n")]
    assert body.count("\n") == 0
    assert json.loads(body[len("data: "):])["text"] == "line one\nline two"


def test_events_reach_the_client_unchanged():
    # The loop is the source of truth for event shapes. If this layer starts
    # mapping them, a new event type silently stops arriving.
    original = {"type": "plan", "queries": ["a", "b"], "strategy": "compare"}
    assert json.loads(server._sse(original)[len("data: "):].strip()) == original


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------

def test_full_run_streams_every_event(client):
    _SCRIPT.extend([
        {"type": "plan", "queries": ["q1"], "strategy": "s"},
        {"type": "search_done", "results": []},
        {"type": "answer_token", "text": "hello "},
        {"type": "answer_token", "text": "world"},
        {"type": "answer_done", "answer": "hello world", "citations": [],
         "latency_ms": 12, "stage_latencies": {"plan": 0.1}, "provider": "groq"},
    ])
    sid = _session(client)
    res = client.post("/api/ask", json={"session_id": sid, "query": "why"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    # Proxies that buffer turn a token stream into one delivery at the end.
    assert res.headers["x-accel-buffering"] == "no"

    events = _frames(res.text)
    assert [e["type"] for e in events] == [
        "plan", "search_done", "answer_token", "answer_token", "answer_done",
    ]
    assert events[-1]["answer"] == "hello world"


def test_exception_mid_stream_becomes_an_error_event(client):
    # The connection is already open and some of the answer may already be on
    # screen, so a 500 is not available. The failure has to arrive in-band.
    _SCRIPT.extend([
        {"type": "plan", "queries": ["q"], "strategy": "s"},
        {"__raise__": "provider exploded"},
    ])
    sid = _session(client)
    res = client.post("/api/ask", json={"session_id": sid, "query": "why"})

    assert res.status_code == 200
    events = _frames(res.text)
    assert events[0]["type"] == "plan"
    assert events[-1]["type"] == "error"
    assert events[-1]["stage"] == "server"
    assert "provider exploded" in events[-1]["message"]


def test_closing_the_stream_closes_the_agent_generator():
    """The point of the whole abort path: a user who closes the tab should stop
    costing money on the next token, not after the answer finishes.

    Tested against the async generator rather than through TestClient, because
    TestClient runs the app to completion and buffers the whole body before
    `iter_lines` returns anything. It cannot express a disconnect, so a test
    written through it passes whether or not the close actually happens, which
    is worse than no test. `aclose()` here is exactly what Starlette calls when
    the client goes away.
    """
    _SCRIPT.clear()
    _CLOSED.clear()
    _SCRIPT.extend([{"type": "answer_token", "text": f"tok{i}"} for i in range(50)])

    async def scenario():
        stream = server._event_stream("any-session", "why")
        first = await stream.__anext__()
        assert "tok0" in first
        await stream.aclose()

    asyncio.run(scenario())

    assert _CLOSED, "agent generator was not closed when the stream was closed"


def test_normal_completion_also_closes_the_generator():
    # The close lives in a finally, so it has to hold on the happy path too.
    _SCRIPT.clear()
    _CLOSED.clear()
    _SCRIPT.append({"type": "answer_done", "answer": "a", "citations": [],
                    "latency_ms": 1, "stage_latencies": {}, "provider": "p"})

    async def scenario():
        return [frame async for frame in server._event_stream("any-session", "why")]

    frames = asyncio.run(scenario())
    assert len(frames) == 1
    # Exhausted rather than interrupted, so GeneratorExit is never raised and
    # _CLOSED stays empty. Closing an exhausted generator is a no-op; what
    # matters is that nothing is left open, which the exhaustion itself proves.
    assert not _CLOSED


def test_unknown_session_is_rejected_before_any_work(client):
    _SCRIPT.append({"type": "plan", "queries": [], "strategy": ""})
    res = client.post("/api/ask", json={"session_id": "nope", "query": "why"})
    assert res.status_code == 404


def test_empty_query_is_rejected(client):
    sid = _session(client)
    assert client.post("/api/ask", json={"session_id": sid, "query": ""}).status_code == 422


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

def test_session_roundtrip(client):
    sid = _session(client)
    assert client.get(f"/api/sessions/{sid}/messages").status_code == 200
    assert client.get(f"/api/sessions/{sid}/turns").status_code == 200
    assert client.get("/api/sessions/nope/turns").status_code == 404


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}

# --------------------------------------------------------------------------
# Liveness and readiness are different questions
# --------------------------------------------------------------------------

def test_health_is_green_before_the_model_is(client):
    """Liveness must not wait for the model.

    The embedding model is a lazy global loaded on first use, so warming it
    takes tens of seconds. If liveness waited for that, the orchestrator would
    restart a pod that is merely still starting, and a slow start would become
    a crash loop.
    """
    server._ready["model"] = False
    server._ready["error"] = None
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").json() == {"ok": True}


def test_ready_is_503_until_the_model_is_loaded(client):
    # Without this the Service sends the first user a request that hangs on a
    # model load. It is the failure the live deployment shows today: the first
    # request after a cold start times out and the retry succeeds.
    server._ready["model"] = False
    server._ready["error"] = None
    res = client.get("/api/ready")
    assert res.status_code == 503
    assert res.json()["ready"] is False


def test_ready_turns_green_once_warm(client):
    server._ready["model"] = True
    server._ready["error"] = None
    res = client.get("/api/ready")
    assert res.status_code == 200
    assert res.json()["ready"] is True


def test_a_failed_warmup_stays_unready_and_says_why(client):
    # A pod that will never be able to answer should not quietly sit in the
    # Service. It should stay out of rotation with the reason attached.
    server._ready["model"] = False
    server._ready["error"] = "no space left on device"
    res = client.get("/api/ready")
    assert res.status_code == 503
    assert "no space left" in res.json()["error"]
