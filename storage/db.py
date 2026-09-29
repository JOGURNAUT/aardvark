import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    rolling_summary TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,        -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    plan TEXT,
    search_queries_json TEXT,    -- list[str]
    urls_opened_json TEXT,       -- list[{url,title,domain,retrieved_at}]
    snippets_json TEXT,          -- list[{url,title,text,score}]
    final_answer TEXT,
    citations_json TEXT,         -- list[{marker,url,title,domain}]
    created_at TEXT NOT NULL,
    latency_ms INTEGER,
    stage_latencies_json TEXT DEFAULT '{}',   -- dict[stage_name -> seconds]
    pending_urls_json TEXT DEFAULT '[]',      -- list[{url,domain,rank,reason}] not yet opened
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        #backfill columns for databases created before they were added.
        #ALTER TABLE ADD COLUMN is the only schema change SQLite does cheaply,
        #and re-running it on a current database is the normal case, not an error.
        for ddl in (
            "ALTER TABLE turns ADD COLUMN stage_latencies_json TEXT DEFAULT '{}'",
            "ALTER TABLE turns ADD COLUMN pending_urls_json TEXT DEFAULT '[]'",
        ):
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass  #column already exists, normal


# ---- sessions ----

def create_session(title: str | None = None) -> str:
    sid = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title or "Untitled", _now()),
        )
    return sid


def list_sessions() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def update_rolling_summary(session_id: str, summary: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET rolling_summary = ? WHERE id = ?",
            (summary, session_id),
        )


def rename_session(session_id: str, title: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )


# ---- messages ----

def add_message(session_id: str, role: str, content: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_messages(session_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---- turns ----

def save_turn(
    session_id: str,
    query: str,
    plan: str,
    search_queries: list[str],
    urls_opened: list[dict[str, Any]],
    snippets: list[dict[str, Any]],
    final_answer: str,
    citations: list[dict[str, Any]],
    latency_ms: int,
    stage_latencies: dict[str, float] | None = None,
    pending_urls: list[dict[str, Any]] | None = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO turns
               (session_id, query, plan, search_queries_json, urls_opened_json,
                snippets_json, final_answer, citations_json, created_at, latency_ms,
                stage_latencies_json, pending_urls_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                query,
                plan,
                json.dumps(search_queries),
                json.dumps(urls_opened),
                json.dumps(snippets),
                final_answer,
                json.dumps(citations),
                _now(),
                latency_ms,
                json.dumps(stage_latencies or {}),
                json.dumps(pending_urls or []),
            ),
        )
        return cur.lastrowid


def get_turns(session_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("search_queries_json", "urls_opened_json", "snippets_json",
                  "citations_json", "pending_urls_json"):
            d[k.replace("_json", "")] = json.loads(d.pop(k) or "[]")
        #stage_latencies is a dict, not a list
        d["stage_latencies"] = json.loads(d.pop("stage_latencies_json", None) or "{}")
        out.append(d)
    return out
