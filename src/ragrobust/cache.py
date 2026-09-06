"""Content-addressed response cache.

P1 Section 5.8 commits to batched, resumable execution so that an interrupted
run continues without re-issuing completed calls. With roughly 69,600 calls in
the full matrix, re-running from scratch after a crash is not affordable, so
resumability is a correctness requirement rather than a convenience.

The cache key covers the model, the family, and every decoding parameter. Change
a prompt or a temperature and you get a new key, which means a stale result can
never be silently reused after a configuration edit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .providers.base import GenerationRequest, GenerationResponse


def cache_key(model: str, family: str, req: GenerationRequest) -> str:
    payload = json.dumps(
        {"model": model, "family": family, **req.cache_key_fields()}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """SQLite-backed cache. Single file, safe across threads, survives restarts."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS responses (
        key TEXT PRIMARY KEY,
        model TEXT NOT NULL,
        family TEXT NOT NULL,
        text TEXT NOT NULL,
        reasoning_trace TEXT,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        latency_s REAL NOT NULL DEFAULT 0,
        meta TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE INDEX IF NOT EXISTS idx_model ON responses(model);
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> GenerationResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model, family, text, reasoning_trace, prompt_tokens, "
                "completion_tokens, latency_s, meta FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return GenerationResponse(
            model=row[0],
            family=row[1],
            text=row[2],
            reasoning_trace=row[3],
            prompt_tokens=row[4],
            completion_tokens=row[5],
            latency_s=row[6],
            meta=json.loads(row[7]),
            cached=True,
        )

    def put(self, key: str, resp: GenerationResponse) -> None:
        # Never cache failures. A transient 429 must not become a permanent
        # empty answer that silently scores as incorrect for the rest of the run.
        if not resp.ok:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, family, text, "
                "reasoning_trace, prompt_tokens, completion_tokens, latency_s, meta) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    resp.model,
                    resp.family,
                    resp.text,
                    resp.reasoning_trace,
                    resp.prompt_tokens,
                    resp.completion_tokens,
                    resp.latency_s,
                    json.dumps(resp.meta),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n, pt, ct = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), "
                "COALESCE(SUM(completion_tokens),0) FROM responses"
            ).fetchone()
            by_model = dict(
                self._conn.execute(
                    "SELECT model, COUNT(*) FROM responses GROUP BY model"
                ).fetchall()
            )
        return {
            "n_cached": n,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "by_model": by_model,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
