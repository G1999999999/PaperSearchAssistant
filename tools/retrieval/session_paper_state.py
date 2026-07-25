from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _safe_sid(session_id: str) -> str:
    return (session_id or "").replace("/", "_").replace("\\", "_") or "default"


class SessionPaperStateStore:
    """Persist current paper + last paper-search candidates per session."""

    def __init__(self, persist_dir: str = "data/conversations") -> None:
        self.persist_dir = Path(persist_dir)
        self._mem: dict[str, dict[str, Any]] = {}
        self._redis = None
        self._redis_prefix = (os.getenv("REDIS_PREFIX", "psa2") or "psa2").strip()
        redis_url = (os.getenv("REDIS_URL", "") or "").strip()
        if redis_url:
            try:
                import redis  # type: ignore[import-not-found]

                cli = redis.Redis.from_url(redis_url, decode_responses=True)
                cli.ping()
                self._redis = cli
            except Exception:
                self._redis = None

    def _key(self, sid: str) -> str:
        return f"{self._redis_prefix}:conv:{_safe_sid(sid)}:paper_state"

    def _path(self, sid: str) -> Path:
        return self.persist_dir / f"{_safe_sid(sid)}_paper_state.json"

    def _load(self, sid: str) -> dict[str, Any]:
        if sid in self._mem:
            return self._mem[sid]
        if self._redis is not None:
            try:
                raw = self._redis.get(self._key(sid))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        self._mem[sid] = data
                        return data
            except Exception:
                pass
        p = self._path(sid)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._mem[sid] = data
                    return data
            except Exception:
                pass
        data = {"current_paper": None, "last_candidates": []}
        self._mem[sid] = data
        return data

    def _save(self, sid: str, data: dict[str, Any]) -> None:
        self._mem[sid] = data
        if self._redis is not None:
            try:
                self._redis.set(self._key(sid), json.dumps(data, ensure_ascii=False))
            except Exception:
                pass
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._path(sid).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def set_candidates(self, session_id: str, candidates: list[dict[str, Any]]) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        data = self._load(sid)
        data["last_candidates"] = list(candidates or [])[:20]
        self._save(sid, data)

    def get_candidates(self, session_id: str) -> list[dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        data = self._load(sid)
        arr = data.get("last_candidates")
        return list(arr) if isinstance(arr, list) else []

    def set_current_paper(self, session_id: str, paper: dict[str, Any]) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        data = self._load(sid)
        data["current_paper"] = dict(paper or {})
        self._save(sid, data)

    def get_current_paper(self, session_id: str) -> dict[str, Any] | None:
        sid = (session_id or "").strip()
        if not sid:
            return None
        data = self._load(sid)
        cur = data.get("current_paper")
        return dict(cur) if isinstance(cur, dict) and cur else None

    def clear_current_paper(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        if not sid:
            return
        data = self._load(sid)
        data["current_paper"] = None
        self._save(sid, data)


paper_state_store = SessionPaperStateStore()

