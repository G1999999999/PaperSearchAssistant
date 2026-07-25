from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.storage.archive.jsonl_writer import append_jsonl


def log_chat_event(session_id: str, event: dict[str, Any]) -> None:
    """会话事件归档：data/archive/chat_events/{session}.jsonl"""
    sid = (session_id or "default").replace("/", "_")
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    append_jsonl(f"data/archive/chat_events/{sid}.jsonl", rec)
