from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select

from tools.storage.sql.db import get_session_factory
from tools.storage.sql.models import ChatAttachment, ChatMessage, ChatSession


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_session(session_id: str, *, namespace: str = "default", user_id: str | None = None) -> None:
    factory = get_session_factory()
    if factory is None or not (session_id or "").strip():
        return
    sid = session_id.strip()
    ns = (namespace or "default").strip() or "default"
    session = factory()
    try:
        row = session.scalar(select(ChatSession).where(ChatSession.id == sid))
        if row is None:
            session.add(
                ChatSession(
                    id=sid,
                    user_id=user_id,
                    namespace=ns,
                    message_count=0,
                    status="active",
                )
            )
        else:
            row.namespace = ns
        session.commit()
    finally:
        session.close()


def append_turn(
    session_id: str,
    *,
    role: str,
    content: str,
    image_paths: list[str] | None = None,
    session_embed: dict[str, Any] | None = None,
    namespace: str = "default",
) -> None:
    factory = get_session_factory()
    if factory is None:
        return
    sid = (session_id or "").strip()
    if not sid:
        return
    ensure_session(sid, namespace=namespace)
    extra: dict[str, Any] = {}
    if image_paths:
        extra["image_paths"] = image_paths
    if session_embed:
        extra["session_embed"] = session_embed
    msg = ChatMessage(
        session_id=sid,
        role=role,
        content=content,
        has_images=bool(image_paths),
        has_files=bool(session_embed),
        extra_json=extra if extra else None,
    )
    session = factory()
    try:
        sess = session.scalar(select(ChatSession).where(ChatSession.id == sid))
        if sess:
            sess.message_count = int(sess.message_count or 0) + 1
            sess.last_message_at = _now()
            sess.updated_at = _now()
        session.add(msg)
        session.commit()
    finally:
        session.close()


def fetch_recent_messages(session_id: str, *, max_turns: int) -> list[dict[str, Any]]:
    """返回与 ConversationContextManager 兼容的消息 dict 列表。"""
    factory = get_session_factory()
    if factory is None:
        return []
    sid = (session_id or "").strip()
    if not sid:
        return []
    limit = max(1, min(500, int(max_turns or 5) * 2))
    session = factory()
    try:
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == sid)
            .order_by(desc(ChatMessage.id))
            .limit(limit)
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    rows = list(reversed(rows))
    out: list[dict[str, Any]] = []
    for r in rows:
        d: dict[str, Any] = {"role": r.role, "content": r.content or ""}
        ex = r.extra_json if isinstance(r.extra_json, dict) else {}
        ips = ex.get("image_paths")
        if isinstance(ips, list) and ips:
            d["image_paths"] = [str(x) for x in ips if str(x).strip()]
        se = ex.get("session_embed")
        if isinstance(se, dict) and str(se.get("ingest_id") or "").strip():
            d["session_embed"] = se
        out.append(d)
    return out


def list_session_ids(limit: int = 500) -> list[str]:
    factory = get_session_factory()
    if factory is None:
        return []
    lim = max(1, min(5000, int(limit)))
    session = factory()
    try:
        stmt = select(ChatSession.id).order_by(desc(ChatSession.updated_at)).limit(lim)
        return [str(x) for x in session.scalars(stmt).all()]
    finally:
        session.close()


def list_attachments_for_session(session_id: str) -> list[dict[str, Any]]:
    factory = get_session_factory()
    if factory is None:
        return []
    sid = (session_id or "").strip()
    if not sid:
        return []
    session = factory()
    try:
        stmt = (
            select(ChatAttachment)
            .where(ChatAttachment.session_id == sid)
            .order_by(ChatAttachment.id.asc())
        )
        rows = list(session.scalars(stmt).all())
    finally:
        session.close()
    return [
        {
            "id": r.id,
            "message_id": r.message_id,
            "file_name": r.file_name,
            "file_path": r.file_path,
            "file_type": r.file_type,
            "file_size": r.file_size,
            "asset_kind": r.asset_kind,
            "session_ingest_id": r.session_ingest_id,
        }
        for r in rows
    ]
