"""
对话上下文管理：按 session 维护多轮问答，供 RAG 时带入历史。

支持持久化：persist_dir 不为空时，每轮对话追加写入 {persist_dir}/{session_id}.jsonl，
get_recent_messages 时若内存无该 session 则从文件加载最近 N 轮。
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from config import CONVERSATION_MAX_TURNS


def _sql_chat_enabled() -> bool:
    try:
        from tools.storage.sql.db import get_session_factory

        return get_session_factory() is not None
    except Exception:
        return False


def collect_session_embed_ids_for_namespace(messages: List[dict], namespace: str) -> List[str]:
    """从会话消息中收集与给定 namespace 一致的「会话上传入库」ingest_id（按出现顺序去重）。"""
    ns = (namespace or "").strip()
    out: list[str] = []
    seen: set[str] = set()
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        se = m.get("session_embed")
        if not isinstance(se, dict):
            continue
        if (se.get("namespace") or "").strip() != ns:
            continue
        iid = str(se.get("ingest_id") or "").strip()
        if iid and iid not in seen:
            seen.add(iid)
            out.append(iid)
    return out


class ConversationContextManager:
    """按 session_id 维护最近 N 轮对话（user + assistant 成对），用于拼进 prompt。"""

    def __init__(
        self,
        max_turns: int = CONVERSATION_MAX_TURNS,
        persist_dir: Optional[str] = "data/conversations",
    ) -> None:
        self.max_turns = max_turns
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._sessions: Dict[str, Deque[dict]] = {}
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
        # 论文「按标题检索」后的待确认列表；与 jsonl 同目录落盘，供 CLI 每次新进程仍能读到
        self._pending_ingest: Dict[str, list[dict[str, Any]]] = {}

    def _get_session(self, session_id: str) -> Deque[dict]:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns * 2)
            loaded = self._load_session_from_redis(session_id)
            if not loaded and _sql_chat_enabled():
                loaded = self._load_session_from_sql(session_id)
            if (not loaded) and self.persist_dir:
                self._load_session_from_file(session_id)
        return self._sessions[session_id]

    def _session_file(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_") or "default"
        return self.persist_dir / f"{safe}.jsonl"

    def _safe_session_slug(self, session_id: str) -> str:
        return session_id.replace("/", "_").replace("\\", "_") or "default"

    def _pending_ingest_path(self, session_id: str) -> Optional[Path]:
        if not self.persist_dir:
            return None
        return self.persist_dir / f"{self._safe_session_slug(session_id)}_pending_ingest.json"

    def _redis_messages_key(self, session_id: str) -> str:
        return f"{self._redis_prefix}:conv:{self._safe_session_slug(session_id)}:messages"

    def _redis_pending_key(self, session_id: str) -> str:
        return f"{self._redis_prefix}:conv:{self._safe_session_slug(session_id)}:pending_ingest"

    def _load_session_from_redis(self, session_id: str) -> bool:
        if self._redis is None:
            return False
        try:
            rows = self._redis.lrange(self._redis_messages_key(session_id), 0, -1) or []
            if not rows:
                return False
            for line in rows:
                if not line:
                    continue
                obj = json.loads(line)
                row: dict = {
                    "role": obj.get("role", "user"),
                    "content": obj.get("content", ""),
                }
                ips = obj.get("image_paths")
                if isinstance(ips, list) and ips:
                    row["image_paths"] = [str(x) for x in ips if str(x).strip()]
                se = obj.get("session_embed")
                if isinstance(se, dict) and str(se.get("ingest_id") or "").strip():
                    row["session_embed"] = se
                self._sessions[session_id].append(row)
            return True
        except Exception:
            return False

    def _load_session_from_sql(self, session_id: str) -> bool:
        try:
            from tools.storage.repos import chat_repo

            rows = chat_repo.fetch_recent_messages(session_id, max_turns=self.max_turns)
            if not rows:
                return False
            for row in rows:
                self._sessions[session_id].append(row)
            return True
        except Exception:
            return False

    def _load_session_from_file(self, session_id: str) -> None:
        path = self._session_file(session_id)
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if not line:
                    continue
                obj = json.loads(line)
                row: dict = {
                    "role": obj.get("role", "user"),
                    "content": obj.get("content", ""),
                }
                ips = obj.get("image_paths")
                if isinstance(ips, list) and ips:
                    row["image_paths"] = [str(x) for x in ips if str(x).strip()]
                se = obj.get("session_embed")
                if isinstance(se, dict) and str(se.get("ingest_id") or "").strip():
                    row["session_embed"] = se
                self._sessions[session_id].append(row)
        except Exception:
            pass

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        image_paths: list[str] | None = None,
    ) -> None:
        """追加一轮：role 为 'user' 或 'assistant'；user 可附带 image_paths（相对/绝对路径字符串）。"""
        sess = self._get_session(session_id)
        msg: dict = {"role": role, "content": content}
        if role == "user" and image_paths:
            cleaned = [str(p).strip() for p in image_paths if str(p).strip()]
            if cleaned:
                msg["image_paths"] = cleaned
        sess.append(msg)
        if _sql_chat_enabled():
            try:
                from tools.storage.repos import chat_repo

                chat_repo.append_turn(
                    session_id,
                    role=role,
                    content=content,
                    image_paths=msg.get("image_paths"),
                    session_embed=None,
                    namespace="default",
                )
                try:
                    from tools.storage.archive.event_logger import log_chat_event

                    log_chat_event(session_id, {"type": "turn", "role": role, "content_len": len(content or "")})
                except Exception:
                    pass
            except Exception:
                pass
        if self._redis is not None:
            try:
                self._redis.rpush(
                    self._redis_messages_key(session_id),
                    json.dumps(msg, ensure_ascii=False),
                )
                self._redis.ltrim(
                    self._redis_messages_key(session_id),
                    -self.max_turns * 2,
                    -1,
                )
                return
            except Exception:
                pass
        if self.persist_dir:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                with self._session_file(session_id).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def add_session_embed(
        self,
        session_id: str,
        *,
        ingest_id: str,
        namespace: str,
        filename: str,
        chunks_added: int,
    ) -> None:
        """记录「本会话文件已写入向量库」，写入内存 deque 并追加 jsonl（供后续检索与会话展示）。"""
        sid = (session_id or "").strip()
        if not sid:
            return
        iid = (ingest_id or "").strip()
        if not iid:
            return
        ns = (namespace or "").strip()
        fn = (filename or "").strip() or "unknown"
        try:
            n_chunks = int(chunks_added)
        except (TypeError, ValueError):
            n_chunks = 0
        content = (
            f"[文件已入库]「{fn}」已写入知识分区「{ns}」（{n_chunks} 个文本块）。"
            f"在同一分区下继续提问时，将优先从该文件检索。"
        )
        msg: dict = {
            "role": "user",
            "content": content,
            "session_embed": {
                "ingest_id": iid,
                "namespace": ns,
                "filename": fn,
                "chunks_added": n_chunks,
            },
        }
        sess = self._get_session(sid)
        sess.append(msg)
        if _sql_chat_enabled():
            try:
                from tools.storage.repos import chat_repo

                chat_repo.append_turn(
                    sid,
                    role="user",
                    content=content,
                    image_paths=None,
                    session_embed=msg.get("session_embed"),
                    namespace=ns or "default",
                )
                try:
                    from tools.storage.archive.event_logger import log_chat_event

                    log_chat_event(sid, {"type": "session_embed", "ingest_id": iid, "namespace": ns})
                except Exception:
                    pass
            except Exception:
                pass
        if self._redis is not None:
            try:
                self._redis.rpush(
                    self._redis_messages_key(sid),
                    json.dumps(msg, ensure_ascii=False),
                )
                self._redis.ltrim(
                    self._redis_messages_key(sid),
                    -self.max_turns * 2,
                    -1,
                )
                return
            except Exception:
                pass
        if self.persist_dir:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                with self._session_file(sid).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def get_recent_messages(self, session_id: str) -> List[dict]:
        """返回该 session 最近 N 轮消息；user 可有可选 image_paths。"""
        return list(self._get_session(session_id))

    def set_pending_ingest_candidates(
        self, session_id: str, candidates: list[dict[str, Any]]
    ) -> None:
        """保存「待用户选序号确认入库」的 arXiv 候选（覆盖该 session 旧列表）。"""
        sid = (session_id or "").strip()
        if not sid:
            return
        self._pending_ingest[sid] = list(candidates)[:20]
        if self._redis is not None:
            try:
                self._redis.set(
                    self._redis_pending_key(sid),
                    json.dumps(self._pending_ingest[sid], ensure_ascii=False),
                )
            except Exception:
                pass
        path = self._pending_ingest_path(sid)
        if path is not None:
            try:
                self.persist_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(self._pending_ingest[sid], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

    def get_pending_ingest_candidates(self, session_id: str) -> list[dict[str, Any]]:
        sid = (session_id or "").strip()
        if not sid:
            return []
        cached = self._pending_ingest.get(sid)
        if cached:
            return list(cached)
        if self._redis is not None:
            try:
                raw = self._redis.get(self._redis_pending_key(sid))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, list) and data:
                        self._pending_ingest[sid] = data[:20]
                        return list(self._pending_ingest[sid])
            except Exception:
                pass
        path = self._pending_ingest_path(sid)
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    self._pending_ingest[sid] = data[:20]
                    return list(self._pending_ingest[sid])
            except Exception:
                pass
        return []

    def clear_pending_ingest(self, session_id: str) -> None:
        sid = (session_id or "").strip()
        self._pending_ingest.pop(sid, None)
        if self._redis is not None and sid:
            try:
                self._redis.delete(self._redis_pending_key(sid))
            except Exception:
                pass
        path = self._pending_ingest_path(sid)
        if path is not None and path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    def clear(self, session_id: str | None = None) -> None:
        """清空指定 session；若 session_id 为 None 则清空所有。"""
        if session_id is None:
            self._sessions.clear()
            self._pending_ingest.clear()
            if self._redis is not None:
                try:
                    keys = self._redis.keys(f"{self._redis_prefix}:conv:*")
                    if keys:
                        self._redis.delete(*keys)
                except Exception:
                    pass
            if self.persist_dir and self.persist_dir.exists():
                for p in self.persist_dir.glob("*_pending_ingest.json"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        else:
            self._sessions.pop(session_id, None)
            self.clear_pending_ingest(session_id)
            if self._redis is not None:
                try:
                    self._redis.delete(self._redis_messages_key(session_id))
                except Exception:
                    pass


# 单例，供智能体 / 接口 使用（默认开启对话持久化到 data/conversations）
conversation_manager = ConversationContextManager()
