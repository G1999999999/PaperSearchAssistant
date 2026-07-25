"""
长期记忆：在向量库中存储并检索精炼摘要内容。

设计上故意保持简单：
- 将每条记忆以短文本块（摘要）形式存入专用 namespace。
- 通过语义检索（vector_store.retrieve）获取与查询相关的记忆。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

from config import DEFAULT_RETRIEVAL_STRATEGY
from models_qwen import qwen
from tools.rag.knowledge import vector_store
from tools.rag.language import expand_retrieval_queries


def memory_namespace(session_id: str | None) -> str:
    """长期记忆的命名空间。"""

    sid = (session_id or "default").strip() or "default"
    return f"memory:{sid}"


def add_memory(
    *,
    session_id: str | None,
    text: str,
    source: str = "agent",
    extra_metadata: dict | None = None,
) -> int:
    """将一条长期记忆写入向量库。"""

    meta = {
        "type": "long_term_memory",
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    # 保持精炼：切分可以，但这里希望写入的是“摘要大小”的文本。
    return vector_store.embed_document(
        text=text,
        namespace=memory_namespace(session_id),
        chunk_size=500,
        chunk_overlap=50,
        extra_metadata=meta,
    )


def retrieve_memories(
    *,
    session_id: str | None,
    query: str,
    k: int = 6,
    score_threshold: float = 0.5,
    strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> List[dict]:
    """检索与查询相关的记忆条目。

    返回一个归一化后的字典列表：
      {"text": ..., "source": ..., "score": ...}
    """

    ns = memory_namespace(session_id)
    queries = expand_retrieval_queries(
        query,
        strategy=strategy,
        llm=qwen,
    )
    if not queries:
        fb = (query or "").strip()
        queries = [fb] if fb else []
    docs_scores = vector_store.retrieve(
        queries=queries,
        namespace=ns,
        k=k,
        score_threshold=score_threshold,
        strategy=strategy,
    )
    items: list[dict] = []
    for doc, score in docs_scores:
        meta = getattr(doc, "metadata", {}) or {}
        items.append(
            {
                "text": getattr(doc, "page_content", str(doc)),
                "source": meta.get("source", "memory"),
                "score": float(score),
                "created_at": meta.get("created_at"),
            }
        )
    return items


def format_memories(memories: Iterable[dict]) -> str:
    parts: list[str] = []
    for m in memories:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        score = m.get("score")
        created_at = m.get("created_at")
        parts.append(
            f"- (score={score:.3f} time={created_at}) {text}"
            if isinstance(score, (int, float))
            else f"- {text}"
        )
    return "\n".join(parts) if parts else "None."


def conversation_memory_namespace(session_id: str | None = None) -> str:
    """已嵌入对话历史的命名空间。"""
    sid = (session_id or "all").strip() or "all"
    return f"conversation_memory:{sid}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_conversation_jsonl(path: Path) -> list[dict]:
    """将对话 JSONL 解析为 user/assistant 的轮次对。"""
    turns: list[dict] = []
    if not path.exists():
        return turns
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return turns

    pending_user: str | None = None
    for idx, line in enumerate(lines):
        line = (line or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        role = str(obj.get("role") or "").strip().lower()
        content = str(obj.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
            continue
        if role == "assistant":
            if pending_user is None:
                continue
            turns.append(
                {
                    "turn_index": len(turns),
                    "line_index": idx,
                    "user": pending_user,
                    "assistant": content,
                }
            )
            pending_user = None
    return turns


def _turn_to_text(session_id: str, turn: dict) -> str:
    return (
        f"session={session_id}\n"
        f"turn={turn.get('turn_index')}\n"
        f"user: {turn.get('user')}\n"
        f"assistant: {turn.get('assistant')}"
    )


def embed_conversation_history(
    *,
    session_id: str,
    persist_dir: str = "data/conversations",
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    incremental: bool = True,
    force_rebuild: bool = False,
    manifest_path: str = "data/conversations/embed_manifest.json",
) -> dict:
    """将某个 session 的对话历史嵌入到向量库。"""
    safe_session = (session_id or "").strip()
    if not safe_session:
        return {"session_id": session_id, "embedded_turns": 0, "skipped": True, "reason": "empty_session_id"}

    conv_file = Path(persist_dir) / f"{safe_session}.jsonl"
    if not conv_file.exists():
        return {"session_id": safe_session, "embedded_turns": 0, "skipped": True, "reason": "session_file_not_found"}

    mpath = Path(manifest_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    except Exception:
        manifest = {}
    key = str(conv_file.resolve())
    old = manifest.get(key) if isinstance(manifest.get(key), dict) else {}
    file_hash = _sha256_file(conv_file)
    ns = conversation_memory_namespace(safe_session)
    unchanged = (
        old.get("sha256") == file_hash
        and old.get("namespace") == ns
        and int(old.get("chunk_size", -1)) == int(chunk_size)
        and int(old.get("chunk_overlap", -1)) == int(chunk_overlap)
    )
    if incremental and (not force_rebuild) and unchanged:
        return {"session_id": safe_session, "embedded_turns": 0, "skipped": True, "reason": "unchanged"}

    turns = _parse_conversation_jsonl(conv_file)
    vector_store.clear_namespace(ns)
    embedded = 0
    for t in turns:
        text = _turn_to_text(safe_session, t)
        embedded += vector_store.embed_document(
            text=text,
            namespace=ns,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata={
                "type": "conversation_turn",
                "session_id": safe_session,
                "turn_index": t.get("turn_index"),
                "source": str(conv_file.as_posix()),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    manifest[key] = {
        "sha256": file_hash,
        "namespace": ns,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "session_id": safe_session,
        "namespace": ns,
        "embedded_turns": len(turns),
        "embedded_chunks": embedded,
        "skipped": False,
    }


def embed_all_conversations(
    *,
    persist_dir: str = "data/conversations",
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    incremental: bool = True,
    force_rebuild: bool = False,
    manifest_path: str = "data/conversations/embed_manifest.json",
) -> dict:
    """将所有会话的 JSONL 文件嵌入到向量库。"""
    root = Path(persist_dir)
    if not root.exists():
        return {"sessions_total": 0, "embedded_sessions": 0, "embedded_turns": 0, "embedded_chunks": 0, "skipped_sessions": 0}
    files = sorted(root.glob("*.jsonl"))
    embedded_sessions = 0
    embedded_turns = 0
    embedded_chunks = 0
    skipped_sessions = 0
    failures: list[dict] = []
    for p in files:
        sid = p.stem
        try:
            r = embed_conversation_history(
                session_id=sid,
                persist_dir=persist_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                incremental=incremental,
                force_rebuild=force_rebuild,
                manifest_path=manifest_path,
            )
            if r.get("skipped"):
                skipped_sessions += 1
            else:
                embedded_sessions += 1
                embedded_turns += int(r.get("embedded_turns") or 0)
                embedded_chunks += int(r.get("embedded_chunks") or 0)
        except Exception as e:
            failures.append({"session_id": sid, "error": str(e)})
    return {
        "sessions_total": len(files),
        "embedded_sessions": embedded_sessions,
        "embedded_turns": embedded_turns,
        "embedded_chunks": embedded_chunks,
        "skipped_sessions": skipped_sessions,
        "failures": failures,
    }


def retrieve_conversation_memories(
    *,
    query: str,
    session_id: str | None = None,
    k: int = 6,
    score_threshold: float = 0.8,
    strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
) -> list[dict]:
    """在已嵌入的对话历史上进行语义检索。"""
    ns = conversation_memory_namespace(session_id)
    queries = expand_retrieval_queries(
        query,
        strategy=strategy,
        llm=qwen,
    )
    if not queries:
        fb = (query or "").strip()
        queries = [fb] if fb else []
    docs_scores = vector_store.retrieve(
        queries=queries,
        namespace=ns,
        k=k,
        score_threshold=score_threshold,
        strategy=strategy,
    )
    items: list[dict] = []
    for doc, score in docs_scores:
        meta = getattr(doc, "metadata", {}) or {}
        items.append(
            {
                "text": getattr(doc, "page_content", str(doc)),
                "score": float(score),
                "session_id": meta.get("session_id"),
                "turn_index": meta.get("turn_index"),
                "source": meta.get("source"),
                "created_at": meta.get("created_at"),
            }
        )
    return items

