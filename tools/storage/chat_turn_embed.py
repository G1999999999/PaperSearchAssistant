"""
将单轮问答嵌入与 RAG 相同的 namespace，使后续本地检索可命中上一轮对话内容。
"""

from __future__ import annotations

from typing import Any

from config import DEFAULT_NAMESPACE
from tools.rag.knowledge import vector_store


def embed_chat_turn_into_rag_namespace(
    *,
    namespace: str | None,
    question: str,
    answer: str,
    citations: list[dict[str, Any]] | None = None,
    source: str = "chat_turn",
) -> None:
    """把本轮 Q&A 写入向量库 `namespace`（默认同 CLI / Agent 的 --namespace）。"""

    rag_namespace = (namespace or "").strip() or DEFAULT_NAMESPACE
    text = f"User Question:\n{question}\n\nAssistant Answer:\n{answer}"
    cite_sources = [
        str(c.get("source") or "")
        for c in (citations or [])
        if c.get("source")
    ]
    meta: dict[str, Any] = {
        "type": "chat_memory",
        "source": source,
    }
    if cite_sources:
        meta["citation_sources"] = "; ".join(cite_sources[:12])
    vector_store.embed_document(
        text=text,
        namespace=rag_namespace,
        extra_metadata=meta,
    )
