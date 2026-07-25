"""
跨 namespace 检索合并：在「当前分区」检索结果中并入「公共分区」的检索结果。

用于会话/论文等专用 namespace 仍可调阅公共 embed 资料（如 default）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, List, Tuple

from config import (
    DEFAULT_NAMESPACE,
    RAG_MERGE_PUBLIC_RETRIEVAL,
    RAG_PUBLIC_NAMESPACE,
)
from tools.rag.math_utils import merge_ranked_lists


def should_merge_public_retrieval(namespace: str | None) -> bool:
    """是否与公共 namespace 做二路合并。"""
    if not RAG_MERGE_PUBLIC_RETRIEVAL:
        return False
    cur = (namespace or "").strip() or DEFAULT_NAMESPACE
    pub = (RAG_PUBLIC_NAMESPACE or "").strip() or DEFAULT_NAMESPACE
    if not pub:
        return False
    # 已在查公共库，避免重复检索与重复融合
    return cur != pub


def retrieve_with_public_merge(
    store: Any,
    *,
    queries: Iterable[str],
    namespace: str,
    k: int,
    score_threshold: float,
    strategy: str,
    session_ingest_ids: list[str] | None = None,
    extra_chroma_filter: dict[str, Any] | None = None,
) -> List[Tuple[object, float]]:
    """先检索当前 namespace，必要时再检索 RAG_PUBLIC_NAMESPACE，按得分合并去重后截断为 top-k。

    session_ingest_ids 仅作用于主 namespace（本会话上传入库的 Chroma 过滤），不用于公共分区。
    """

    qlist = list(queries)
    ns = (namespace or "").strip()
    # 论文专用 namespace（paper:<id>:*）下：硬隔离，禁止并入公共库避免串入其他论文证据。
    if ns.startswith("paper:"):
        try:
            # trace_event 内部会基于 RAG_TRACE_ENABLED 自动降级为 no-op
            from tools.agent.middleware import trace_event

            trace_event(
                "paper_namespace_no_public_merge",
                {"namespace": ns, "k": int(k)},
            )
        except Exception:
            pass
        return store.retrieve(
            queries=qlist,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=extra_chroma_filter,
        )
    if not should_merge_public_retrieval(namespace):
        return store.retrieve(
            queries=qlist,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=extra_chroma_filter,
        )
    pub = (RAG_PUBLIC_NAMESPACE or "").strip() or DEFAULT_NAMESPACE

    def _primary() -> List[Tuple[object, float]]:
        return store.retrieve(
            queries=qlist,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=extra_chroma_filter,
        )

    def _public() -> List[Tuple[object, float]]:
        return store.retrieve(
            queries=qlist,
            namespace=pub,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=None,
            extra_chroma_filter=None,
        )

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_primary)
        f2 = ex.submit(_public)
        primary = f1.result()
        public_hits = f2.result()
    merged = merge_ranked_lists([primary, public_hits])
    return merged[: int(k)]
