from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import (
    RAG_PAPER_ENABLE_CONTEXT_EXPANSION,
    RAG_PAPER_ENABLE_FIGURE_QA,
    RAG_PAPER_ENABLE_TABLE_QA,
    RAG_PAPER_FIGURE_TOP_K,
    RAG_PAPER_MULTIMODAL_RERANK_TOP_K,
    RAG_PAPER_TEXT_TOP_K,
)
from tools.rag.math_utils import merge_ranked_lists
from tools.rag.retrieval_merge import retrieve_with_public_merge
from tools.retrieval.context_assembler import assemble_multichannel_context
from tools.retrieval.figure_retriever import retrieve_figure_evidence
from tools.retrieval.table_retriever import retrieve_table_evidence
from tools.storage.repos.chunk_repo import (
    list_chunks_for_paper_on_pages,
    paper_id_from_rag_namespace,
)


def run_paper_content_multichannel_retrieval(
    store: Any,
    *,
    question: str,
    namespace: str,
    strategy: str,
    score_threshold: float,
    session_ingest_ids: list[str] | None,
    wants_table: bool,
    wants_figure: bool,
    final_top_k: int,
) -> list[tuple[object, float]]:
    """正文/表格/图片三通道召回 -> 融合 -> 统一重排 -> 上下文扩展。"""
    text_k = max(int(final_top_k), int(RAG_PAPER_TEXT_TOP_K))
    if wants_figure:
        # 图像问答常见“caption 不完整”，适度增加正文召回作为补证。
        text_k = max(text_k, int(final_top_k) * 2, 12)
    text_hits = retrieve_with_public_merge(
        store,
        queries=[question],
        namespace=namespace,
        k=text_k,
        score_threshold=score_threshold,
        strategy=strategy,
        session_ingest_ids=session_ingest_ids,
        extra_chroma_filter=None,
    )

    table_hits: list[tuple[object, float]] = []
    figure_hits: list[tuple[object, float]] = []
    if RAG_PAPER_ENABLE_TABLE_QA and wants_table:
        table_hits = retrieve_table_evidence(
            store,
            question=question,
            namespace=namespace,
            strategy=strategy,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids,
        )
    if RAG_PAPER_ENABLE_FIGURE_QA and wants_figure:
        fig_k = max(int(RAG_PAPER_FIGURE_TOP_K), int(final_top_k) + 4, 8)
        figure_hits = retrieve_figure_evidence(
            store,
            question=question,
            namespace=namespace,
            strategy=strategy,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids,
            top_k=fig_k,
        )

    # 插图所在页的正文 chunk（PG）并入 text 通道，便于「图 + 同页文字」联合回答
    if wants_figure and figure_hits:
        pid = paper_id_from_rag_namespace(namespace)
        if pid:
            pages: list[int] = []
            for doc, _sc in figure_hits:
                md = getattr(doc, "metadata", None) or {}
                if not isinstance(md, dict):
                    continue
                pg = md.get("page")
                if pg is None:
                    continue
                try:
                    pages.append(int(pg))
                except (TypeError, ValueError):
                    pass
            if pages:
                rows = list_chunks_for_paper_on_pages(
                    pid,
                    pages,
                    exclude_roles=frozenset({"figure", "table"}),
                    limit_total=28,
                )
                same_hits: list[tuple[object, float]] = []
                for i, row in enumerate(rows):
                    txt = (row.get("content") or "").strip()
                    if not txt:
                        continue
                    doc_pg = Document(
                        page_content=txt[:12000],
                        metadata={
                            "source": "postgresql_same_page_as_figure",
                            "paper_chunk_id": row.get("id"),
                            "chunk_role": row.get("chunk_role"),
                            "page_from": row.get("page_from"),
                            "page_to": row.get("page_to"),
                            "type": "same_page_text",
                        },
                    )
                    same_hits.append((doc_pg, 0.05 + i * 0.0005))
                if same_hits:
                    text_hits = merge_ranked_lists([same_hits, text_hits])

    tw, tbw, fw = 1.0, 1.0, 1.0
    if wants_table and not wants_figure:
        tw, tbw, fw = 0.95, 1.35, 0.7
    elif wants_figure and not wants_table:
        tw, tbw, fw = 0.95, 0.7, 1.35
    elif wants_figure and wants_table:
        tw, tbw, fw = 1.0, 1.2, 1.2

    merged = assemble_multichannel_context(
        text_hits=text_hits,
        table_hits=table_hits,
        figure_hits=figure_hits,
        text_weight=tw,
        table_weight=tbw,
        figure_weight=fw,
        top_k=max(int(final_top_k), int(RAG_PAPER_MULTIMODAL_RERANK_TOP_K)),
    )

    try:
        from tools.rag.rerank import rerank_with_evidence_metadata

        merged = rerank_with_evidence_metadata(
            question,
            merged,
            top_k=max(int(final_top_k), int(RAG_PAPER_MULTIMODAL_RERANK_TOP_K)),
        )
    except Exception:
        merged = merged[: max(int(final_top_k), int(RAG_PAPER_MULTIMODAL_RERANK_TOP_K))]

    if RAG_PAPER_ENABLE_CONTEXT_EXPANSION:
        try:
            merged = store.expand_neighbor_chunks(merged, namespace=namespace, window=1)
        except Exception:
            pass

    return merged[: int(final_top_k)]

