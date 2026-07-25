from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import (
    RAG_LAYERED_PAPER_RETRIEVAL,
    RAG_PAPER_BM25_TOP_K,
    RAG_PAPER_CONTEXT_EXPANSION_DEPTH,
    RAG_PAPER_ENABLE_CONTEXT_EXPANSION,
    RAG_PAPER_ENABLE_MULTI_QUERY,
    RAG_PAPER_ENABLE_MMR_PACKING,
    RAG_PAPER_ENABLE_QUERY_DECOMPOSITION,
    RAG_PAPER_ENABLE_RERANK,
    RAG_PAPER_METHOD_SECTION_RRF_WEIGHT,
    RAG_PAPER_RERANK_FETCH_K,
    RAG_PAPER_RERANK_TOP_K,
    RAG_PAPER_SEARCH_PROFILE,
    RAG_PAPER_VECTOR_TOP_K,
)
from tools.agent.router import extract_arxiv_id
from tools.rag.language import expand_retrieval_queries
from tools.rag.retrieval_merge import retrieve_with_public_merge
from tools.retrieval.context_expander import expand_and_pack_context
from tools.retrieval.fusion import (
    metadata_intent_boost,
    penalize_table_chunks_for_method_intent,
    weighted_rrf_fusion,
)
from tools.retrieval.missing_evidence_detector import analyze_missing_evidence
from tools.retrieval.paper_content_qa import run_paper_content_multichannel_retrieval
from tools.retrieval.query_router import build_query_route
from tools.retrieval.section_fallback import build_section_fallback_queries, run_section_fallback
from tools.rag.math_utils import merge_ranked_lists
from tools.retrieval.query_understanding import analyze_paper_query, subquestions_for_decomposition
from tools.storage.redis.section_cache import (
    get_section_roles,
    get_summary_bundle,
    set_section_roles,
    set_summary_bundle,
)
from tools.storage.repos.chunk_repo import list_chunks_by_section_ids
from tools.storage.repos.retrieval_repo import fts_search_chunks
from tools.storage.repos.section_repo import (
    get_paper_by_arxiv_id,
    list_sections_for_paper,
    section_ids_by_role,
)
from tools.storage.repos.summary_repo import get_paper_summary_bundle
from tools.agent.middleware import trace_event


def _profile_k(profile: str) -> tuple[int, int, int]:
    p = (profile or "balanced").strip().lower()
    if p == "fast":
        return 15, 15, 24
    if p == "deep" or p == "multi_stage_deep":
        return 50, 50, 80
    return (RAG_PAPER_VECTOR_TOP_K, RAG_PAPER_BM25_TOP_K, RAG_PAPER_RERANK_FETCH_K)


def _build_chroma_extra_filter(
    question: str, namespace: str
) -> dict[str, Any] | None:
    """窄化候选：论文 namespace 下若出现 arXiv id，则加上 metadata 过滤。"""
    if not (namespace or "").strip().startswith("paper:"):
        return None
    aid = extract_arxiv_id(question or "")
    if not aid:
        return None
    return {"arxiv_id": aid}


def _fts_documents(question: str, paper_id: int | None) -> list[tuple[object, float]]:
    rows = fts_search_chunks(question, paper_id=paper_id, limit=25)
    if not rows:
        q = (question or "").lower()
        probes: list[str] = []
        # 兜底：中文提问命中英文论文正文时，补充少量英文探针词做 PG-FTS。
        if any(k in q for k in ["方法", "method", "methodology", "approach", "怎么做"]):
            probes = [
                "method approach architecture training loss",
                "methodology implementation details",
            ]
        elif any(
            k in q
            for k in ["实验", "experiment", "evaluation", "结果", "result", "ablation"]
        ):
            probes = [
                "experiment evaluation results baseline metrics",
                "ablation study qualitative quantitative",
            ]
        elif any(k in q for k in ["总结", "概述", "讲了什么", "summary", "overview"]):
            probes = [
                "abstract introduction conclusion contributions",
            ]
        for pq in probes:
            rows = fts_search_chunks(pq, paper_id=paper_id, limit=25)
            if rows:
                trace_event(
                    "paper_fts_probe_fallback_used",
                    {"paper_id": int(paper_id or 0), "probe": pq[:120], "hits": len(rows)},
                )
                break
    if not rows:
        return []
    out: list[tuple[object, float]] = []
    for i, r in enumerate(rows):
        text = (r.get("content") or "")[:12000]
        doc = Document(
            page_content=text,
            metadata={
                "source": "postgresql_fts",
                "paper_chunk_id": r.get("id"),
                "chunk_role": r.get("chunk_role"),
                "paper_id": r.get("paper_id"),
                "chroma_doc_id": r.get("chroma_doc_id"),
            },
        )
        # 用排名构造伪距离，便于 merge_ranked_lists（越小越好）
        out.append((doc, float(i) * 0.01))
    return out


def _paper_id_from_namespace(namespace: str) -> int | None:
    ns = (namespace or "").strip()
    if not ns.startswith("paper:"):
        return None
    # namespace 形如 paper:<arxiv_id>:full，需仅提取中间 arXiv id
    parts = ns.split(":")
    arxiv_id = parts[1].strip() if len(parts) >= 2 else ""
    if not arxiv_id:
        return None
    row = get_paper_by_arxiv_id(arxiv_id)
    if not row:
        return None
    try:
        return int(row.get("id"))
    except Exception:
        return None


def _pg_section_docs(
    *,
    paper_id: int | None,
    role: str,
    limit: int = 80,
) -> list[tuple[object, float]]:
    if not paper_id:
        return []
    role_norm = (role or "").strip().lower()
    roles = get_section_roles(paper_id) or {}
    sec_ids = [int(x) for x in roles.get(role_norm, []) if str(x).strip()]
    if not sec_ids:
        sec_ids = section_ids_by_role(paper_id, role_norm)
        if sec_ids:
            roles[role_norm] = sec_ids
            set_section_roles(paper_id, roles)
    # 为缺证据检测（missing_evidence_detector）提供更可靠的 section 标题字段
    # 避免仅靠 chunk 里的 section_role_hint 导致 method_hits 总为 0。
    sec_title_map: dict[int, str] = {}
    try:
        for s in list_sections_for_paper(paper_id):
            sid = int(s.get("id") or 0)
            if sid <= 0:
                continue
            sec_title = str(s.get("title") or "").strip()
            if sec_title:
                sec_title_map[sid] = sec_title
    except Exception:
        sec_title_map = {}

    rows = list_chunks_by_section_ids(sec_ids, limit=limit)
    out: list[tuple[object, float]] = []
    for i, r in enumerate(rows):
        text = str(r.get("content") or "")[:12000]
        if not text:
            continue
        section_title = ""
        try:
            section_title = str(sec_title_map.get(int(r.get("section_id") or 0)) or "").strip()
        except Exception:
            section_title = ""
        doc = Document(
            page_content=text,
            metadata={
                "source": "postgresql_section_prefilter",
                "paper_chunk_id": r.get("id"),
                "section_id": r.get("section_id"),
                "chunk_role": r.get("chunk_role"),
                "paper_id": r.get("paper_id"),
                "chroma_doc_id": r.get("chroma_doc_id"),
                "section_role_hint": role_norm,
                "section_title": section_title,
                "heading": section_title,
            },
        )
        out.append((doc, float(i) * 0.01))
    return out


def _summary_bundle_docs(paper_id: int | None) -> list[tuple[object, float]]:
    if not paper_id:
        return []
    bundle = get_summary_bundle(paper_id)
    if not bundle:
        bundle = get_paper_summary_bundle(paper_id)
        if bundle:
            set_summary_bundle(paper_id, bundle)
    if not bundle:
        return []
    fields = [
        ("abstract", bundle.get("abstract_summary")),
        ("introduction", bundle.get("intro_summary")),
        ("method", bundle.get("method_summary")),
        ("result", bundle.get("result_summary")),
        ("conclusion", bundle.get("conclusion_summary")),
    ]
    out: list[tuple[object, float]] = []
    rank = 0
    for role, text in fields:
        body = (text or "").strip()
        if not body:
            continue
        out.append(
            (
                Document(
                    page_content=body,
                    metadata={
                        "source": "paper_summary_bundle",
                        "paper_id": paper_id,
                        "chunk_role": "summary",
                        "section_role_hint": role,
                    },
                ),
                float(rank) * 0.01,
            )
        )
        rank += 1
    return out


def _paper_abstract_fallback_docs(paper_id: int | None) -> list[tuple[object, float]]:
    """最后兜底：直接读取 papers.abstract/title，避免整条链路返回空上下文。"""
    if not paper_id:
        return []
    try:
        from sqlalchemy import select
        from tools.storage.sql.db import get_session_factory
        from tools.storage.sql.models import Paper
    except Exception:
        return []
    factory = get_session_factory()
    if factory is None:
        return []
    session = factory()
    try:
        row = session.scalar(select(Paper).where(Paper.id == int(paper_id)))
        if row is None:
            return []
        title = str(getattr(row, "title", "") or "").strip()
        abstract = str(getattr(row, "abstract", "") or "").strip()
        text = ""
        if title:
            text += f"Title: {title}\n\n"
        if abstract:
            text += f"Abstract: {abstract}"
        text = text.strip()
        if not text:
            return []
        return [
            (
                Document(
                    page_content=text[:12000],
                    metadata={
                        "source": "paper_meta_abstract_fallback",
                        "paper_id": int(paper_id),
                        "chunk_role": "summary",
                        "section_role_hint": "abstract",
                    },
                ),
                0.0,
            )
        ]
    except Exception:
        return []
    finally:
        session.close()


def layered_paper_retrieve(
    store: Any,
    *,
    question: str,
    namespace: str,
    strategy: str,
    k: int,
    score_threshold: float,
    session_ingest_ids: list[str] | None,
    llm: Any,
    use_layered: bool | None = None,
) -> list[tuple[object, float]]:
    """分层论文检索：查询理解 → 多路召回 → 加权 RRF → 证据重排 → 邻接扩展 → 多样性打包。"""
    if use_layered is None:
        use_layered = bool(RAG_LAYERED_PAPER_RETRIEVAL)
    if not use_layered:
        queries = expand_retrieval_queries(question, strategy=strategy, llm=llm)
        if not queries:
            fb = (question or "").strip()
            queries = [fb] if fb else []
        return retrieve_with_public_merge(
            store,
            queries=queries,
            namespace=namespace,
            k=k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=None,
        )

    vec_k, _bm_k, rerank_fetch = _profile_k(RAG_PAPER_SEARCH_PROFILE)
    merge_k = max(int(k), min(rerank_fetch, vec_k + 10))
    route = build_query_route(question)
    qu = analyze_paper_query(question)
    paper_id = _paper_id_from_namespace(namespace)
    if route.needs_table:
        qu.wants_table = True
    if route.needs_figure:
        qu.wants_figure = True
    extra_filter = _build_chroma_extra_filter(question, namespace)

    base_queries = expand_retrieval_queries(question, strategy=strategy, llm=llm)
    if not base_queries:
        base_queries = [(question or "").strip()] if (question or "").strip() else []

    subqs: list[str] = []
    if RAG_PAPER_ENABLE_QUERY_DECOMPOSITION:
        subqs = subquestions_for_decomposition(question)

    multi: list[str] = []
    if RAG_PAPER_ENABLE_MULTI_QUERY:
        multi = list(qu.extra_retrieval_queries)

    all_q: list[str] = []
    seen: set[str] = set()
    for q in list(base_queries) + subqs + multi:
        t = (q or "").strip()
        if t and t not in seen:
            seen.add(t)
            all_q.append(t)
    if not all_q:
        return []

    # 多模态问答主路径：正文+表格+图片三通道（方案文档要求）
    if qu.wants_table or qu.wants_figure:
        return run_paper_content_multichannel_retrieval(
            store,
            question=question,
            namespace=namespace,
            strategy=strategy,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids,
            wants_table=bool(qu.wants_table),
            wants_figure=bool(qu.wants_figure),
            final_top_k=max(int(k), RAG_PAPER_RERANK_TOP_K),
        )

    section_role = ""
    if qu.intent == "paper_method":
        section_role = "method"
    elif qu.intent == "paper_result":
        section_role = "result"
    elif qu.intent == "paper_conclusion":
        section_role = "conclusion"
    section_hits: list[tuple[object, float]] = []
    if section_role:
        section_hits = _pg_section_docs(
            paper_id=paper_id, role=section_role, limit=max(int(k) * 8, 60)
        )

    method_fallback_added = False
    if qu.intent == "paper_method" and not section_hits:
        qstrip = (question or "").strip()
        extras = [
            f"{qstrip} method section approach pipeline".strip(),
            "method section methodology implementation details Section 3 approach",
        ]
        for eq in extras:
            t = (eq or "").strip()
            if t and t not in seen:
                seen.add(t)
                all_q.append(t)
                method_fallback_added = True
        trace_event(
            "paper_method_no_pg_section_fallback_queries",
            {
                "namespace": namespace,
                "fallback_queries_added": bool(method_fallback_added),
            },
        )

    channels: list[tuple[list[tuple[object, float]], float]] = []

    if qu.intent == "paper_summary":
        bundle_hits = _summary_bundle_docs(paper_id)
        if bundle_hits:
            channels.append((bundle_hits, 1.2))

    for q in all_q:
        hits = retrieve_with_public_merge(
            store,
            queries=[q],
            namespace=namespace,
            k=merge_k,
            score_threshold=score_threshold,
            strategy=strategy,
            session_ingest_ids=session_ingest_ids,
            extra_chroma_filter=extra_filter,
        )
        channels.append((hits, 1.0))

    if section_hits:
        sec_w = (
            float(RAG_PAPER_METHOD_SECTION_RRF_WEIGHT)
            if section_role == "method"
            else 1.15
        )
        channels.append((section_hits, sec_w))
        trace_event(
            "paper_method_section_channel",
            {
                "section_role": section_role,
                "rrf_weight": sec_w,
                "section_docs": len(section_hits),
                "fallback_queries_added": bool(method_fallback_added),
            },
        )

    fts_hits = _fts_documents(question, paper_id=paper_id)
    if fts_hits:
        channels.append((fts_hits, 1.05))

    def _boost(doc: object) -> float:
        return metadata_intent_boost(
            doc,
            wants_table=qu.wants_table,
            wants_figure=qu.wants_figure,
            preferred_roles=qu.preferred_roles,
            intent=str(qu.intent or ""),
        )

    fused = weighted_rrf_fusion(channels, intent_boost_fn=_boost)
    fused.sort(key=lambda x: x[1])

    if RAG_PAPER_ENABLE_RERANK:
        try:
            from tools.rag.rerank import rerank_with_evidence_metadata

            rerank_in = fused[: max(12, min(RAG_PAPER_RERANK_FETCH_K, len(fused)))]
            reranked = rerank_with_evidence_metadata(
                question,
                rerank_in,
                top_k=max(int(k), RAG_PAPER_RERANK_TOP_K),
            )
            reranked = penalize_table_chunks_for_method_intent(
                reranked,
                intent=str(qu.intent or ""),
                wants_table=bool(qu.wants_table),
                scores_higher_is_better=True,
            )
        except Exception:
            reranked = fused[: max(int(k), RAG_PAPER_RERANK_TOP_K)]
            reranked = penalize_table_chunks_for_method_intent(
                reranked,
                intent=str(qu.intent or ""),
                wants_table=bool(qu.wants_table),
                scores_higher_is_better=False,
            )
    else:
        reranked = fused[: max(int(k), RAG_PAPER_RERANK_TOP_K)]
        reranked = penalize_table_chunks_for_method_intent(
            reranked,
            intent=str(qu.intent or ""),
            wants_table=bool(qu.wants_table),
            scores_higher_is_better=False,
        )

    # 精排后做缺失证据检测，再按需补全（方案：先粗排，再精排，再补全）。
    report = analyze_missing_evidence(
        question=question,
        reranked=reranked,
        intent=qu.intent,
        wants_table=bool(qu.wants_table),
        wants_figure=bool(qu.wants_figure),
    )
    fallback_qs = build_section_fallback_queries(
        question=question,
        intent=qu.intent,
        missing_method=bool(report.missing_method),
        missing_result=bool(report.missing_result),
        missing_conclusion=bool(report.missing_conclusion),
        abstract_truncated=bool(report.abstract_truncated),
    )
    fb_hits: list[tuple[object, float]] = []
    if report.needs_expansion and fallback_qs:
        fb_hits = run_section_fallback(
            store,
            queries=fallback_qs,
            namespace=namespace,
            strategy=strategy,
            score_threshold=score_threshold,
            session_ingest_ids=session_ingest_ids,
            k=max(int(k), 12),
            extra_chroma_filter=extra_filter,
        )
    trace_event(
        "paper_missing_evidence_analysis",
        {
            "intent": qu.intent,
            "wants_table": bool(qu.wants_table),
            "wants_figure": bool(qu.wants_figure),
            "thin_context": bool(report.thin_context),
            "abstract_truncated": bool(report.abstract_truncated),
            "missing_method": bool(report.missing_method),
            "missing_result": bool(report.missing_result),
            "missing_conclusion": bool(report.missing_conclusion),
            "needs_expansion": bool(report.needs_expansion),
            "fallback_queries_count": len(fallback_qs),
            "fallback_hits_count": len(fb_hits),
            "reranked_before_expand": len(reranked),
        },
    )

    win = int(RAG_PAPER_CONTEXT_EXPANSION_DEPTH) if RAG_PAPER_ENABLE_CONTEXT_EXPANSION else 0
    if RAG_PAPER_ENABLE_MMR_PACKING:
        reranked = expand_and_pack_context(
            store,
            namespace=namespace,
            base_hits=reranked,
            extra_hits=fb_hits,
            neighbor_window=max(1, win),
            top_k=max(int(k), 6),
            max_per_parent=2,
        )
    else:
        reranked = expand_and_pack_context(
            store,
            namespace=namespace,
            base_hits=reranked,
            extra_hits=fb_hits,
            neighbor_window=max(1, win),
            top_k=max(int(k), 8),
            max_per_parent=4,
        )
    trace_event(
        "paper_context_expansion_done",
        {
            "namespace": namespace,
            "final_hits_count": len(reranked),
            "neighbor_window": max(1, win),
            "mmr_packing": bool(RAG_PAPER_ENABLE_MMR_PACKING),
        },
    )

    # 兜底：paper namespace 下若向量检索为 0，直接回落到 PG summary bundle / section docs。
    if (not reranked) and str(namespace or "").startswith("paper:"):
        pg_fallback = _summary_bundle_docs(paper_id)
        if not pg_fallback:
            for role in ("method", "result", "conclusion"):
                sec_docs = _pg_section_docs(
                    paper_id=paper_id,
                    role=role,
                    limit=max(int(k) * 4, 24),
                )
                if sec_docs:
                    pg_fallback.extend(sec_docs[: max(int(k), 8)])
            # 去重并保持分数有序
            if pg_fallback:
                pg_fallback = merge_ranked_lists([pg_fallback])[: max(int(k), 8)]
        if not pg_fallback:
            pg_fallback = _paper_abstract_fallback_docs(paper_id)
        if pg_fallback:
            trace_event(
                "paper_pg_summary_fallback_used",
                {
                    "namespace": namespace,
                    "intent": str(qu.intent or ""),
                    "fallback_hits": len(pg_fallback),
                    "paper_id": int(paper_id or 0),
                },
            )
            return pg_fallback[: max(int(k), 8)]

    return reranked


def merge_session_and_paper_hits(
    session_hits: list[tuple[object, float]],
    paper_hits: list[tuple[object, float]],
    *,
    cap: int,
) -> list[tuple[object, float]]:
    return merge_ranked_lists([session_hits, paper_hits])[:cap]
