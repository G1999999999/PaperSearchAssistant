"""
重排序模块：对初检结果用 CrossEncoder 精排，提升 Top-K 相关性。

面试点：两阶段检索（retrieve then rerank）、CrossEncoder vs BiEncoder。
"""

from __future__ import annotations

from typing import List, Tuple

from config import (
    RAG_ENABLE_RERANK,
    RAG_RERANK_BATCH_SIZE,
    RAG_RERANK_DEVICE,
    RAG_RERANK_EAGER_LOAD,
    RAG_RERANK_MODEL,
    RAG_RERANK_WARMUP,
)

_RERANKER = None
_RERANK_MODEL_LOADED: str | None = None


def _get_reranker():
    """懒加载 CrossEncoder，避免启动时拖慢。"""
    global _RERANKER, _RERANK_MODEL_LOADED
    if not RAG_ENABLE_RERANK:
        return None
    if _RERANKER is None or _RERANK_MODEL_LOADED != RAG_RERANK_MODEL:
        try:
            from sentence_transformers import CrossEncoder

            # 模型名来自 config.RAG_RERANK_MODEL（默认 config.RERANK_CROSS_ENCODER_MODEL）。
            _RERANKER = CrossEncoder(
                RAG_RERANK_MODEL,
                max_length=512,
                device=RAG_RERANK_DEVICE,
            )
            _RERANK_MODEL_LOADED = RAG_RERANK_MODEL
            if RAG_RERANK_WARMUP:
                try:
                    _RERANKER.predict(
                        [["warmup", "warmup"]],
                        batch_size=1,
                        show_progress_bar=False,
                    )
                except Exception:
                    pass
        except Exception:
            _RERANKER = False  # 表示未安装或加载失败
            _RERANK_MODEL_LOADED = None
    return _RERANKER if _RERANKER else None


if RAG_RERANK_EAGER_LOAD and RAG_ENABLE_RERANK:
    try:
        _get_reranker()
    except Exception:
        pass


def build_rerank_doc_text(doc: object) -> str:
    """论文检索计划推荐格式：Paper / Section / Role + Content。"""
    meta = getattr(doc, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    paper_t = meta.get("title") or meta.get("name") or meta.get("arxiv_id") or meta.get("source") or ""
    section = meta.get("section_title") or meta.get("heading") or ""
    role = meta.get("chunk_role") or meta.get("type") or ""
    body = getattr(doc, "page_content", str(doc)) or ""
    return (
        f"Paper: {paper_t}\nSection: {section}\nRole: {role}\nContent: {body[:8000]}"
    )


def rerank_with_evidence_metadata(
    query: str,
    doc_score_pairs: List[Tuple[object, float]],
    top_k: int = 6,
) -> List[Tuple[object, float]]:
    """与 `rerank` 相同，但 CrossEncoder 输入使用结构化证据文本。"""
    if not doc_score_pairs:
        return []
    encoder = _get_reranker()
    if encoder is None:
        return list(doc_score_pairs)[:top_k]
    docs = [p[0] for p in doc_score_pairs]
    texts = [build_rerank_doc_text(d) for d in docs]
    pairs = [[query, t] for t in texts]
    try:
        scores = encoder.predict(
            pairs,
            batch_size=RAG_RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception:
        return list(doc_score_pairs)[:top_k]
    indexed = list(zip(docs, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in indexed[:top_k]]


def rerank(
    query: str,
    doc_score_pairs: List[Tuple[object, float]],
    top_k: int = 6,
) -> List[Tuple[object, float]]:
    """对 (doc, semantic_score) 列表用 CrossEncoder 重排，返回 top_k 个 (doc, rerank_score)。

    - doc 需有 page_content 或可转为字符串。
    - 若 RAG_ENABLE_RERANK=0，或未安装 sentence_transformers，或加载出错，则按原顺序截断返回。
    """
    if not doc_score_pairs:
        return []
    encoder = _get_reranker()
    if encoder is None:
        return list(doc_score_pairs)[:top_k]

    docs = [p[0] for p in doc_score_pairs]
    texts = []
    for d in docs:
        if hasattr(d, "page_content"):
            texts.append(d.page_content)
        else:
            texts.append(str(d))
    pairs = [[query, t] for t in texts]
    try:
        scores = encoder.predict(
            pairs,
            batch_size=RAG_RERANK_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception:
        return list(doc_score_pairs)[:top_k]

    # 分数越高越相关；这里将原“距离型”得分（越小越相关）转换为一致的重排分数
    indexed = list(zip(docs, scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in indexed[:top_k]]
