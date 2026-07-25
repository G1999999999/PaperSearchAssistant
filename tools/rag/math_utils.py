"""
与检索结果打分、合并相关的数学辅助函数。
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def apply_score_threshold(
    results: Sequence[Tuple[T, float]], threshold: float
) -> list[Tuple[T, float]]:
    """过滤掉得分大于阈值的结果。

    对于余弦距离，得分越小越相似；在这里简单地用“<= 阈值”作为保留条件。
    """

    return [(item, score) for item, score in results if score <= threshold]


def merge_ranked_lists(
    lists: Iterable[Sequence[Tuple[T, float]]],
) -> list[Tuple[T, float]]:
    """将多路排序列表按得分合并，并去重。

    - 输入：多个 (item, score) 列表（例如来自不同 query）
    - 逻辑：按最小 score 排序，同一 item 只保留得分最小的一次
    """

    def _stable_key(item: T) -> str:
        """构造稳定去重键，避免要求 item 必须可哈希（如 langchain Document）。"""
        # 先尝试可哈希的快速路径
        try:
            hash(item)
            return f"hash:{repr(item)}"
        except Exception:
            pass

        # LangChain 文档这类对象
        page_content = getattr(item, "page_content", None)
        metadata = getattr(item, "metadata", None)
        if isinstance(page_content, str):
            md = metadata if isinstance(metadata, dict) else {}
            source = str(md.get("source", ""))
            parent_id = str(md.get("parent_id", ""))
            chunk_idx = str(md.get("chunk_index", ""))
            h = hashlib.sha1(page_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
            return f"doc:{source}|{parent_id}|{chunk_idx}|{h}"

        # 兜底：使用类型名与 id 生成去重键
        return f"id:{id(item)}:{type(item).__name__}"

    best_scores: dict[str, Tuple[T, float]] = {}
    for lst in lists:
        for item, score in lst:
            key = _stable_key(item)
            prev = best_scores.get(key)
            if prev is None or score < prev[1]:
                best_scores[key] = (item, score)
    merged = list(best_scores.values())
    merged.sort(key=lambda x: x[1])
    return merged


def mmr_rerank(
    embedding: list[float],
    candidates: Sequence[Tuple[list[float], T, float]],
    lambda_mult: float,
    top_k: int,
) -> list[T]:
    """简单版 MMR（Maximal Marginal Relevance）实现。

    参数：
        embedding: 查询向量
        candidates: (doc_embedding, doc, score) 列表，score 是与查询的距离
        lambda_mult: 平衡相关性与多样性的系数，越大越偏向相关性
        top_k: 需要选出的文档数量

    这里为了简化，实现一个基于欧式距离的近似版本，重点是便于讲解思路。
    """

    if not candidates:
        return []

    import math

    def euclidean(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("inf")
        for idx in remaining:
            doc_emb, _doc, dist = candidates[idx]
            if not selected:
                mmr_score = dist
            else:
                diversity = min(
                    euclidean(doc_emb, candidates[s][0]) for s in selected
                )
                mmr_score = lambda_mult * dist - (1 - lambda_mult) * diversity
            if mmr_score < best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)  # type: ignore[arg-type]

    return [candidates[i][1] for i in selected]

