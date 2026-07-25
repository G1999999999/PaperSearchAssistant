"""
关键词检索（BM25）与混合检索融合（RRF）。

面试点：稀疏检索 vs 稠密检索、Reciprocal Rank Fusion 融合多路排序。
"""

from __future__ import annotations

import re
from typing import Callable, List, Tuple

from config import RRF_K


def default_tokenizer(text: str) -> List[str]:
    """简单分词：按空白切分，并保留连续字母数字；中文按单字切便于 BM25 匹配。"""
    # 英文/数字成词，其余（含中文）按字符
    tokens = []
    for part in re.split(r"\s+", text.strip()):
        if not part:
            continue
        if re.match(r"^[a-zA-Z0-9]+$", part):
            tokens.append(part.lower())
        else:
            tokens.extend(list(part))
    return [t for t in tokens if t]


def build_bm25_index(
    documents: List[object],
    tokenizer: Callable[[str], List[str]] | None = None,
):
    """为文档列表构建 BM25 索引。documents 需有 page_content 或可 str()。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, []

    tok = tokenizer or default_tokenizer
    texts = []
    for d in documents:
        if hasattr(d, "page_content"):
            texts.append(d.page_content)
        else:
            texts.append(str(d))
    corpus = [tok(t) for t in texts]
    if not corpus:
        return None, []
    bm25 = BM25Okapi(corpus)
    return bm25, list(documents)


def bm25_top_k(
    bm25,
    doc_list: List[object],
    query: str,
    k: int,
    tokenizer: Callable[[str], List[str]] | None = None,
) -> List[Tuple[object, float]]:
    """BM25 检索：返回 (doc, score) 列表，score 越大越相关。"""
    if bm25 is None or not doc_list:
        return []
    tok = tokenizer or default_tokenizer
    q_tokens = tok(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    indexed = list(zip(doc_list, scores, strict=False))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(doc, float(s)) for doc, s in indexed[:k]]


def rrf_fusion(
    ranked_lists: List[List[Tuple[object, float]]],
    k: int = RRF_K,
) -> List[Tuple[object, float]]:
    """倒数名次融合（Reciprocal Rank Fusion / RRF）：多路排序列表融合，按 RRF 分数排序。

    每路列表中 rank 从 1 开始，RRF_score(doc) = sum 1/(k + rank_i)。
    同一 doc 按内容去重（用 page_content 或 str(doc) 作为 key）。
    """
    if not ranked_lists:
        return []

    def doc_key(d):
        if hasattr(d, "page_content"):
            return d.page_content
        return str(d)

    rrf_scores: dict = {}
    for lst in ranked_lists:
        for rank, (doc, _) in enumerate(lst, start=1):
            key = doc_key(doc)
            inc = 1.0 / (k + rank)
            if key not in rrf_scores:
                rrf_scores[key] = (doc, inc)
            else:
                rrf_scores[key] = (rrf_scores[key][0], rrf_scores[key][1] + inc)

    out = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
    return list(out)
