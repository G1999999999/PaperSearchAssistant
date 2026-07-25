"""
查询与语言相关策略。

这里实现几个典型的 query strategy：
0. 可选：LLM Query 改写（口语 → 检索友好查询，见 prompts.QUERY_REWRITE_SYSTEM_PROMPT）
1. 归一化（去空白、小写等）
2. 简单同义词扩展（rule-based 的 fake query expansion）
3. 过长查询的截断/摘要（用规则近似）
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional


_SYNONYM_TABLE = {
    "llm": ["large language model", "大语言模型"],
    "rag": ["retrieval augmented generation", "检索增强生成"],
}


def normalize_query(query: str) -> str:
    """简单归一化：去掉首尾空白，并统一为小写。

    真实场景下可以加上：
    - 标点规范化
    - 全角/半角转换等
    """

    return " ".join(query.strip().split()).lower()


def expand_query_with_synonyms(query: str) -> str:
    """基于一个很小的同义词表做 query expansion。

    实际效果有限，但足以在面试时说明“扩展召回范围”的思路。
    """

    expanded = query
    for key, syns in _SYNONYM_TABLE.items():
        if key in query.lower():
            expanded += " " + " ".join(syns)
    return expanded


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        s = (x or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _dedupe_subquestions(subs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for s in subs:
        t = (s or "").strip()
        if not t:
            continue
        key = " ".join(t.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _parse_json_string_array(text: str) -> Optional[List[str]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out: List[str] = []
    for x in data:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out or None


def split_compound_question(
    raw: str,
    llm: Any,
    *,
    max_subquestions: Optional[int] = None,
    force: bool = False,
) -> List[str]:
    """用 LLM 将一次输入中的多个问题拆成子问题列表；失败或未开启时返回单元素列表。

    force=True 时忽略 RAG_SUBQUESTION_SPLIT，始终调用 LLM（供 preview / 调试）。
    """

    from config import RAG_MAX_SUBQUESTIONS, RAG_SUBQUESTION_SPLIT

    q = (raw or "").strip()
    if not q:
        return []
    if not force and not RAG_SUBQUESTION_SPLIT:
        return [q]

    cap = max_subquestions if max_subquestions is not None else RAG_MAX_SUBQUESTIONS

    from prompts import SUBQUESTION_SPLIT_SYSTEM_PROMPT

    system = SUBQUESTION_SPLIT_SYSTEM_PROMPT.format(max_subquestions=cap)
    try:
        model = llm.bind(temperature=0.2) if hasattr(llm, "bind") else llm
        resp = model.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ]
        )
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        parsed = _parse_json_string_array(text)
        if not parsed:
            return [q]
        deduped = _dedupe_subquestions(parsed)[:cap]
        return deduped if deduped else [q]
    except Exception:
        return [q]


def expand_retrieval_queries(
    raw_question: str,
    *,
    strategy: str,
    llm: Any,
    use_llm_rewrite: Optional[bool] = None,
    use_subquestion_split: Optional[bool] = None,
) -> List[str]:
    """子问题拆分（可选）+ 每条子问题的 build_search_queries，去重后供向量库 retrieve。"""

    from config import RAG_SUBQUESTION_SPLIT

    q = (raw_question or "").strip()
    if not q:
        return []

    split_on = use_subquestion_split if use_subquestion_split is not None else RAG_SUBQUESTION_SPLIT
    if split_on:
        subqs = split_compound_question(q, llm)
    else:
        subqs = [q]

    queries: List[str] = []
    for sq in subqs:
        queries.extend(
            build_search_queries(sq, use_llm_rewrite=use_llm_rewrite, llm=llm)
        )
    return _dedupe_preserve_order(queries)


def rewrite_query_for_retrieval(raw_query: str, llm: Any) -> str:
    """用对话模型将自然语言问题改写成更适合检索的短查询；失败则回退原文。"""

    q = (raw_query or "").strip()
    if not q:
        return q

    from prompts import QUERY_REWRITE_SYSTEM_PROMPT

    try:
        resp = llm.invoke(
            [
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
        )
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        line = text.splitlines()[0].strip() if text else ""
        # 防止模型输出过长或空
        if line and len(line) <= 800:
            return line
    except Exception:
        pass
    return q


def summarize_query_if_too_long(query: str, max_tokens: int = 64) -> str:
    """当查询过长时，用简单规则“摘要”一下。

    这里用非常粗糙的按词数截断，主要目的是讲清楚：
    - 在发起检索前可以先做一次 query 压缩，控制 token 长度。
    """

    words = query.split()
    if len(words) <= max_tokens:
        return query
    return " ".join(words[:max_tokens])


def build_search_queries(
    raw_query: str,
    *,
    use_llm_rewrite: Optional[bool] = None,
    llm: Any = None,
) -> List[str]:
    """组合以上策略，生成一个或多个检索用 query。

    返回的列表可以用于 multi-query 检索策略：
    - 第一个元素通常是 baseline（normalized+截断）
    - 第二个元素可以是扩展后的版本

    use_llm_rewrite:
        - None：读取 config.RAG_LLM_QUERY_REWRITE（可用环境变量 RAG_LLM_QUERY_REWRITE 开启）
        - True/False：强制开/关 LLM 改写
    llm:
        - 为 None 且需要改写时，使用 models_qwen.qwen
    """

    from config import RAG_LLM_QUERY_REWRITE

    if use_llm_rewrite is None:
        use_llm_rewrite = RAG_LLM_QUERY_REWRITE

    base = (raw_query or "").strip()
    if use_llm_rewrite and base:
        _llm = llm
        if _llm is None:
            from models_qwen import qwen as default_llm

            _llm = default_llm
        base = rewrite_query_for_retrieval(base, _llm)

    normalized = normalize_query(base)
    truncated = summarize_query_if_too_long(normalized)
    expanded = expand_query_with_synonyms(truncated)

    queries: List[str] = [truncated]
    if expanded != truncated:
        queries.append(expanded)
    return queries

