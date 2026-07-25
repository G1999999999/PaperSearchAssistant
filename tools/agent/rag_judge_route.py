"""
是否在本轮 RAG 中执行「检索充分性评判 → 可能联网补充」的路由。

- 与 ``Route.RAG``、``RAG_WEB_FALLBACK`` 等组合使用；本模块只回答「要不要跑评判」。
"""

from __future__ import annotations

import json
import re
from typing import Any


_LOCAL_ONLY_PHRASES_ZH = (
    "不要联网",
    "勿联网",
    "禁止联网",
    "别联网",
    "无需联网",
    "不用联网",
    "不要上网",
    "勿上网",
    "仅本地",
    "只用本地",
    "只要本地",
    "仅依据库",
    "只根据库",
    "只根据文档",
    "仅根据文档",
    "只依据库内",
    "基于库内即可",
    "基于上文即可",
    "仅根据上文",
    "只看上文",
    "不要搜索互联网",
    "别搜网上",
)
_LOCAL_ONLY_PHRASES_EN = (
    "local only",
    "offline only",
    "do not search the web",
    "no web search",
    "without web search",
    "only from the documents",
    "only from the context",
)


def _rule_based_use_judge(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    ql = q.lower()
    if any(p in q for p in _LOCAL_ONLY_PHRASES_ZH):
        return False
    if any(p in ql for p in _LOCAL_ONLY_PHRASES_EN):
        return False
    return True


def _parse_router_json(text: str) -> dict[str, Any] | None:
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
    return data if isinstance(data, dict) else None


def _llm_use_judge(llm: Any, question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    system = (
        "你是 RAG 策略路由。判断：该用户问题是否值得在「私有向量库检索」之后，"
        "再跑一次「检索充分性评判」，并在不足时用互联网摘要补充？\n"
        "need_judge=true：通用知识/定义/新闻时效/库外事实等，私库可能不全。\n"
        "need_judge=false：明确只要本地文档、纯元问题、或极短无实质内容。\n"
        "只输出一个 JSON 对象，不要 markdown，例如："
        '{"need_judge": true, "reason": "通用概念"}'
    )
    try:
        model = llm.bind(temperature=0.0, max_tokens=120) if hasattr(llm, "bind") else llm
        resp = model.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": q[:2000]},
            ]
        )
        text = (resp.content if hasattr(resp, "content") else str(resp)) or ""
        data = _parse_router_json(text)
        if not data:
            return _rule_based_use_judge(question)
        v = data.get("need_judge")
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes"):
            return True
        if isinstance(v, str) and v.strip().lower() in ("false", "0", "no"):
            return False
        return _rule_based_use_judge(question)
    except Exception:
        return _rule_based_use_judge(question)


def retrieval_judge_enabled_for_question(
    question: str,
    *,
    llm: Any | None = None,
    use_llm_router: bool = False,
    user_forces_web: bool = False,
) -> bool:
    """
    :param user_forces_web: 已为 True 时用户将走「强制联网」分支，可跳过评判以省一次 LLM。
    """
    if user_forces_web:
        return False
    if not _rule_based_use_judge(question):
        return False
    if use_llm_router and llm is not None:
        return _llm_use_judge(llm, question)
    return True
