"""
用 LLM 评判「本地检索到的片段是否足以回答用户问题」；
若不充分，返回建议的联网搜索查询，供 agent 补充网络上下文。
"""

from __future__ import annotations

import json
import re
from typing import Any


_RETRIEVAL_JUDGE_SYSTEM = """你是「检索充分性」评判器。你会看到用户问题和从向量知识库检索到的若干片段（可能不完整或偏离问题）。

任务：判断这些片段加在一起，是否足以让用户得到**准确、完整**的回答。

只输出一个 JSON 对象，不要 markdown 代码块，不要其它文字。字段：
- "score": 整数 0-10。10=完全足以回答；7-9=基本可答有小缺；4-6=明显缺关键信息；0-3=基本无关或严重不够
- "sufficient": 布尔。score>=7 且用户核心问题可被片段覆盖时为 true，否则 false
- "reason": 字符串，一两句中文说明判断依据
- "web_queries": 字符串数组，1-3 条。若 sufficient 为 true 则必须为空数组 []；若不充分，给出适合搜索引擎的短查询（可中英混合），用于补充缺失信息

示例（不充分）：
{"score":4,"sufficient":false,"reason":"只有泛泛介绍，缺少与问题直接相关的对比与细节","web_queries":["Python C++ 内存管理 区别","Python vs C++ 应用场景"]}

示例（充分）：
{"score":8,"sufficient":true,"reason":"片段覆盖了定义与主要区别","web_queries":[]}
""".strip()


def _parse_judge_json(text: str) -> dict[str, Any] | None:
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
    if not isinstance(data, dict):
        return None
    return data


def judge_retrieval_context(
    llm: Any,
    question: str,
    grouped: list[tuple[Any, float]],
    *,
    score_min: float = 6.0,
) -> dict[str, Any]:
    """返回 score、sufficient、reason、web_queries、should_supplement_web。"""

    q = (question or "").strip()
    default_out: dict[str, Any] = {
        "score": 10.0,
        "sufficient": True,
        "reason": "评判跳过或失败，使用仅本地上下文",
        "web_queries": [],
        "should_supplement_web": False,
    }

    if not q or not grouped:
        return default_out

    parts: list[str] = []
    for doc, sc in grouped[:14]:
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "unknown")
        txt = (getattr(doc, "page_content", "") or "").strip()[:700]
        if txt:
            parts.append(f"[来源: {src}, 相关度: {float(sc):.3f}]\n{txt}")

    if not parts:
        return default_out

    passages = "\n\n---\n\n".join(parts)
    user_msg = f"用户问题：{q}\n\n检索到的片段：\n\n{passages}"

    try:
        model = llm.bind(temperature=0.1) if hasattr(llm, "bind") else llm
        resp = model.invoke(
            [
                {"role": "system", "content": _RETRIEVAL_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        data = _parse_judge_json(text)
        if not data:
            return default_out

        score_raw = data.get("score", 10)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 10.0
        score = max(0.0, min(10.0, score))

        sufficient = bool(data.get("sufficient", score >= 7))
        reason = str(data.get("reason") or "").strip() or "无"

        wq = data.get("web_queries")
        web_queries: list[str] = []
        if isinstance(wq, list):
            for x in wq[:4]:
                if isinstance(x, str) and x.strip():
                    web_queries.append(x.strip())
        web_queries = web_queries[:3]

        should = (score < score_min) or (not sufficient)
        if should and not web_queries:
            web_queries = [q]

        return {
            "score": score,
            "sufficient": sufficient,
            "reason": reason,
            "web_queries": web_queries if should else [],
            "should_supplement_web": should,
        }
    except Exception:
        return default_out
