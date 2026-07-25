"""本地论文问答优先：与 agent 共用的轻量判断，避免 router ↔ agent 循环导入。"""

from __future__ import annotations

import re


def looks_like_paper_content_qa(question: str) -> bool:
    """用户是在问某篇论文的内容，而非「找/列多篇论文」。"""
    q = (question or "").strip()
    if not q:
        return False
    ql = q.lower()
    if any(
        k in q
        for k in (
            "从本地库",
            "从本地",
            "本地库",
            "仅在本地",
            "仅本地",
            "只查本地",
            "仅从本地",
            "本地检索",
            "库里检索",
            "知识库",
            "向量库",
        )
    ):
        return True
    if any(
        k in q
        for k in (
            "说一说",
            "讲讲",
            "介绍",
            "解释",
            "描述",
            "概述",
            "总结",
            "归纳",
            "方法部分",
            "实验部分",
            "结论部分",
            "讲了什么",
            "说了什么",
            "什么意思",
            "如何理解",
            "如何实现",
            "怎么做",
            "原理",
            "核心思想",
            "创新点",
            "贡献",
            "依据",
            "根据这篇",
            "针对这篇",
            "关于这篇",
        )
    ):
        return True
    if re.search(
        r"(论文|paper).{0,12}(的|里|中).{0,8}(方法|实验|结论|摘要|贡献|结果|表|图|架构)",
        q,
        re.I,
    ):
        return True
    if re.search(
        r"(检索|查询|查找).{0,16}(说一说|介绍|解释|内容|方法|实验|结论)",
        q,
        re.I,
    ):
        return True
    if "这篇" in q and re.search(
        r"(吗|么|什么|如何|怎么|为什么|是否|请讲|详细|概括)", ql
    ):
        return True
    return False
