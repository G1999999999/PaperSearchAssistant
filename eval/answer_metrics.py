from __future__ import annotations


def keyword_recall(answer: str, must_contain: list[str], casefold: bool = True) -> tuple[float, list[str]]:
    """粗粒度答案评测：必须出现的关键短语是否命中。

    返回 (命中比例, 未命中的短语列表)。不进行同义归一，适合集成测试或人工对齐后的关键词表。
    """
    if not must_contain:
        return 1.0, []
    text = (answer or "").strip()
    if casefold:
        text = text.casefold()
    missing: list[str] = []
    for phrase in must_contain:
        p = phrase.strip()
        if casefold:
            p = p.casefold()
        if p and p not in text:
            missing.append(phrase.strip())
    hit = len(must_contain) - len(missing)
    return hit / len(must_contain), missing
