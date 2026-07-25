"""
可选：对搜索结果的首条 URL 拉取 HTML 并抽取纯文本，补充 DDG 摘要（赛程页常只有导航摘要）。
"""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup


def fetch_url_plain_text(
    url: str,
    *,
    max_chars: int = 8000,
    timeout: float = 14.0,
) -> str | None:
    u = (url or "").strip()
    if not u.startswith("http") or "duckduckgo" in u.lower():
        return None
    try:
        r = requests.get(
            u,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; PaperSearchAssistant/1.0; +research)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        r.raise_for_status()
        if "html" not in r.headers.get("Content-Type", "").lower() and "<html" not in (
            r.text[:500] or ""
        ).lower():
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if len(text) < 80:
            return None
        return text[:max_chars]
    except Exception:
        return None
