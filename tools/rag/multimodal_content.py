"""
OpenAI 兼容多模态消息拼装：本地图片 → data URL，user content 多段列表。
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def image_file_to_data_url(path: Path) -> str | None:
    """读取本地图片文件，返回 data:image/...;base64,... 。失败返回 None。"""
    try:
        p = path.expanduser().resolve()
        if not p.is_file():
            return None
        raw = p.read_bytes()
        if not raw:
            return None
        mime, _ = mimetypes.guess_type(str(p))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        b64 = base64.standard_b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except OSError:
        return None


def build_openai_multimodal_user_content(
    text: str,
    image_paths: Sequence[Path | str],
    *,
    max_images: int,
) -> str | list[dict[str, Any]]:
    """返回纯 str（无图）或 OpenAI 风格多段 content 列表。"""
    paths: list[Path] = []
    seen: set[str] = set()
    for p in image_paths:
        ps = str(p).strip()
        if not ps or ps in seen:
            continue
        seen.add(ps)
        paths.append(Path(ps))
        if len(paths) >= max_images:
            break

    parts: list[dict[str, Any]] = [{"type": "text", "text": text or ""}]
    for p in paths:
        url = image_file_to_data_url(p)
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if len(parts) == 1:
        return text or ""
    return parts


def dict_messages_to_lc(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """将 build_rag_prompt 等返回的 dict 列表转为 LangChain BaseMessage。"""
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=str(content)))
        elif role == "assistant":
            out.append(AIMessage(content=str(content)))
        else:
            if isinstance(content, list):
                out.append(HumanMessage(content=content))
            else:
                out.append(HumanMessage(content=str(content)))
    return out
