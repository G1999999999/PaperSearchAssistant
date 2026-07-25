"""
时间相关工具。

在 RAG 场景中，可以用时间维度做：
- 写入文档创建时间，方便后续过滤“最近的内容”
- 按时间窗口过滤检索结果
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def add_timestamp_metadata(metadata: Optional[dict] = None) -> dict:
    data = dict(metadata or {})
    if "created_at" not in data:
        data["created_at"] = to_iso(now_utc())
    return data


def filter_by_time(
    results: Iterable[Tuple[T, float]],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[Tuple[T, float]]:
    """基于 metadata['created_at'] 对结果做时间过滤。

    这里假设 item 有 `metadata` 属性（和 LangChain Document 一致），
    在面试时可以说明真实项目里会用更严格的类型系统。
    """

    filtered: List[Tuple[T, float]] = []
    for item, score in results:
        try:
            created_at_str = getattr(item, "metadata", {}).get("created_at")
            if not created_at_str:
                filtered.append((item, score))
                continue
            created_at = datetime.fromisoformat(created_at_str)
            if since and created_at < since:
                continue
            if until and created_at > until:
                continue
            filtered.append((item, score))
        except Exception:
            # 如果解析失败，就保留该结果，避免因为时间问题把结果全过滤掉。
            filtered.append((item, score))
    return filtered

