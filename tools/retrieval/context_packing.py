from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def diversity_pack_by_parent(
    items: Sequence[Tuple[T, float]],
    *,
    top_k: int,
    max_per_parent: int = 3,
) -> List[Tuple[T, float]]:
    """控制同一 parent_id / source 父文档占比，减轻冗余。"""
    if not items:
        return []
    counts: dict[str, int] = {}
    out: list[Tuple[T, float]] = []
    for doc, score in items:
        meta = getattr(doc, "metadata", None) or {}
        pid = str(meta.get("parent_id") or meta.get("paper_id") or meta.get("source") or "_")
        c = counts.get(pid, 0)
        if c >= max_per_parent:
            continue
        counts[pid] = c + 1
        out.append((doc, score))
        if len(out) >= top_k:
            break
    return out
