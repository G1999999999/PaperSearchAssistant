from __future__ import annotations

from typing import Any

from tools.rag.math_utils import merge_ranked_lists
from tools.retrieval.context_packing import diversity_pack_by_parent


def expand_and_pack_context(
    store: Any,
    *,
    namespace: str,
    base_hits: list[tuple[object, float]],
    extra_hits: list[tuple[object, float]] | None = None,
    neighbor_window: int = 1,
    top_k: int = 10,
    max_per_parent: int = 3,
) -> list[tuple[object, float]]:
    merged = list(base_hits or [])
    if extra_hits:
        merged = merge_ranked_lists([merged, list(extra_hits)])
    try:
        if neighbor_window > 0:
            merged = store.expand_neighbor_chunks(
                retrieved=merged,
                namespace=namespace,
                window=int(neighbor_window),
            )
    except Exception:
        pass
    return diversity_pack_by_parent(
        merged,
        top_k=max(6, int(top_k)),
        max_per_parent=max(1, int(max_per_parent)),
    )

