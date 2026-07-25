from __future__ import annotations

from typing import List, Tuple

from config import RAG_PAPER_ENABLE_MULTIMODAL_PACKING
from tools.retrieval.context_packing import diversity_pack_by_parent
from tools.retrieval.fusion import weighted_rrf_fusion


def assemble_multichannel_context(
    *,
    text_hits: list[tuple[object, float]],
    table_hits: list[tuple[object, float]],
    figure_hits: list[tuple[object, float]],
    text_weight: float = 1.0,
    table_weight: float = 1.0,
    figure_weight: float = 1.0,
    top_k: int = 8,
) -> List[Tuple[object, float]]:
    channels = [
        (list(text_hits or []), float(text_weight)),
        (list(table_hits or []), float(table_weight)),
        (list(figure_hits or []), float(figure_weight)),
    ]
    fused = weighted_rrf_fusion(channels)
    fused.sort(key=lambda x: x[1])
    if not RAG_PAPER_ENABLE_MULTIMODAL_PACKING:
        return fused[:top_k]
    return diversity_pack_by_parent(fused, top_k=top_k, max_per_parent=3)

