from __future__ import annotations

from typing import Any

from tools.rag.retrieval_merge import retrieve_with_public_merge


def build_section_fallback_queries(
    *,
    question: str,
    intent: str,
    missing_method: bool,
    missing_result: bool,
    missing_conclusion: bool,
    abstract_truncated: bool,
) -> list[str]:
    q = (question or "").strip()
    out: list[str] = []
    if missing_method:
        out.extend(
            [
                f"{q} method approach architecture model design",
                "method approach architecture model",
            ]
        )
    if missing_result:
        out.extend(
            [
                f"{q} experiment evaluation results ablation benchmark",
                "experiment evaluation result ablation",
            ]
        )
    if missing_conclusion:
        out.extend([f"{q} conclusion discussion limitation", "conclusion discussion"])
    if abstract_truncated or intent == "paper_summary":
        out.extend([f"{q} abstract contribution summary", "paper abstract contribution"])
    # 去重保序
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        t = " ".join((x or "").split()).strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(t)
    return dedup[:8]


def run_section_fallback(
    store: Any,
    *,
    queries: list[str],
    namespace: str,
    strategy: str,
    score_threshold: float,
    session_ingest_ids: list[str] | None,
    k: int,
    extra_chroma_filter: dict[str, Any] | None = None,
) -> list[tuple[object, float]]:
    if not queries:
        return []
    return retrieve_with_public_merge(
        store,
        queries=queries,
        namespace=namespace,
        k=max(8, int(k)),
        score_threshold=min(float(score_threshold), 0.3),
        strategy=strategy,
        session_ingest_ids=session_ingest_ids,
        extra_chroma_filter=extra_chroma_filter,
    )

