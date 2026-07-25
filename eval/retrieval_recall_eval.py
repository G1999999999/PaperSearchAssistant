from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.rag.knowledge import CHROMA_PERSIST_DIR, NamespaceVectorStore


@dataclass
class RetrievalHitSpec:
    """单条「应被召回」约束：Top-K 中存在至少一个 chunk 满足即算命中该条。"""

    match_any_text: list[str] = field(default_factory=list)
    metadata_contains: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def normalized(self) -> RetrievalHitSpec:
        texts = [t.strip() for t in self.match_any_text if t and str(t).strip()]
        meta = {str(k).strip(): v for k, v in (self.metadata_contains or {}).items()}
        return RetrievalHitSpec(
            match_any_text=texts,
            metadata_contains=meta,
            note=self.note,
        )


@dataclass
class RetrievalEvalCase:
    id: str
    namespace: str
    queries: list[str]
    k: int = 10
    strategy: str = "vector"
    score_threshold: float = 10.0
    chroma_filter: dict[str, Any] | None = None
    hits: list[RetrievalHitSpec] = field(default_factory=list)
    min_hit_ratio: float = 1.0
    enabled: bool = True


def _fixture_default_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "golden_retrieval_chunk_ids.json"


def load_retrieval_eval_cases(path: Path | None = None) -> tuple[dict[str, Any], list[RetrievalEvalCase]]:
    p = path or _fixture_default_path()
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    cases: list[RetrievalEvalCase] = []
    for item in raw.get("cases") or []:
        if item.get("enabled") is False:
            continue
        hits_raw = item.get("hits") or []
        hits = []
        for h in hits_raw:
            hits.append(
                RetrievalHitSpec(
                    match_any_text=list(h.get("match_any_text") or []),
                    metadata_contains=dict(h.get("metadata_contains") or {}),
                    note=str(h.get("note") or ""),
                ).normalized()
            )
        cases.append(
            RetrievalEvalCase(
                id=str(item["id"]),
                namespace=str(item.get("namespace") or "default"),
                queries=[str(q).strip() for q in (item.get("queries") or []) if str(q).strip()],
                k=int(item.get("k") or 10),
                strategy=str(item.get("strategy") or "vector"),
                score_threshold=float(item.get("score_threshold") if item.get("score_threshold") is not None else 10.0),
                chroma_filter=item.get("chroma_filter"),
                hits=hits,
                min_hit_ratio=float(item.get("min_hit_ratio") if item.get("min_hit_ratio") is not None else 1.0),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return raw, cases


def _meta_value_equal(got: Any, want: Any) -> bool:
    if got is None and want is None:
        return True
    if got is None or want is None:
        return False
    if isinstance(want, bool):
        return bool(got) == want
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return str(got).strip() == str(want).strip()
    return str(got).strip() == str(want).strip()


def hit_satisfied_by_any_doc(docs: list[Any], hit: RetrievalHitSpec) -> bool:
    """若 hit 无文本、无元数据约束，视为非法（永不满足）。"""
    h = hit.normalized()
    if not h.match_any_text and not h.metadata_contains:
        return False
    for doc in docs:
        md = getattr(doc, "metadata", None) or {}
        if not isinstance(md, dict):
            md = {}
        text = (getattr(doc, "page_content", None) or "").casefold()

        meta_ok = True
        for mk, mv in h.metadata_contains.items():
            if not _meta_value_equal(md.get(mk), mv):
                meta_ok = False
                break
        if not meta_ok:
            continue

        if h.match_any_text:
            if not any(p.casefold() in text for p in h.match_any_text):
                continue

        return True
    return False


def recall_hit_ratio(docs: list[Any], hits: list[RetrievalHitSpec]) -> tuple[float, int, int]:
    if not hits:
        return 1.0, 0, 0
    sat = sum(1 for hit in hits if hit_satisfied_by_any_doc(docs, hit))
    return sat / len(hits), sat, len(hits)


def get_eval_vector_store(*, persist_dir: str | None = None) -> NamespaceVectorStore:
    from models_qwen import qwen_embeddings

    env_dir = (os.environ.get("CHROMA_EVAL_PERSIST_DIR") or "").strip()
    path = persist_dir or env_dir or CHROMA_PERSIST_DIR
    return NamespaceVectorStore(embeddings=qwen_embeddings, persist_directory=path)


def run_retrieval_case(store: NamespaceVectorStore, case: RetrievalEvalCase) -> dict[str, Any]:
    if not case.queries:
        raise ValueError(f"case {case.id}: queries 不能为空")
    rows = store.retrieve(
        queries=case.queries,
        namespace=case.namespace,
        k=case.k,
        score_threshold=case.score_threshold,
        strategy=case.strategy,
        extra_chroma_filter=case.chroma_filter,
    )
    docs = [d for d, _ in rows]
    ratio, sat, nhit = recall_hit_ratio(docs, [h.normalized() for h in case.hits])
    ok = ratio + 1e-9 >= case.min_hit_ratio
    return {
        "id": case.id,
        "ok": ok,
        "hit_ratio": ratio,
        "hits_satisfied": sat,
        "hits_total": nhit,
        "min_hit_ratio": case.min_hit_ratio,
        "retrieved": len(docs),
    }


def run_all_retrieval_cases(
    store: NamespaceVectorStore | None = None,
    *,
    fixture: Path | None = None,
    persist_dir: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw, cases = load_retrieval_eval_cases(fixture)
    root_persist = raw.get("chroma_persist_dir")
    if isinstance(root_persist, str) and root_persist.strip():
        persist_dir = root_persist.strip()
    st = store or get_eval_vector_store(persist_dir=persist_dir)
    report = [run_retrieval_case(st, c) for c in cases]
    return report, raw


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chroma 检索 recall 评测（需数据与嵌入环境）")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=_fixture_default_path(),
        help="黄金 JSON（默认 eval/fixtures/golden_retrieval_chunk_ids.json）",
    )
    args = parser.parse_args()
    report, _raw = run_all_retrieval_cases(fixture=args.fixture)
    for row in report:
        status = "PASS" if row["ok"] else "FAIL"
        print(
            f"{status} {row['id']} hit_ratio={row['hit_ratio']:.3f} "
            f"({row['hits_satisfied']}/{row['hits_total']}) retrieved={row['retrieved']}"
        )
    if any(not r["ok"] for r in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
