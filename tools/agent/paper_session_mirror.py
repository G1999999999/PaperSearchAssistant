"""
将本地已入库论文的向量内容只读复制到「会话」Chroma namespace。

约束：不修改 `paper:<arxiv_id>:full` collection，不写入/更新 SQLite 论文表；
仅向目标会话 namespace 追加（或 replace 时先删本会话内该篇镜像再追加）。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from tools.agent.agent_tools import _normalize_arxiv_id
from tools.rag.document import load_pdf
from tools.rag.knowledge import NamespaceVectorStore
from tools.rag.time_utils import add_timestamp_metadata
from tools.storage.paper_library import get_paper

SESSION_PAPER_MIRROR_ROLE = "session_paper_mirror"


def _mirror_where_clause(arxiv_id: str) -> dict:
    return {
        "$and": [
            {"doc_role": SESSION_PAPER_MIRROR_ROLE},
            {"source_arxiv_id": arxiv_id},
        ]
    }


def delete_session_paper_mirror(
    store: NamespaceVectorStore,
    session_namespace: str,
    arxiv_id_raw: str,
) -> None:
    """删除某会话 namespace 内、指定 arXiv ID 的镜像文档（不触碰论文库）。"""
    aid = _normalize_arxiv_id(arxiv_id_raw)
    if not aid:
        return
    store.delete_by_where(session_namespace.strip(), _mirror_where_clause(aid))


def mirror_local_paper_to_namespace(
    store: NamespaceVectorStore,
    session_namespace: str,
    arxiv_id_raw: str,
    *,
    replace: bool = False,
) -> str:
    """把本地论文向量从 `paper:<id>:full` 复制到 `session_namespace`；若无向量则从 PDF 只读嵌入到会话。

    若 `session_namespace` 与 canonical 论文库 `paper:<id>:full` 相同则拒绝，避免污染全局论文分区。
    """
    aid = _normalize_arxiv_id(arxiv_id_raw)
    if not aid:
        return "无效的 arXiv ID。"

    sess = (session_namespace or "").strip()
    if not sess:
        return "请指定会话 namespace（例如 conv_001）。"

    paper_ns = f"paper:{aid}:full"
    if sess == paper_ns:
        return (
            "拒绝：目标 namespace 与论文库 `paper:<id>:full` 相同。"
            "请使用会话分区（如 conv_001），以免与全局论文索引混淆。"
        )

    # 只读校验：本地是否有记录或 PDF（不修改 SQLite）
    rec = get_paper(aid) or {}
    pdf_path = rec.get("pdf_path") or f"data/papers/{aid}.pdf"
    pdf_ok = Path(pdf_path).exists()

    if replace:
        delete_session_paper_mirror(store, sess, aid)

    source_docs = store.export_documents(paper_ns)
    mirror_meta = {
        "doc_role": SESSION_PAPER_MIRROR_ROLE,
        "source_arxiv_id": aid,
        "mirrored_from_paper_ns": paper_ns,
    }

    if source_docs:
        to_add: list[Document] = []
        for d in source_docs:
            md = dict(d.metadata or {})
            md.pop("namespace", None)
            to_add.append(
                Document(page_content=d.page_content or "", metadata=md)
            )
        n = store.add_documents(
            to_add,
            namespace=sess,
            extra_metadata=mirror_meta,
        )
        return (
            f"已从论文库 `{paper_ns}` 只读复制 {n} 条向量块到会话 `{sess}`（未修改论文库与 SQLite）。"
        )

    if not pdf_ok:
        return (
            f"论文库 `{paper_ns}` 中暂无向量，且未找到本地 PDF：`{pdf_path}`。\n"
            "请先执行全文入库（例如：`帮我把 <id> 下载并入库`）。"
        )

    text, meta = load_pdf(str(pdf_path), parent_id=f"{aid}:session_mirror")
    base = dict(meta or {})
    base.setdefault("arxiv_id", aid)
    base.setdefault("type", "session_mirrored_pdf")
    extra = add_timestamp_metadata(
        {
            **base,
            **mirror_meta,
            "mirrored_from_pdf_only": True,
        }
    )
    n = store.embed_document(
        text=text,
        namespace=sess,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        extra_metadata=extra,
    )
    return (
        f"论文库无向量，已从本地 PDF 只读嵌入 {n} 块到会话 `{sess}`（未写入 `{paper_ns}`，未修改 SQLite）。"
    )
