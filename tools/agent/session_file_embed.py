"""
将会话中的本地文件嵌入指定 RAG namespace。

- 默认走 LangChain Loader 组合（见 ``session_document_loaders.load_file_for_session``）。
- 若上传的是 **arXiv 新格式文件名的 PDF**（如 ``1706.03762.pdf``），会 **额外** 复制到
  ``data/papers`` 并调用 ``ingest_arxiv_paper_full_pipeline`` 写入论文库与论文向量；
  会话 namespace 仍会嵌入，便于当前对话直接检索。

供 CLI ``chat --embed-file`` 与 API ``POST /session/embed_file`` 共用。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from tools.rag.knowledge import NamespaceVectorStore, vector_store
from tools.rag.session_document_loaders import load_file_for_session

PAPERS_DIR = Path("data/papers")
_PAPER_INGEST_NOTE_MAX = 800


@dataclass
class SessionEmbedResult:
    chunks_added: int
    arxiv_id: str | None = None
    paper_library_ingested: bool = False
    paper_ingest_note: str | None = None


def _maybe_ingest_arxiv_style_pdf(local_pdf: Path) -> tuple[str | None, bool, str | None]:
    """若文件名为 arXiv 风格 PDF，复制到论文目录并跑入库流水线。返回 (arxiv_id, ingested_ok, note)。"""
    from tools.agent.router import arxiv_id_from_arxiv_style_pdf_basename

    if local_pdf.suffix.lower() != ".pdf":
        return None, False, None
    aid = arxiv_id_from_arxiv_style_pdf_basename(local_pdf.name)
    if not aid:
        return None, False, None

    try:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        dest = PAPERS_DIR / f"{aid}.pdf"
        shutil.copy2(local_pdf, dest)
    except Exception as exc:
        return aid, False, f"复制到 data/papers 失败: {exc}"

    try:
        from tools.agent.paper_ingest import ingest_arxiv_paper_full_pipeline

        msg = ingest_arxiv_paper_full_pipeline(aid)
        note = (msg or "").strip()
        if len(note) > _PAPER_INGEST_NOTE_MAX:
            note = note[:_PAPER_INGEST_NOTE_MAX] + "…"
        # 流水线以返回文案为主；若明确是「无效 ID」则标为未成功入库
        if note.startswith("请提供有效的 arXiv ID"):
            return aid, False, note
        # 文件已落盘；即使全文向量失败，论文库/SQLite 通常已更新
        return aid, True, note
    except Exception as exc:
        return aid, False, f"论文入库流水线异常: {exc}"


def embed_session_file(
    path: Path | str,
    namespace: str,
    *,
    store: NamespaceVectorStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> SessionEmbedResult:
    """读取本地文件、可选论文库入库、再写入会话 namespace。返回结构化结果。"""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    ns = (namespace or "").strip()
    if not ns:
        raise ValueError("namespace is required")

    arxiv_id: str | None = None
    paper_ok = False
    paper_note: str | None = None
    arxiv_id, paper_ok, paper_note = _maybe_ingest_arxiv_style_pdf(p)

    text, meta = load_file_for_session(p)
    md = dict(meta or {})
    md.setdefault("type", "session_upload")
    md.setdefault("source", str(p.as_posix()))
    md.setdefault("original_filename", p.name)
    if arxiv_id:
        md.setdefault("arxiv_id", arxiv_id)
        md.setdefault("paper_library_touched", paper_ok)
    if extra_meta:
        md.update(extra_meta)

    cs = int(chunk_size) if chunk_size is not None else DEFAULT_CHUNK_SIZE
    co = int(chunk_overlap) if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP
    st = store or vector_store
    n = st.embed_document(
        text=text,
        namespace=ns,
        chunk_size=max(100, cs),
        chunk_overlap=max(0, co),
        extra_metadata=md,
    )

    return SessionEmbedResult(
        chunks_added=n,
        arxiv_id=arxiv_id,
        paper_library_ingested=bool(arxiv_id and paper_ok),
        paper_ingest_note=paper_note,
    )


def embed_local_path_into_namespace(
    path: Path | str,
    namespace: str,
    *,
    store: NamespaceVectorStore | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> int:
    """兼容旧接口：仅返回新增 chunk 数。"""
    return embed_session_file(
        path,
        namespace,
        store=store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_meta=extra_meta,
    ).chunks_added
