"""
对话场景（会话上传 / chat --embed-file）用的文件读取。

对齐 LangChain 教程里常见组合：TextLoader、PyMuPDFLoader、Docx2txtLoader、HTML 用 BeautifulSoup；
Excel 仍用与 `document.py` 一致的 openpyxl 表格展开，避免重复实现。

与通用 `load_file()` 的区别：PDF/Docx 走 LangChain Community Loader，便于与教程代码对照；
若缺少可选依赖或 Loader 失败，会回退到 `document.load_file`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from langchain_core.documents import Document

from tools.rag.document import (
    _basic_clean,
    _load_docx,
    _load_excel,
    _load_html,
    _load_txt,
    _metadata_from_path,
    load_file as load_file_legacy,
)


def _merge_lc_documents(
    docs: list[Document],
    p: Path,
    *,
    loader_name: str,
) -> Tuple[str, dict]:
    """将 Loader 产出的多页/多段 Document 拼成一段文本，并附 metadata。"""
    parts = [((d.page_content or "").strip()) for d in docs]
    text = _basic_clean("\n\n".join(x for x in parts if x))
    meta = _metadata_from_path(p)
    meta["session_loader"] = loader_name
    if len(docs) > 1:
        meta["document_parts"] = len(docs)
    return text, meta


def load_file_for_session(path: str | Path) -> Tuple[str, dict]:
    """
    为「对话中附加文件」加载纯文本 + metadata。

    扩展名策略（与 LangChainProject/RAG_v1/01_RAG_document.py 思路一致）：
    - .txt / .md：TextLoader
    - .html / .htm：BeautifulSoup（复用 document._load_html）
    - .pdf：PyMuPDFLoader
    - .docx：Docx2txtLoader（不可用时回退 python-docx）
    - .xlsx 等：openpyxl（复用 document._load_excel）
    - 其它：TextLoader(utf-8)，失败则 load_file_legacy
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()

    if suffix in {".html", ".htm"}:
        text, meta = _load_html(p)
        meta = dict(meta)
        meta["session_loader"] = "beautifulsoup_html"
        return text, meta

    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        text, meta = _load_excel(p)
        meta = dict(meta)
        meta["session_loader"] = "openpyxl_excel"
        return text, meta

    if suffix in {".txt", ".md", ".markdown"}:
        try:
            from langchain_community.document_loaders import TextLoader

            docs = TextLoader(str(p), encoding="utf-8").load()
            return _merge_lc_documents(docs, p, loader_name="langchain_text")
        except Exception:
            return _load_txt(p) if suffix == ".txt" else load_file_legacy(str(p))

    if suffix == ".pdf":
        try:
            from langchain_community.document_loaders import PyMuPDFLoader

            docs = PyMuPDFLoader(str(p)).load()
            if not any((d.page_content or "").strip() for d in docs):
                return load_file_legacy(str(p))
            return _merge_lc_documents(docs, p, loader_name="langchain_pymupdf")
        except Exception:
            return load_file_legacy(str(p))

    if suffix == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            docs = Docx2txtLoader(str(p)).load()
            return _merge_lc_documents(docs, p, loader_name="langchain_docx2txt")
        except Exception:
            return _load_docx(p)

    # 未知扩展名：先按 UTF-8 文本 Loader，再回退 legacy
    try:
        from langchain_community.document_loaders import TextLoader

        docs = TextLoader(str(p), encoding="utf-8").load()
        return _merge_lc_documents(docs, p, loader_name="langchain_text_fallback")
    except Exception:
        return load_file_legacy(str(p))
