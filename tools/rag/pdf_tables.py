"""
从 PDF 抽取表格（pdfplumber），写入 Chroma 与可选的 ``paper_tables`` 同步列表。

与 ``pdf_figures`` 类似：每条表格一条向量文档，metadata 含 ``chunk_role=table``、``type=pdf_table``。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import (
    PDF_EXTRACT_TABLE_MAX_PER_PAPER,
    RAG_PDF_TABLE_EXTRACT_ENABLED,
)
from tools.rag.time_utils import add_timestamp_metadata

_TABLE_REF_PAT = re.compile(
    r"\b(?:Table|Tab\.)\s*([0-9]+(?:\.[0-9]+)?[A-Za-z]?)\b",
    re.IGNORECASE,
)


def _normalize_cell(c: object) -> str:
    return re.sub(r"\s+", " ", str(c or "").replace("\n", " ").replace("|", "\\|")).strip()[:800]


def _rows_to_markdown(rows: list[list[Any]], *, max_rows: int = 120) -> str:
    if not rows:
        return ""
    cells: list[list[str]] = []
    for row in rows:
        if row is None:
            continue
        cells.append([_normalize_cell(x) for x in row])
    if not cells:
        return ""
    width = max(len(r) for r in cells)
    if width < 2 and len(cells) < 3:
        # 极小网格，多为版面噪声
        return ""
    norm: list[list[str]] = []
    for r in cells:
        pad = list(r) + [""] * (width - len(r))
        norm.append(pad[:width])
    norm = norm[:max_rows]
    header = norm[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in norm[1:]:
        lines.append("| " + " | ".join(r) + " |")
    if len(norm) >= max_rows:
        lines.append("\n…（行数已截断，详见 PDF）")
    return "\n".join(lines)


def _caption_candidates(page_text: str) -> list[tuple[str, str]]:
    """从页文本中提取 (table_number, 整行或短句) 作为 caption 候选，按出现顺序。"""
    out: list[tuple[str, str]] = []
    if not page_text:
        return out
    for m in _TABLE_REF_PAT.finditer(page_text):
        num = str(m.group(1) or "").strip()
        if not num:
            continue
        start = max(0, m.start() - 20)
        end = min(len(page_text), m.end() + 180)
        snippet = page_text[start:end].replace("\n", " ").strip()
        out.append((num, snippet[:400]))
    return out


def extract_tables_from_pdf(
    pdf_path: str | Path,
    *,
    max_tables: int | None = None,
) -> list[dict[str, Any]]:
    """逐页 ``extract_tables``，返回结构化列表（不写库）。"""
    cap = max_tables if max_tables is not None else PDF_EXTRACT_TABLE_MAX_PER_PAPER
    cap = max(1, min(500, int(cap or 1)))
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return []
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return []

    results: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf)) as doc:
            for page_no, page in enumerate(doc.pages, start=1):
                if len(results) >= cap:
                    break
                page_text = page.extract_text() or ""
                caps = _caption_candidates(page_text)
                cap_i = 0
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []
                for t_idx, tbl in enumerate(raw_tables):
                    if len(results) >= cap:
                        break
                    if not tbl or not isinstance(tbl, list):
                        continue
                    md = _rows_to_markdown(tbl)
                    if not md or len(md) < 12:
                        continue
                    table_number: str | None = None
                    caption_text: str | None = None
                    if cap_i < len(caps):
                        table_number, caption_text = caps[cap_i]
                        cap_i += 1
                    results.append(
                        {
                            "page_no": page_no,
                            "table_index_on_page": t_idx,
                            "table_number": table_number,
                            "caption_text": caption_text,
                            "markdown_text": md,
                            "summary_text": (
                                f"Table {table_number}（第 {page_no} 页）"
                                if table_number
                                else f"表格（第 {page_no} 页，序 {t_idx + 1}）"
                            )[:500],
                        }
                    )
    except Exception:
        return results
    return results


def embed_pdf_tables_to_namespace(
    store: Any,
    *,
    pdf_path: str | Path,
    parent_id: str,
    namespace: str,
    arxiv_id: str | None = None,
    chroma_id_prefix: str | None = None,
    pg_chunks_out: list | None = None,
    pg_tables_out: list | None = None,
    enabled: bool | None = None,
) -> int:
    """抽取表格并写入 ``namespace``。返回新增 Document 条数。"""
    use_enabled = RAG_PDF_TABLE_EXTRACT_ENABLED if enabled is None else bool(enabled)
    if not use_enabled:
        return 0

    extracted = extract_tables_from_pdf(pdf_path, max_tables=PDF_EXTRACT_TABLE_MAX_PER_PAPER)
    if not extracted:
        return 0

    docs: list[Document] = []
    prefix = (chroma_id_prefix or "").strip() or None
    aid = arxiv_id or parent_id

    for i, item in enumerate(extracted):
        page = int(item.get("page_no") or 0)
        tnum = item.get("table_number")
        tnum_s = str(tnum).strip() if tnum else ""
        cap = (item.get("caption_text") or "").strip() or None
        md = (item.get("markdown_text") or "").strip()
        summ = (item.get("summary_text") or "").strip()
        head = f"[Table page {page}"
        if tnum_s:
            head += f" number {tnum_s}"
        head += "]"
        cap_line = f"\nCaption: {cap}" if cap else ""
        # 向量正文：摘要 + 表格 Markdown（控制长度，避免单条过大）
        body_max = 14000
        body = f"{head}{cap_line}\n\n{md}"[:body_max]

        meta = add_timestamp_metadata(
            {
                "parent_id": parent_id,
                "source": f"pdf_table:p{page}:{i}",
                "page": page,
                "modality": "table",
                "type": "pdf_table",
                "arxiv_id": aid,
                "chunk_role": "table",
                "has_table": True,
                "table_number": tnum_s or None,
            }
        )
        docs.append(Document(page_content=body, metadata=meta))
        if pg_tables_out is not None:
            pg_tables_out.append(
                {
                    "page_no": page,
                    "table_number": tnum_s or None,
                    "title": None,
                    "caption_text": cap,
                    "summary_text": summ or None,
                    "markdown_text": md[:200000] if md else None,
                }
            )

    if not docs:
        return 0

    ids = [f"{prefix}_tbl_{i:04d}" for i in range(len(docs))] if prefix else None
    n = store.add_documents(docs, namespace=namespace, extra_metadata={}, ids=ids)

    if prefix and pg_chunks_out is not None and ids:
        for i, d in enumerate(docs):
            pg_chunks_out.append(
                {
                    "chroma_doc_id": ids[i],
                    "content": d.page_content or "",
                    "chunk_role": "table",
                    "has_table": True,
                    "has_figure": False,
                    "chunk_index": -1,
                }
            )
    return n
