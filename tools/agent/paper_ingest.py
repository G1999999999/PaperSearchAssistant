"""
论文入库流水线：PDF → data/papers、SQLite、向量库（论文专用 namespace + 可选公共 namespace）。

供普通对话（任意会话 namespace）与 tool_ingest_arxiv_paper 共用。

另：可将 arXiv 元数据（标题/作者/摘要）写成独立向量文档，与 PDF 全文 chunk 同库，
    hybrid 检索时 BM25 与向量均可命中摘要（见 RAG_INGEST_ARXIV_ABSTRACT_VECTOR）。

当配置 ``DATABASE_URL`` 且 ``RAG_PG_SYNC_ON_INGEST=1`` 时，同步写入 PostgreSQL：
``papers``、``paper_chunks``（与 Chroma 稳定 ``chroma_doc_id`` 对齐）、
插图开启时 ``paper_figures``、表格抽取开启时 ``paper_tables``（需 ``pdfplumber``）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NAMESPACE,
    RAG_INGEST_ARXIV_ABSTRACT_VECTOR,
    RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC,
    RAG_INGEST_SECTION_MODE,
    RAG_PDF_FIGURE_CAPTION_ENABLED,
    RAG_PDF_TABLE_EXTRACT_ENABLED,
    RAG_PUBLIC_NAMESPACE,
)
from langchain_text_splitters import TokenTextSplitter
from tools.agent.agent_tools import _normalize_arxiv_id
from tools.agent.arxiv_search import download_pdf, get_paper_by_id
from tools.rag.document import load_pdf
from tools.rag.knowledge import NamespaceVectorStore, vector_store
from tools.rag.time_utils import add_timestamp_metadata
from tools.storage.paper_library import LocalPaper, upsert_paper
from tools.storage.repos import paper_ingest_repo
from tools.storage.repos.section_repo import (
    infer_section_role as _infer_section_role_sql,
    looks_like_section_heading as _looks_like_section_heading_sql,
)
from tools.storage.redis.cache import cache_section_tree
from tools.storage.redis.section_cache import set_section_roles, set_summary_bundle


_SEC_HEADING_PAT = re.compile(
    r"^\s*((\d+(\.\d+)*)\s+)?(abstract|introduction|related work|method|methods|approach|experiments?|results?|discussion|conclusion[s]?)\b",
    re.IGNORECASE,
)


def _infer_role(title: str) -> str:
    # 复用 PG 章节角色推断逻辑，避免 ingest 与检索两端口径不一致。
    return _infer_section_role_sql(title)


_SEC_HEADING_PAT_EXT = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)|(?P<roman>[IVXLCDM]+))?\s*\.?\s*"
    r"(?P<key>"
    r"abstract|introduction|related work|method(?:s|ology)?|approach|"
    r"experimental(?: setup| design)?|experiments?|results?|evaluation|"
    r"ablation|discussion|conclusion(?:s)?|limitations|background|preliminaries"
    r")\b",
    re.IGNORECASE,
)


def _basic_clean_light(text: str) -> str:
    """比 load_pdf 更轻的清洗：只做换行统一、空行压缩与首尾裁剪。"""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _is_toc_page(page_text: str) -> bool:
    t = (page_text or "").lower()
    if "contents" not in t and "目录" not in t:
        return False
    # 若 TOC 页面同时出现摘要/引言关键词，可能是正文开头，尽量不要误判。
    if re.search(r"\babstract\b|\bintroduction\b|\b引言\b", t, re.I):
        return False
    return True


def _detect_first_heading_in_page(page_text: str) -> dict[str, str] | None:
    """
    从单页文本里尽量找“最早出现的章节标题行”。
    返回 {title, section_number, role}（role 由 _infer_role 生成）。
    """
    if not page_text or not (page_text or "").strip():
        return None
    if _is_toc_page(page_text):
        return None

    lines = (page_text or "").splitlines()
    for raw in lines[:80]:
        line = (raw or "").strip()
        if not line:
            continue
        if len(line) > 180:
            continue
        lower = line.lower()
        # 避免把图/表标题当作“章节标题”导致 section 切分严重偏移。
        if lower.startswith("fig") or lower.startswith("figure") or lower.startswith("tab") or lower.startswith("table"):
            continue

        m = _SEC_HEADING_PAT_EXT.match(line)
        if not m:
            # 排版可能导致关键字后置（如 "1.  Method" 等），降级为关键字包含判断
            line_short = line[:90]
            kw_pat = (
                r"(?i)\b(abstract|introduction|related work|method(?:s|ology)?|approach|"
                r"experimental(?: setup| design)?|experiments?|results?|evaluation|"
                r"ablation|discussion|conclusion(?:s)?|limitations|background|preliminaries)\b"
            )
            kw_m = re.search(kw_pat, line_short)
            if not kw_m or kw_m.start() > 28:
                continue
            title = line[:180].strip()
            sec_no = ""
        else:
            title = line[:180].strip()
            sec_no = (m.group("num") or m.group("roman") or "").strip()

        if not title:
            continue
        if not _looks_like_section_heading_sql(title):
            continue
        return {"title": title, "section_number": sec_no, "role": _infer_role(title)}

    return None


def _extract_sections_from_pdf_pages(
    *,
    page_texts: list[tuple[int, str]],
) -> list[dict[str, str | int]]:
    """
    基于 PyMuPDF 页级文本的章节粗切：仅用“页首最早章节标题行”推断起止页。
    输出形态兼容现有 PG 写入逻辑（带 start_page/end_page/title/role/section_number）。
    """
    headings: list[dict[str, str | int]] = []
    for page_idx, text in page_texts:
        h = _detect_first_heading_in_page(text)
        if not h:
            continue
        norm_title = str(h["title"]).strip().casefold()
        if not norm_title:
            continue
        # 去重：连续重复标题跳过
        if headings and str(headings[-1].get("title") or "").strip().casefold() == norm_title:
            continue
        headings.append(
            {
                "page_idx": int(page_idx),
                "title": str(h["title"]).strip(),
                "section_number": str(h.get("section_number") or "").strip(),
                "role": str(h.get("role") or "other"),
            }
        )

    if not headings:
        return [
            {
                "order_index": 0,
                "section_number": "0",
                "section_level": 1,
                "title": "Body",
                "start_page": page_texts[0][0] if page_texts else 1,
                "end_page": page_texts[-1][0] if page_texts else 1,
                "role": "other",
            }
        ]

    first_page = int(page_texts[0][0])
    last_page = int(page_texts[-1][0])
    sections: list[dict[str, str | int]] = []

    # Body：从第一页到第一个标题页之前
    body_end = int(headings[0]["page_idx"]) - 1
    sections.append(
        {
            "order_index": 0,
            "section_number": "0",
            "section_level": 1,
            "title": "Body",
            "start_page": first_page,
            "end_page": body_end,
            "role": "other",
        }
    )

    # 逐标题切分
    for i, h in enumerate(headings):
        start_p = int(h["page_idx"])
        end_p = int(headings[i + 1]["page_idx"]) - 1 if i + 1 < len(headings) else last_page
        sections.append(
            {
                "order_index": len(sections),
                "section_number": str(h.get("section_number") or "").strip(),
                "section_level": 1,
                "title": str(h.get("title") or "").strip(),
                "start_page": start_p,
                "end_page": end_p,
                "role": str(h.get("role") or "other"),
            }
        )
    return sections


def _extract_sections_from_body_pairs(body_pairs: list[tuple[str, str]]) -> list[dict]:
    sections: list[dict] = []
    order = 0
    # 默认加一个正文入口，避免空 section。
    sections.append(
        {
            "order_index": order,
            "section_number": "0",
            "section_level": 1,
            "title": "Body",
            "start_chunk_index": 1,
            "end_chunk_index": len(body_pairs),
            "role": "other",
        }
    )
    order += 1
    for i, (_cid, text) in enumerate(body_pairs, start=1):
        first_lines = "\n".join((text or "").splitlines()[:3]).strip()
        m = _SEC_HEADING_PAT.search(first_lines)
        if not m:
            continue
        heading = first_lines.splitlines()[0][:180]
        if not _looks_like_section_heading_sql(heading):
            continue
        sec_no = (m.group(2) or "").strip() or str(order)
        role = _infer_role(heading)
        sections.append(
            {
                "order_index": order,
                "section_number": sec_no,
                "section_level": 1,
                "title": heading,
                "start_chunk_index": i,
                "end_chunk_index": len(body_pairs),
                "role": role,
            }
        )
        order += 1
    sections = sorted(sections, key=lambda x: int(x["start_chunk_index"]))
    # 重算区间 end
    for idx, sec in enumerate(sections):
        nxt = sections[idx + 1] if idx + 1 < len(sections) else None
        sec["end_chunk_index"] = (
            max(int(sec["start_chunk_index"]), int(nxt["start_chunk_index"]) - 1)
            if nxt
            else len(body_pairs)
        )
        sec["order_index"] = idx
    return sections


def _build_section_binding(
    sections: list[dict],
    chunk_start_index: int,
    chunk_end_index: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    if chunk_end_index < chunk_start_index:
        return out
    for cidx in range(int(chunk_start_index), int(chunk_end_index) + 1):
        for sidx, sec in enumerate(sections):
            if int(sec["start_chunk_index"]) <= cidx <= int(sec["end_chunk_index"]):
                out[cidx] = sidx
                break
    return out


def _build_section_summaries(
    sections: list[dict],
    body_pairs: list[tuple[str, str]],
) -> tuple[list[dict], dict]:
    body_map = {i: txt for i, (_cid, txt) in enumerate(body_pairs, start=1)}
    sec_summaries: list[dict] = []
    role_to_text: dict[str, str] = {}
    for sec in sections:
        s = int(sec["start_chunk_index"])
        e = int(sec["end_chunk_index"])
        parts: list[str] = []
        for i in range(s, min(e, s + 1) + 1):
            t = (body_map.get(i) or "").strip()
            if t:
                parts.append(t[:1200])
        summary = "\n".join(parts).strip()
        role = str(sec.get("role") or "other")
        if summary and role not in role_to_text:
            role_to_text[role] = summary[:1500]
        sec_summaries.append(
            {
                "order_index": int(sec["order_index"]),
                "section_role": role,
                "summary_text": summary[:1800] if summary else "",
                "keywords_json": [],
            }
        )
    bundle = {
        "abstract_summary": role_to_text.get("abstract") or "",
        "intro_summary": role_to_text.get("introduction") or "",
        "method_summary": role_to_text.get("method") or "",
        "result_summary": role_to_text.get("result") or "",
        "conclusion_summary": role_to_text.get("conclusion") or "",
    }
    return sec_summaries, bundle


def _chroma_id_root(arxiv_id: str) -> str:
    a = (arxiv_id or "").strip().replace(".", "_").replace(":", "_").replace("/", "_")
    a = a[:48] if len(a) > 48 else a
    return f"psa_{a}" if a else "psa_unknown"


def _env_true(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def _resolve_ingest_profile() -> tuple[str, bool, bool]:
    """返回 (section_mode, enable_figures, enable_tables)。

    规则：
    - RAG_INGEST_PROFILE=fast: section_aware + 关闭图/表（极速正文入库）
    - RAG_INGEST_PROFILE=full: section_aware + 开启图/表（完整入库）
    - RAG_INGEST_PROFILE=auto(默认):
      - 批量模式（RAG_INGEST_BULK=1）=> fast
      - 其他 => full
    """
    profile = (os.getenv("RAG_INGEST_PROFILE", "auto") or "auto").strip().lower()
    bulk = _env_true("RAG_INGEST_BULK", False)
    if profile == "fast":
        return "section_aware", False, False
    if profile == "full":
        return "section_aware", True, True
    if bulk:
        return "section_aware", False, False
    # auto + 非批量：优先完整质量
    return "section_aware", True, True


def build_arxiv_abstract_text(
    arxiv_id: str,
    title: str | None,
    authors: Sequence[str] | None,
    summary: str,
) -> str:
    """拼成一条可检索文本（偏 BM25 友好）。"""
    parts: list[str] = [f"arXiv:{arxiv_id}"]
    t = (title or "").strip()
    if t:
        parts.append(f"标题: {t}")
    auth = authors or []
    if auth:
        parts.append("作者: " + ", ".join(str(a) for a in auth if a))
    parts.append("摘要（arXiv）:")
    parts.append(summary.strip())
    return "\n".join(parts)


def embed_arxiv_abstract_documents(
    store: NamespaceVectorStore,
    *,
    arxiv_id: str,
    title: str | None,
    authors: list[str] | None,
    summary: str | None,
    paper_namespace: str,
    public_namespace: str | None = None,
    also_embed_public: bool = False,
    chroma_doc_id: str | None = None,
    public_chroma_doc_id: str | None = None,
) -> int:
    """将标题+作者+摘要写入 Chroma（论文 namespace；可选再写入公共 namespace）。返回新增文档条数（每 namespace 最多 1）。

    ``chroma_doc_id`` 非空时用于论文分区写入，便于与 PostgreSQL ``paper_chunks.chroma_doc_id`` 对齐。
    """
    if not RAG_INGEST_ARXIV_ABSTRACT_VECTOR:
        return 0
    s = (summary or "").strip()
    if not s:
        return 0
    body = build_arxiv_abstract_text(arxiv_id, title, authors, s)
    base_meta = add_timestamp_metadata(
        {
            "arxiv_id": arxiv_id,
            # 与 PDF 全文 parent_id（通常为 arxiv_id）区分，避免 _group_by_parent 把摘要与数百页拼成一条
            "parent_id": f"{arxiv_id}:arxiv_abstract",
            "type": "arxiv_abstract",
            "source": "arxiv_metadata",
            "chunk_role": "paper_summary",
        }
    )
    doc = Document(page_content=body, metadata=base_meta)
    ids_paper = [chroma_doc_id] if (chroma_doc_id or "").strip() else None
    n = store.add_documents(
        [doc],
        namespace=paper_namespace,
        extra_metadata={"vector_partition": "paper"},
        ids=ids_paper,
    )
    pub = (public_namespace or "").strip() or DEFAULT_NAMESPACE
    if also_embed_public and pub and pub != paper_namespace:
        pid = (public_chroma_doc_id or "").strip()
        ids_pub = [pid] if pid else None
        n += store.add_documents(
            [Document(page_content=body, metadata=dict(base_meta))],
            namespace=pub,
            extra_metadata={
                "vector_partition": "public",
                "paper_vector_namespace": paper_namespace,
            },
            ids=ids_pub,
        )
    return n


def ingest_arxiv_paper_full_pipeline(
    arxiv_id_raw: str,
    *,
    embed_full_text: bool = True,
    embed_public: bool | None = None,
    paper_namespace_override: str | None = None,
) -> str:
    """下载 PDF、写入 SQLite，并将全文嵌入论文 namespace 与（默认）公共 namespace。

    - 论文向量 namespace 默认 ``paper:<id>:full``，可通过 paper_namespace_override 覆盖。
    - 公共库默认使用 ``RAG_PUBLIC_NAMESPACE``（常与 ``default`` 一致）。
    """
    arxiv_id = _normalize_arxiv_id(arxiv_id_raw)
    if not arxiv_id:
        return "请提供有效的 arXiv ID，例如 2401.12345。"

    do_public = RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC if embed_public is None else bool(embed_public)
    public_ns = (RAG_PUBLIC_NAMESPACE or "").strip() or DEFAULT_NAMESPACE
    paper_ns = (
        (paper_namespace_override or "").strip()
        if paper_namespace_override and paper_namespace_override.strip()
        else f"paper:{arxiv_id}:full"
    )

    pdf_path = Path("data/papers") / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        pdf_path = download_pdf(arxiv_id, dest_dir="data/papers")

    paper = get_paper_by_id(arxiv_id)
    rec = upsert_paper(
        LocalPaper(
            arxiv_id=arxiv_id,
            pdf_path=str(pdf_path.as_posix()),
            title=paper.title if paper else None,
            authors=paper.authors if paper else [],
            published=(paper.published.isoformat() if paper and paper.published else None),
            url=(paper.url if paper else f"https://arxiv.org/abs/{arxiv_id}"),
            summary=(paper.summary or "").strip() if paper else None,
        )
    )

    if not embed_full_text:
        if paper_ingest_repo.pg_sync_enabled():
            abs_s = (rec.get("summary") or "").strip() or (
                (paper.summary or "").strip() if paper else ""
            )
            paper_ingest_repo.upsert_paper_row_for_ingest(
                arxiv_id=arxiv_id,
                title=rec.get("title") or (paper.title if paper else None),
                abstract=abs_s or None,
                authors=list(rec.get("authors") or []) or (paper.authors if paper else []),
                pdf_path=str(pdf_path.as_posix()),
                source_url=paper.url if paper else f"https://arxiv.org/abs/{arxiv_id}",
                published=rec.published,
                page_count=None,
            )
        view_url = f"/papers/view/{pdf_path.name}"
        return (
            f"已保存 PDF 与论文库元数据（未做全文向量入库）。\n"
            f"- arxiv_id: {arxiv_id}\n"
            f"- pdf: {pdf_path.as_posix()}\n"
            f"- view: {view_url}\n"
        )

    pg_note = ""
    try:
        section_mode_eff, enable_figures_eff, enable_tables_eff = _resolve_ingest_profile()
        text, meta = load_pdf(str(pdf_path), parent_id=arxiv_id)
        base_meta = dict(meta or {})
        base_meta.setdefault("arxiv_id", arxiv_id)
        base_meta.setdefault("type", "arxiv_pdf_full")

        page_count: int | None = None
        try:
            from pypdf import PdfReader

            page_count = len(PdfReader(str(pdf_path)).pages)
        except Exception:
            page_count = None

        id_root = _chroma_id_root(arxiv_id)
        abs_summary = (rec.get("summary") or "").strip() or (
            (paper.summary or "").strip() if paper else ""
        )
        pub_url = paper.url if paper else f"https://arxiv.org/abs/{arxiv_id}"

        paper_pg_id: int | None = None
        if paper_ingest_repo.pg_sync_enabled():
            paper_pg_id = paper_ingest_repo.upsert_paper_row_for_ingest(
                arxiv_id=arxiv_id,
                title=rec.get("title") or (paper.title if paper else None),
                abstract=abs_summary or None,
                authors=list(rec.get("authors") or []) or (paper.authors if paper else []),
                pdf_path=str(pdf_path.as_posix()),
                source_url=pub_url,
                published=rec.published,
                page_count=page_count,
            )

        pg_items: list[dict] = []
        namespaces_to_record: set[str] = set()

        abs_chroma = f"{id_root}_abstract"
        pub_abs = f"{id_root}_abstract_pub"
        embed_arxiv_abstract_documents(
            vector_store,
            arxiv_id=arxiv_id,
            title=rec.get("title") or (paper.title if paper else None),
            authors=list(rec.get("authors") or []) or (paper.authors if paper else []),
            summary=abs_summary or None,
            paper_namespace=paper_ns,
            public_namespace=public_ns,
            also_embed_public=bool(do_public and public_ns and public_ns != paper_ns),
            chroma_doc_id=abs_chroma,
            public_chroma_doc_id=pub_abs,
        )
        if (
            paper_pg_id
            and RAG_INGEST_ARXIV_ABSTRACT_VECTOR
            and (abs_summary or "").strip()
        ):
            pg_items.append(
                {
                    "chroma_doc_id": abs_chroma,
                    "content": build_arxiv_abstract_text(
                        arxiv_id,
                        rec.get("title") or (paper.title if paper else None),
                        list(rec.get("authors") or []) or (paper.authors if paper else []),
                        abs_summary,
                    ),
                    "chunk_index": 0,
                    "chunk_role": "paper_summary",
                    "has_figure": False,
                    "has_table": False,
                }
            )

        # 智能入库策略：优先使用 profile 解析结果；保留原常量仅作兼容兜底。
        section_aware_mode = (section_mode_eff or RAG_INGEST_SECTION_MODE or "").strip().lower() == "section_aware"
        sections_for_pg: list[dict] = []
        body_pairs: list[tuple[str, str]] = []

        if section_aware_mode:
            # Re-ingest 时避免旧 chunks 残留（只清理当前 paper namespace）。
            try:
                vector_store.delete_by_where(
                    namespace=paper_ns,
                    where={"arxiv_id": arxiv_id, "type": "arxiv_pdf_full"},
                )
            except Exception:
                pass
            try:
                vector_store.delete_by_where(
                    namespace=paper_ns,
                    where={"arxiv_id": arxiv_id, "type": "pdf_figure"},
                )
            except Exception:
                pass
            try:
                vector_store.delete_by_where(
                    namespace=paper_ns,
                    where={"arxiv_id": arxiv_id, "type": "pdf_table"},
                )
            except Exception:
                pass

            try:
                from langchain_community.document_loaders import PyMuPDFLoader

                loader = PyMuPDFLoader(str(pdf_path))
                page_docs = loader.load()
            except Exception:
                # 若 PyMuPDFLoader 不可用，则退回原全文切分逻辑（保证可运行）。
                section_aware_mode = False

            if section_aware_mode:
                # 以“页”为单位做章节范围粗切
                page_texts: list[tuple[int, str]] = []
                for idx, d in enumerate(page_docs, start=1):
                    content = (d.page_content or "").strip()
                    content = _basic_clean_light(content)
                    page_texts.append((idx, content))

                sections_pages = _extract_sections_from_pdf_pages(page_texts=page_texts)
                prefix = f"{id_root}_body"
                splitter = TokenTextSplitter(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    model_name="gpt-3.5-turbo",
                )

                # body_docs/ids：确保 Chroma 的 chunk_index 与 PG chunk_index（1-based vs 0-based）可对应。
                body_docs: list[Document] = []
                body_ids: list[str] = []
                body_seq0 = 0  # Chroma chunk_index: 0-based
                pg_chunk_items: list[dict] = []

                page_text_by_idx = {int(pi): pt for pi, pt in page_texts}

                for sorder, sec in enumerate(sections_pages):
                    sec_title = str(sec.get("title") or "").strip()
                    sec_number = str(sec.get("section_number") or "").strip()
                    sec_role = str(sec.get("role") or "other").strip() or "other"
                    sec_level = int(sec.get("section_level") or 1)
                    start_page = int(sec.get("start_page") or 1)
                    end_page = int(sec.get("end_page") or start_page)

                    pages_text = [
                        page_text_by_idx.get(p, "")
                        for p in range(start_page, end_page + 1)
                        if page_text_by_idx.get(p, "").strip()
                    ]
                    section_text = "\n\n".join(pages_text).strip()
                    if not section_text:
                        sec_start_chunk = body_seq0 + 1
                        sec_end_chunk = body_seq0
                        sections_for_pg.append(
                            {
                                "order_index": sorder,
                                "section_number": sec_number,
                                "section_level": sec_level,
                                "title": sec_title,
                                "start_page": start_page,
                                "end_page": end_page,
                                "start_chunk_index": sec_start_chunk,
                                "end_chunk_index": sec_end_chunk,
                                "role": sec_role,
                            }
                        )
                        continue

                    chunk_texts = splitter.split_text(section_text)
                    sec_start_chunk = body_seq0 + 1
                    for chunk_text in chunk_texts:
                        ct = (chunk_text or "").strip()
                        if not ct:
                            continue
                        cid = f"{prefix}_{body_seq0:06d}"
                        pg_chunk_items.append(
                            {
                                "chroma_doc_id": cid,
                                "content": ct,
                                "chunk_index": body_seq0 + 1,  # PG: 1-based
                                "chunk_role": "generic",
                                "has_figure": False,
                                "has_table": False,
                                "page_from": int(start_page),
                                "page_to": int(end_page),
                            }
                        )
                        meta = dict(base_meta)
                        meta.update(
                            {
                                "vector_partition": "paper",
                                "chunk_index": int(body_seq0),  # Chroma: 0-based
                                "chunk_role": "generic",
                                "has_figure": False,
                                "has_table": False,
                                "section_title": sec_title,
                                "heading": sec_title,
                                "section_number": sec_number,
                                "section_level": sec_level,
                                "chunk_role_hint": sec_role,
                                "page_from": int(start_page),
                                "page_to": int(end_page),
                            }
                        )
                        body_docs.append(Document(page_content=ct, metadata=meta))
                        body_ids.append(cid)
                        body_pairs.append((cid, ct))
                        body_seq0 += 1
                    sec_end_chunk = body_seq0
                    sections_for_pg.append(
                        {
                            "order_index": sorder,
                            "section_number": sec_number,
                            "section_level": sec_level,
                            "title": sec_title,
                            "start_page": start_page,
                            "end_page": end_page,
                            "start_chunk_index": sec_start_chunk,
                            "end_chunk_index": sec_end_chunk,
                            "role": sec_role,
                        }
                    )

                # 确保 order_index 与列表顺序一致（供 replace_paper_sections + binding 逻辑对齐）
                sections_for_pg = sorted(
                    sections_for_pg,
                    key=lambda x: int(x.get("start_chunk_index") or 1),
                )
                for idx, sec in enumerate(sections_for_pg):
                    sec["order_index"] = idx
                if paper_pg_id:
                    pg_items.extend(pg_chunk_items)
                else:
                    pg_items = []

                # 写入 Chroma
                vector_store.add_documents(
                    documents=body_docs,
                    namespace=paper_ns,
                    ids=body_ids,
                )

        if not body_pairs:
            # 退回原全文切分（旧逻辑不改，保证可用）
            body_pairs = vector_store.add_chunked_text_with_prefixed_ids(
                text,
                namespace=paper_ns,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                id_prefix=f"{id_root}_body",
                extra_metadata={**base_meta, "vector_partition": "paper"},
            )
            if paper_pg_id:
                for i, (cid, chunk_text) in enumerate(body_pairs, start=1):
                    pg_items.append(
                        {
                            "chroma_doc_id": cid,
                            "content": chunk_text,
                            "chunk_index": i,
                            "chunk_role": "generic",
                            "has_figure": False,
                            "has_table": False,
                        }
                    )

        from tools.rag.pdf_figures import embed_pdf_figures_to_namespace
        from tools.rag.pdf_tables import embed_pdf_tables_to_namespace

        fig_chunks_accum: list[dict] = []
        fig_meta_accum: list[dict] = []
        embed_pdf_figures_to_namespace(
            vector_store,
            pdf_path=str(pdf_path),
            parent_id=arxiv_id,
            namespace=paper_ns,
            arxiv_id=arxiv_id,
            chroma_id_prefix=id_root,
            pg_chunks_out=fig_chunks_accum if paper_pg_id else None,
            pg_figures_out=fig_meta_accum if paper_pg_id else None,
            enabled=bool(enable_figures_eff and RAG_PDF_FIGURE_CAPTION_ENABLED),
        )
        next_idx = len(body_pairs) + 1
        if paper_pg_id and fig_chunks_accum:
            for row in fig_chunks_accum:
                row["chunk_index"] = next_idx
                next_idx += 1
                pg_items.append(row)

        table_chunks_accum: list[dict] = []
        table_meta_accum: list[dict] = []
        embed_pdf_tables_to_namespace(
            vector_store,
            pdf_path=str(pdf_path),
            parent_id=arxiv_id,
            namespace=paper_ns,
            arxiv_id=arxiv_id,
            chroma_id_prefix=id_root,
            pg_chunks_out=table_chunks_accum if paper_pg_id else None,
            pg_tables_out=table_meta_accum if paper_pg_id else None,
            enabled=bool(enable_tables_eff and RAG_PDF_TABLE_EXTRACT_ENABLED),
        )
        if paper_pg_id and table_chunks_accum:
            for row in table_chunks_accum:
                row["chunk_index"] = next_idx
                next_idx += 1
                pg_items.append(row)

        chunks_ok = False
        sec_ok = False
        sec_sum_ok = False
        summary_view_ok = False
        role_cache_ok = False
        section_tree_cache_ok = False
        summary_cache_ok = False
        if paper_pg_id:
            # 幂等索引确保：仅优化 IO/检索性能，不改变检索逻辑。
            try:
                paper_ingest_repo.ensure_ingest_performance_indexes()
            except Exception:
                pass
            chunks_ok = bool(paper_ingest_repo.replace_paper_chunks(paper_pg_id, pg_items))
            # 自动抽 section + section summary + summary bundle + Redis 预热
            body_only = [(cid, txt) for cid, txt in body_pairs if (txt or "").strip()]
            if body_only:
                if section_aware_mode and sections_for_pg:
                    sections = sections_for_pg
                else:
                    sections = _extract_sections_from_body_pairs(body_only)
                section_items = [
                    {
                        "order_index": int(s["order_index"]),
                        "section_level": int(s.get("section_level") or 1),
                        "section_number": str(s.get("section_number") or ""),
                        "title": str(s.get("title") or ""),
                        "page_start": s.get("start_page"),
                        "page_end": s.get("end_page"),
                    }
                    for s in sections
                ]
                sec_id_by_order = paper_ingest_repo.replace_paper_sections(paper_pg_id, section_items)
                sec_ok = bool(sec_id_by_order)
                if sec_ok:
                    cidx_to_sec_order = _build_section_binding(sections, 1, len(body_only))
                    cidx_to_sec_id = {
                        int(cidx): int(sec_id_by_order.get(int(sorder), 0))
                        for cidx, sorder in cidx_to_sec_order.items()
                        if int(sec_id_by_order.get(int(sorder), 0)) > 0
                    }
                    paper_ingest_repo.bind_chunk_sections_by_index(paper_pg_id, cidx_to_sec_id)
                    sec_summaries, bundle = _build_section_summaries(sections, body_only)
                    sec_summary_rows = []
                    for row in sec_summaries:
                        sid = int(sec_id_by_order.get(int(row["order_index"]), 0))
                        if sid <= 0:
                            continue
                        sec_summary_rows.append(
                            {
                                "section_id": sid,
                                "section_role": row.get("section_role"),
                                "summary_text": row.get("summary_text"),
                                "keywords_json": row.get("keywords_json") or [],
                            }
                        )
                    sec_sum_ok = bool(
                        paper_ingest_repo.replace_section_summaries(paper_pg_id, sec_summary_rows)
                    )
                    summary_view_ok = bool(
                        paper_ingest_repo.upsert_paper_summary_view(paper_pg_id, bundle)
                    )
                    # Redis 预热：section tree / section roles / summary bundle
                    try:
                        tree_payload = []
                        roles_payload: dict[str, list[int]] = {}
                        for s in sections:
                            sid = int(sec_id_by_order.get(int(s["order_index"]), 0))
                            if sid <= 0:
                                continue
                            role = str(s.get("role") or "other")
                            roles_payload.setdefault(role, []).append(sid)
                            tree_payload.append(
                                {
                                    "id": sid,
                                    "title": s.get("title"),
                                    "section_number": s.get("section_number"),
                                    "section_level": s.get("section_level"),
                                    "order_index": s.get("order_index"),
                                }
                            )
                        cache_section_tree(paper_pg_id, json.dumps(tree_payload, ensure_ascii=False))
                        section_tree_cache_ok = True
                        set_section_roles(paper_pg_id, roles_payload)
                        role_cache_ok = True
                        set_summary_bundle(paper_pg_id, bundle)
                        summary_cache_ok = True
                    except Exception:
                        pass
        fig_ok = True
        table_ok = True
        table_write_enabled = False
        if paper_pg_id and bool(enable_tables_eff and RAG_PDF_TABLE_EXTRACT_ENABLED):
            try:
                import pdfplumber  # noqa: F401
                _pdfplumber_ok = True
            except ImportError:
                _pdfplumber_ok = False
            if _pdfplumber_ok:
                table_write_enabled = True
        if paper_pg_id:
            # 图/表元数据写入互不依赖，改并行提交以减少等待。
            def _write_fig() -> bool:
                if fig_meta_accum:
                    return bool(
                        paper_ingest_repo.replace_paper_figures(paper_pg_id, fig_meta_accum)
                    )
                return True

            def _write_table() -> bool:
                if table_write_enabled:
                    return bool(
                        paper_ingest_repo.replace_paper_tables(
                            paper_pg_id, table_meta_accum or []
                        )
                    )
                return True

            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_fig = ex.submit(_write_fig)
                fut_tab = ex.submit(_write_table)
                fig_ok = bool(fut_fig.result())
                table_ok = bool(fut_tab.result())
        if paper_pg_id:
            pg_note = (
                f"\n- PostgreSQL: paper_id={paper_pg_id}，paper_chunks 同步"
                f"{'成功' if chunks_ok else '（失败或跳过）'}"
                f"，sections：{'成功' if sec_ok else '跳过/失败'}"
                f"，section_summaries：{'成功' if sec_sum_ok else '跳过/失败'}"
                f"，summary_bundle：{'成功' if summary_view_ok else '跳过/失败'}"
                f"，插图元数据：{'成功' if fig_ok else '失败'}"
                f"，表格元数据 paper_tables：{'成功' if table_ok else '失败/跳过'}"
                f"，Redis预热(tree/roles/bundle)："
                f"{'成功' if (section_tree_cache_ok and role_cache_ok and summary_cache_ok) else '部分/失败'}\n"
            )

        namespaces_to_record.add(paper_ns)

        if do_public and public_ns and public_ns != paper_ns:
            vector_store.add_chunked_text_with_prefixed_ids(
                text,
                namespace=public_ns,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                id_prefix=f"{id_root}_body_pub",
                extra_metadata={
                    **base_meta,
                    "vector_partition": "public",
                    "paper_vector_namespace": paper_ns,
                },
            )
            namespaces_to_record.add(public_ns)
        elif do_public and public_ns == paper_ns:
            namespaces_to_record.add(paper_ns)

        upsert_paper(
            LocalPaper(
                arxiv_id=arxiv_id,
                pdf_path=str(pdf_path.as_posix()),
                title=rec.title,
                authors=rec.authors or [],
                published=rec.published,
                url=rec.url,
                namespaces=sorted(namespaces_to_record),
            )
        )
    except Exception as e:
        return (
            f"已下载并登记到本地论文库：{pdf_path.as_posix()}\n"
            f"但全文向量入库失败：{e}\n"
            f"你仍可先用 PDF 路径或 papers/view 查看。"
        )

    view_url = f"/papers/view/{pdf_path.name}"
    public_line = (
        f"- 公共向量库 namespace: {public_ns}（与论文库共用全文索引，便于在 default 等会话中检索）\n"
        if do_public and public_ns != paper_ns
        else (
            f"- 公共向量库: 与论文 namespace 相同（{public_ns}），未重复写入。\n"
            if do_public
            else "- 公共向量库: 已跳过（配置 RAG_INGEST_ARXIV_ALSO_EMBED_PUBLIC=0）\n"
        )
    )
    abs_note = (
        "另：已将 arXiv 标题/作者/摘要写入同一向量 namespace，便于 hybrid/BM25 命中摘要。\n"
        if RAG_INGEST_ARXIV_ABSTRACT_VECTOR
        else ""
    )
    return (
        f"已完成：PDF → data/papers、SQLite 论文库、向量全文入库。\n"
        f"{abs_note}"
        f"{pg_note}"
        f"- arxiv_id: {arxiv_id}\n"
        f"- pdf: {pdf_path.as_posix()}\n"
        f"- view: {view_url}\n"
        f"- 论文专用向量 namespace: {paper_ns}\n"
        f"{public_line}"
        f"说明：你仍可使用原来的会话 namespace（如 conv_001）聊天；精读该篇时可问「这篇论文……」或指定 {paper_ns}。\n"
    )
