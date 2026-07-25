"""
从 PDF 抽取内嵌位图，可选调用 VL 生成 caption 并写入向量库（与全文文本 chunk 并列）。

插图按页内纵向位置排序，并与页面上按阅读顺序出现的 Figure N / 图 N 做配对，写入 ``figure_number``。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import (
    PDF_EXTRACT_IMAGE_MAX_PER_PAPER,
    RAG_FIGURE_CAPTION_MAX_TOKENS,
    RAG_PDF_FIGURE_CAPTION_ENABLED,
)
from tools.rag.multimodal_content import image_file_to_data_url
from tools.rag.time_utils import add_timestamp_metadata

_FIG_REF_NUM_PAT = re.compile(
    r"\b(?:Figure|Fig\.|FIG|图)\s*(?:[:\-–—]?\s*)?([0-9]+(?:\.[0-9]+)?[A-Za-z]?)\b",
    re.IGNORECASE,
)


def _figure_numbers_unique_reading_order(page_text: str) -> list[str]:
    """页内第一次出现的 Figure / 图 编号序列（去重保序）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for m in sorted(_FIG_REF_NUM_PAT.finditer(page_text or ""), key=lambda x: x.start()):
        n = str(m.group(1) or "").strip()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _caption_line_for_figure(page_text: str, figure_num: str) -> str | None:
    """取包含 ``Figure N`` / ``图 N`` 的一行作为 PDF 侧图注片段。"""
    fn = (figure_num or "").strip()
    if not fn:
        return None
    for raw in (page_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if fn in line and _FIG_REF_NUM_PAT.search(line):
            return line[:800] + ("…" if len(line) > 800 else "")
    return None


def extract_images_from_pdf(
    pdf_path: str | Path,
    out_dir: Path,
    *,
    stem: str,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """使用 PyMuPDF 写出内嵌图片；按页内图像纵向位置与 Figure N 配对。

    返回项含：path, page, index, figure_number?, caption_text_pdf?
    """
    cap = max_images if max_images is not None else PDF_EXTRACT_IMAGE_MAX_PER_PAPER
    cap = max(1, min(500, int(cap or 1)))
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    seen_xref: set[int] = set()

    try:
        doc = fitz.open(str(pdf))
    except Exception:
        return []

    try:
        img_idx = 0
        for page_num in range(len(doc)):
            if len(results) >= cap:
                break
            page = doc[page_num]
            page_text = page.get_text("text") or ""
            fig_nums = _figure_numbers_unique_reading_order(page_text)
            candidates: list[tuple[float, int]] = []
            for img in page.get_images(full=True):
                if len(results) >= cap:
                    break
                xref = int(img[0])
                if xref in seen_xref:
                    continue
                try:
                    rects = page.get_image_rects(xref)
                    y0 = min(float(r.y0) for r in rects) if rects else 0.0
                except Exception:
                    y0 = 0.0
                candidates.append((y0, xref))
            candidates.sort(key=lambda x: x[0])
            slot = 0
            for _y0, xref in candidates:
                if len(results) >= cap:
                    break
                if xref in seen_xref:
                    continue
                seen_xref.add(xref)
                fnum: str | None = None
                cap_pdf: str | None = None
                if slot < len(fig_nums):
                    fnum = fig_nums[slot]
                    cap_pdf = _caption_line_for_figure(page_text, fnum)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    name = f"{stem}_p{page_num + 1}_i{img_idx}.png"
                    dest = out_dir / name
                    pix.save(str(dest))
                    pix = None  # noqa: F841
                    if dest.exists() and dest.stat().st_size > 0:
                        item = {
                            "path": dest.resolve(),
                            "page": page_num + 1,
                            "index": img_idx,
                        }
                        if fnum:
                            item["figure_number"] = fnum
                        if cap_pdf:
                            item["caption_text_pdf"] = cap_pdf
                        results.append(item)
                        img_idx += 1
                        slot += 1
                except Exception:
                    continue
    finally:
        doc.close()

    return results


def caption_figure_image(image_path: Path, *, caption_hint: str | None = None) -> str:
    """单张插图生成论文导向描述（失败返回空串）。"""
    from langchain_core.messages import HumanMessage

    from models_qwen import qwen

    url = image_file_to_data_url(image_path)
    if not url:
        return ""
    hint = (caption_hint or "").strip()
    prompt = (
        "你是一个专门阅读学术论文图片的助手，擅长分析计算机、人工智能、"
        "机器学习、计算机视觉等论文中的图像内容。\n\n"
        "你的任务不是简单描述图片，而是像科研人员阅读论文一样，对图片进行结构化理解与解释。\n\n"
        "请按以下步骤分析图片：\n"
        "1. 识别图片类型\n"
        "- 判断该图属于哪一类：方法框架图 / 流程图 / 网络结构图 / 曲线图 / 柱状图 / 折线图 / "
        "散点图 / 热力图 / 表格截图 / 定性可视化图 / 消融实验图 / 公式图 / 其他\n"
        "- 如果是一张复合图，指出每个子图分别是什么类型\n\n"
        "2. 提取图中的关键信息\n"
        "- 提取标题、子图编号（如(a)(b)(c)）\n"
        "- 提取核心模块、箭头关系、输入输出\n"
        "- 若是实验图，提取横轴、纵轴、图例、对比方法、指标名称\n"
        "- 若是表格或结果图，提取最重要的比较对象和结果趋势\n"
        "- 若有文字标签，概括其含义\n\n"
        "3. 解释这张图在论文中想表达什么\n"
        "- 说明该图是在表达方法流程、模型结构、实验性能、消融结论或可视化效果\n"
        "- 作者希望通过这张图证明什么\n\n"
        "4. 给出通俗解释\n"
        "- 用简洁易懂的话解释“为什么这样画、它说明了什么”\n\n"
        "5. 指出不确定点\n"
        "- 看不清、信息缺失、只能推测的地方，请明确写“无法从图中确定”\n"
        "- 禁止编造图中没有的信息或场景化猜测（例如椅子、绿幕、人物身份等）\n\n"
        "输出格式要求：\n"
        "【图类型】\n"
        "【图中关键信息】\n"
        "【这张图想说明什么】\n"
        "【通俗解释】\n"
        "【不确定或需结合正文确认的点】"
    )
    if hint:
        prompt += f"\n已知图注线索（可能截断，仅作弱提示）: {hint[:400]}\n"
    try:
        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]
        )
        lc = qwen.bind(max_tokens=RAG_FIGURE_CAPTION_MAX_TOKENS, temperature=0.2)
        resp = lc.invoke([msg])
        text = (resp.content if hasattr(resp, "content") else str(resp)) or ""
        return str(text).strip()
    except Exception:
        return ""


def embed_pdf_figures_to_namespace(
    store: Any,
    *,
    pdf_path: str | Path,
    parent_id: str,
    namespace: str,
    arxiv_id: str | None = None,
    chroma_id_prefix: str | None = None,
    pg_chunks_out: list | None = None,
    pg_figures_out: list | None = None,
    enabled: bool | None = None,
) -> int:
    """抽取 PDF 图片、caption 后写入 namespace。返回新增 Document 条数。

    ``chroma_id_prefix`` 非空时为每条分配稳定 Chroma id（``{prefix}_fig_{i:04d}``），便于 PostgreSQL 对齐。
    ``pg_chunks_out`` / ``pg_figures_out`` 若传入，会追加用于 ``paper_chunks`` / ``paper_figures`` 的记录（由上层填 chunk_index）。"""
    use_enabled = RAG_PDF_FIGURE_CAPTION_ENABLED if enabled is None else bool(enabled)
    if not use_enabled:
        return 0

    pdf = Path(pdf_path)
    if not pdf.is_file():
        return 0

    stem = (arxiv_id or pdf.stem).replace("/", "_").replace("\\", "_") or pdf.stem
    out_dir = Path("data/paper_figures") / stem
    extracted = extract_images_from_pdf(pdf, out_dir, stem=stem)
    if not extracted:
        return 0

    docs: list[Document] = []
    vl_caps: list[str] = []
    rel_root = Path.cwd()

    for item in extracted:
        p: Path = item["path"]
        page = int(item.get("page") or 0)
        fnum_raw = item.get("figure_number")
        fnum = str(fnum_raw).strip() if fnum_raw is not None else ""
        cap_pdf = (item.get("caption_text_pdf") or "").strip() or None
        vl_caption = caption_figure_image(p, caption_hint=cap_pdf)
        if not vl_caption:
            vl_caption = f"（第 {page} 页插图，自动描述失败）"
        vl_caps.append(vl_caption)
        head = f"[Figure{f' {fnum}' if fnum else ''} page {page}]"
        parts = [head]
        if cap_pdf:
            parts.append(f"PDF_caption: {cap_pdf}")
        parts.append(vl_caption)
        body = "\n".join(parts)
        try:
            rel = str(p.resolve().relative_to(rel_root))
        except ValueError:
            rel = str(p.resolve())
        meta = add_timestamp_metadata(
            {
                "parent_id": parent_id,
                "source": rel,
                "image_path": rel,
                "page": page,
                "modality": "figure",
                "type": "pdf_figure",
                "arxiv_id": arxiv_id or parent_id,
                "chunk_role": "figure",
                "has_figure": True,
                "caption_text_pdf": cap_pdf or "",
            }
        )
        if fnum:
            meta["figure_number"] = fnum
        docs.append(Document(page_content=body, metadata=meta))

    if not docs:
        return 0
    prefix = (chroma_id_prefix or "").strip() or None
    ids = [f"{prefix}_fig_{i:04d}" for i in range(len(docs))] if prefix else None
    n = store.add_documents(docs, namespace=namespace, extra_metadata={}, ids=ids)
    if prefix and pg_chunks_out is not None:
        for i, d in enumerate(docs):
            pg = int((d.metadata or {}).get("page") or 0)
            pg_chunks_out.append(
                {
                    "chroma_doc_id": ids[i] if ids else "",
                    "content": d.page_content or "",
                    "chunk_role": "figure",
                    "has_figure": True,
                    "has_table": False,
                    "chunk_index": -1,
                    "page_from": pg if pg > 0 else None,
                    "page_to": pg if pg > 0 else None,
                }
            )
    if pg_figures_out is not None:
        for item, vl_line, d in zip(extracted, vl_caps, docs):
            p = item["path"]
            page = int(item.get("page") or 0)
            fnum = str(item.get("figure_number") or "").strip() or None
            cap_pdf = (item.get("caption_text_pdf") or "").strip() or None
            try:
                rel = str(p.resolve().relative_to(rel_root))
            except ValueError:
                rel = str(p.resolve())
            pg_figures_out.append(
                {
                    "page_no": page,
                    "image_path": rel,
                    "figure_number": fnum[:64] if fnum else None,
                    "caption_text": (cap_pdf[:8000] if cap_pdf else None),
                    "vision_summary": (vl_line[:8000] if vl_line else None),
                    "ocr_text": None,
                }
            )
    return n
