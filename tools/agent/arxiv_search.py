from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import arxiv
import requests

from config import ARXIV_PDF_DOWNLOAD_TIMEOUT, ARXIV_PDF_USER_AGENT


@dataclass
class Paper:
    title: str
    authors: list[str]
    summary: str
    url: str
    published: datetime


def get_arxiv_id(paper: Paper) -> str:
    """根据论文的 URL 提取 arXiv ID。"""

    return paper.url.rstrip("/").split("/")[-1]


def get_pdf_url(arxiv_id: str) -> str:
    """根据 arXiv ID 构造 PDF 下载链接。"""

    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(
    arxiv_id: str,
    dest_dir: str | Path = "data/papers",
) -> Path:
    """下载指定 arXiv 论文的 PDF 到本地并返回路径。

    注意：只有显式调用本函数时才会下载和保存 PDF。
    """

    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir_path / f"{arxiv_id}.pdf"

    url = get_pdf_url(arxiv_id)
    headers = {
        "User-Agent": ARXIV_PDF_USER_AGENT
        or "Mozilla/5.0 (compatible; PaperSearchAssistant/1.0)",
        "Accept": "application/pdf,application/octet-stream,*/*",
    }
    timeout = float(ARXIV_PDF_DOWNLOAD_TIMEOUT)
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            with pdf_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        if not pdf_path.exists() or pdf_path.stat().st_size < 1024:
            pdf_path.unlink(missing_ok=True)
            raise RuntimeError("文件过小或为空，可能不是有效 PDF（请检查网络/代理）。")
        sig = pdf_path.read_bytes()[:4]
        if sig != b"%PDF":
            pdf_path.unlink(missing_ok=True)
            raise RuntimeError(
                "保存的内容不是 PDF 文件头（可能被拦截返回了 HTML/登录页等）。"
                "请检查 HTTPS_PROXY 或浏览器能否打开同一链接。"
            )
        return pdf_path
    except Exception as e:
        hint = (
            "若在国内网络，请设置代理后重试，例如："
            "export HTTPS_PROXY=http://127.0.0.1:7890"
        )
        raise RuntimeError(
            f"从 arXiv 下载 PDF 失败：{url}\n原因：{e}\n{hint}"
        ) from e


def search_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    category: Optional[str] = None,
) -> List[Paper]:
    """在 arXiv 上检索论文并返回标准化后的 Paper 列表。

    - query: 关键词或布尔查询字符串
    - max_results: 返回的最大论文数
    - sort_by: 排序方式，当前支持 \"relevance\" 或 \"lastUpdatedDate\"
    - category: 可选的 arXiv 分类（例如 \"cs.CL\"、\"cs.LG\"），会拼接到查询中
    """

    if category:
        full_query = f"({query}) AND cat:{category}"
    else:
        full_query = query

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=sort_criterion,
    )

    papers: list[Paper] = []
    try:
        for result in search.results():
            papers.append(
                Paper(
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    summary=result.summary,
                    url=result.entry_id,
                    published=result.published,
                )
            )
    except Exception:
        # 出错时返回当前已收集到的结果（可能为空），避免直接抛到上层
        return papers

    return papers


def search_arxiv_for_ingest_disambiguation(
    query: str,
    *,
    display_max: int = 8,
    fetch_max: int = 30,
    sort_by: str = "relevance",
) -> List[Paper]:
    """供「按标题入库」选论文：优先 ``ti:\"...\"`` 精确短语，再补充全文检索；多取再截断。

    避免仅用 ``all:`` + 少条数时，经典论文被同名梗/新作挤出前 N 条。
    """
    q = (query or "").strip()
    if not q:
        return []
    # ti: 短语内避免未闭合引号
    safe_phrase = q.replace('"', " ").strip()
    if not safe_phrase:
        return []

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
    cap = max(display_max, min(50, int(fetch_max)))

    def _iter_papers(api_query: str) -> list[Paper]:
        out: list[Paper] = []
        search = arxiv.Search(
            query=api_query,
            max_results=cap,
            sort_by=sort_criterion,
        )
        try:
            for result in search.results():
                out.append(
                    Paper(
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        summary=result.summary,
                        url=result.entry_id,
                        published=result.published,
                    )
                )
        except Exception:
            pass
        return out

    seen: set[str] = set()
    merged: list[Paper] = []

    # 1) 标题短语优先（arXiv API: ti:"exact phrase"）
    ti_query = f'ti:"{safe_phrase}"'
    for p in _iter_papers(ti_query):
        aid = get_arxiv_id(p)
        if aid and aid not in seen:
            seen.add(aid)
            merged.append(p)

    # 2) 仍不足则补充默认全文检索（与 search_arxiv 行为一致）
    if len(merged) < display_max:
        for p in _iter_papers(q):
            aid = get_arxiv_id(p)
            if aid and aid not in seen:
                seen.add(aid)
                merged.append(p)
            if len(merged) >= cap:
                break

    return merged[:display_max]


def get_paper_by_id(arxiv_id: str) -> Paper | None:
    """按 arXiv ID 获取论文元数据（不下载 PDF）。"""
    arxiv_id = (arxiv_id or "").strip().replace("arXiv:", "").replace("ARXIV:", "")
    if not arxiv_id:
        return None
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        for result in search.results():
            return Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                summary=result.summary,
                url=result.entry_id,
                published=result.published,
            )
    except Exception:
        return None
    return None

