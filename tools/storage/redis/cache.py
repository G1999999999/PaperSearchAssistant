from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

from tools.storage.redis.keys import RedisKeys


def _redis_client():
    url = (os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        return None
    try:
        import redis

        cli = redis.Redis.from_url(url, decode_responses=True)
        cli.ping()
        return cli
    except Exception:
        return None


def search_cache_get(namespace: str, question: str, strategy: str) -> Optional[list[dict[str, Any]]]:
    """轻量缓存：仅存可 JSON 序列化的命中摘要（非完整 Document）。"""
    cli = _redis_client()
    if cli is None:
        return None
    try:
        from config import RAG_PAPER_RETRIEVAL_CACHE_TTL

        if int(RAG_PAPER_RETRIEVAL_CACHE_TTL or 0) <= 0:
            return None
        raw_key = f"{namespace}|{strategy}|{question}"
        h = hashlib.sha256(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:32]
        keys = RedisKeys()
        blob = cli.get(keys.search_paper(h))
        if not blob:
            return None
        data = json.loads(blob)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def search_cache_set(
    namespace: str, question: str, strategy: str, payload: list[dict[str, Any]]
) -> None:
    cli = _redis_client()
    if cli is None:
        return
    try:
        from config import RAG_PAPER_RETRIEVAL_CACHE_TTL

        ttl = int(RAG_PAPER_RETRIEVAL_CACHE_TTL or 0)
        if ttl <= 0:
            return
        raw_key = f"{namespace}|{strategy}|{question}"
        h = hashlib.sha256(raw_key.encode("utf-8", errors="ignore")).hexdigest()[:32]
        keys = RedisKeys()
        cli.setex(keys.search_paper(h), ttl, json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass


def cache_section_tree(paper_id: int, tree_json: str, ttl_sec: int = 86400) -> None:
    cli = _redis_client()
    if cli is None:
        return
    try:
        keys = RedisKeys()
        cli.setex(keys.paper_sections_tree(paper_id), max(60, int(ttl_sec)), tree_json)
    except Exception:
        pass


def get_section_tree(paper_id: int) -> Optional[str]:
    cli = _redis_client()
    if cli is None:
        return None
    try:
        keys = RedisKeys()
        return cli.get(keys.paper_sections_tree(paper_id))
    except Exception:
        return None
