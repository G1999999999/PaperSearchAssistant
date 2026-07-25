from __future__ import annotations

import json
from typing import Any

from tools.storage.redis.cache import _redis_client
from tools.storage.redis.keys import RedisKeys


def get_section_roles(paper_id: int) -> dict[str, list[int]] | None:
    cli = _redis_client()
    if cli is None:
        return None
    try:
        raw = cli.get(RedisKeys().paper_section_roles(paper_id))
        if not raw:
            return None
        data = json.loads(raw)
        if isinstance(data, dict):
            out: dict[str, list[int]] = {}
            for k, v in data.items():
                if isinstance(v, list):
                    out[str(k)] = [int(x) for x in v]
            return out
    except Exception:
        return None
    return None


def set_section_roles(paper_id: int, roles: dict[str, list[int]], ttl_sec: int = 86400) -> None:
    cli = _redis_client()
    if cli is None:
        return
    try:
        cli.setex(
            RedisKeys().paper_section_roles(paper_id),
            max(60, int(ttl_sec)),
            json.dumps(roles, ensure_ascii=False),
        )
    except Exception:
        pass


def get_summary_bundle(paper_id: int) -> dict[str, Any] | None:
    cli = _redis_client()
    if cli is None:
        return None
    try:
        raw = cli.get(RedisKeys().paper_summary_bundle(paper_id))
        if not raw:
            return None
        data = json.loads(raw)
        return dict(data) if isinstance(data, dict) else None
    except Exception:
        return None


def set_summary_bundle(paper_id: int, bundle: dict[str, Any], ttl_sec: int = 86400) -> None:
    cli = _redis_client()
    if cli is None:
        return
    try:
        cli.setex(
            RedisKeys().paper_summary_bundle(paper_id),
            max(60, int(ttl_sec)),
            json.dumps(bundle, ensure_ascii=False),
        )
    except Exception:
        pass

