from __future__ import annotations

import json
from typing import Any

from tools.storage.redis.cache import _redis_client
from tools.storage.redis.keys import RedisKeys


def get_chunk_neighbors_cached(chunk_id: int) -> dict[str, Any] | None:
    cli = _redis_client()
    if cli is None:
        return None
    try:
        raw = cli.get(RedisKeys().chunk_neighbors(chunk_id))
        if not raw:
            return None
        data = json.loads(raw)
        return dict(data) if isinstance(data, dict) else None
    except Exception:
        return None


def set_chunk_neighbors_cached(chunk_id: int, payload: dict[str, Any], ttl_sec: int = 86400) -> None:
    cli = _redis_client()
    if cli is None:
        return
    try:
        cli.setex(
            RedisKeys().chunk_neighbors(chunk_id),
            max(60, int(ttl_sec)),
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception:
        pass

