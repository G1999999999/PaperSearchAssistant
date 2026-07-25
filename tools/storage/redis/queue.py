from __future__ import annotations

import json
import os
from typing import Any


def _client():
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


def enqueue_job(queue_name: str, payload: dict[str, Any]) -> bool:
    """可选：后台 ingest / embed 队列；无 Redis 时返回 False。"""
    cli = _client()
    if cli is None:
        return False
    try:
        prefix = (os.getenv("REDIS_PREFIX", "psa2") or "psa2").strip()
        key = f"{prefix}:queue:{queue_name}"
        cli.rpush(key, json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False
