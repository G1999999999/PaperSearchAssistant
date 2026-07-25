from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_jsonl(rel_path: str, record: dict[str, Any]) -> None:
    """追加一行 JSON（审计 / 冷归档）。"""
    p = Path(rel_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
