from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os


ROOT = Path(__file__).resolve().parent

# 统一运行时环境加载顺序：.env.runtime -> .env
load_dotenv(ROOT / ".env.runtime")
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class RuntimeSettings:
    rag_network_mode: str
    qwen_chat_base_url: str
    qwen_chat_api_key: str
    qwen_chat_model: str
    qwen_max_tokens: int
    qwen_timeout: float
    qwen_offline_max_tokens: int
    local_embed_model: str
    local_embed_device: str


def _to_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


RUNTIME = RuntimeSettings(
    rag_network_mode=(os.getenv("RAG_NETWORK_MODE", "online") or "online").strip().lower(),
    qwen_chat_base_url=(
        os.getenv(
            "QWEN_CHAT_BASE_URL",
            os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip(),
    qwen_chat_api_key=(
        os.getenv("QWEN_CHAT_API_KEY", os.getenv("QWEN_API_KEY", "")) or ""
    ).strip(),
    qwen_chat_model=(os.getenv("QWEN_CHAT_MODEL", "") or "").strip(),
    qwen_max_tokens=_to_int(os.getenv("QWEN_MAX_TOKENS"), 8192),
    qwen_timeout=_to_float(os.getenv("QWEN_TIMEOUT"), 120.0),
    qwen_offline_max_tokens=_to_int(os.getenv("QWEN_OFFLINE_MAX_TOKENS"), 128),
    local_embed_model=(
        os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3"
    ).strip(),
    local_embed_device=(os.getenv("LOCAL_EMBED_DEVICE", "cpu") or "cpu").strip(),
)
