from __future__ import annotations

from langchain_openai import ChatOpenAI

from config import (
    LOCAL_EMBED_DEVICE,
    LOCAL_EMBED_MODEL,
    QWEN_CHAT_MODEL,
    RAG_NETWORK_MODE,
)
from runtime_settings import RUNTIME

"""
Qwen 模型配置：
- online：Chat 走远程 OpenAI 兼容服务；Embeddings 走本地 sentence-transformers
- offline：Chat 走本地 OpenAI 兼容服务；Embeddings 走本地 sentence-transformers

环境变量：
- QWEN_API_KEY: 通义千问兼容 OpenAI 协议的 key（必填，勿将 key 提交到仓库）
- QWEN_BASE_URL: 兼容模式的 base_url（默认 dashscope）
- QWEN_CHAT_MODEL: 对话模型（默认见 config.QWEN_CHAT_MODEL，多模态建议 qwen-vl-plus 等）
- QWEN_MAX_TOKENS: 单次回复最大**生成** token 上限（默认 8192）
- QWEN_TIMEOUT: 请求超时秒数（默认 120，带图场景建议加大）
"""

_QWEN_API_KEY = RUNTIME.qwen_chat_api_key or ""
_QWEN_CHAT_BASE_URL = RUNTIME.qwen_chat_base_url
_QWEN_CHAT_API_KEY = RUNTIME.qwen_chat_api_key or _QWEN_API_KEY

_max_tokens = RUNTIME.qwen_max_tokens
_max_tokens = max(256, min(32768, _max_tokens))
_offline_cap = RUNTIME.qwen_offline_max_tokens
_offline_cap = max(16, min(2048, _offline_cap))

_timeout = RUNTIME.qwen_timeout
_timeout = max(15.0, min(600.0, _timeout))

if RAG_NETWORK_MODE == "offline":
    if not _QWEN_CHAT_BASE_URL:
        raise RuntimeError(
            "RAG_NETWORK_MODE=offline 时必须提供 QWEN_CHAT_BASE_URL（本地 OpenAI 兼容地址）。"
        )
    if not QWEN_CHAT_MODEL:
        raise RuntimeError(
            "RAG_NETWORK_MODE=offline 时必须提供 QWEN_CHAT_MODEL（本地服务中的模型名）。"
        )
else:
    if not _QWEN_CHAT_BASE_URL:
        raise RuntimeError(
            "RAG_NETWORK_MODE=online 时缺少 QWEN_CHAT_BASE_URL/QWEN_BASE_URL。"
        )
    if not (_QWEN_CHAT_API_KEY or _QWEN_API_KEY):
        raise RuntimeError(
            "RAG_NETWORK_MODE=online 时缺少 QWEN_CHAT_API_KEY/QWEN_API_KEY。"
        )

# 离线模型常见上下文窗口较小（例如 512）；默认限制生成长度，避免 400 超上下文。
if RAG_NETWORK_MODE == "offline":
    _max_tokens = min(_max_tokens, _offline_cap)

# 对话模型：online/offline 共用 OpenAI 兼容客户端，配置来源由模式决定
qwen = ChatOpenAI(
    api_key=_QWEN_CHAT_API_KEY or "dummy-key-set-QWEN_CHAT_API_KEY",
    base_url=_QWEN_CHAT_BASE_URL,
    model=QWEN_CHAT_MODEL,
    temperature=0.7,
    max_tokens=_max_tokens,
    timeout=_timeout,
)

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "固定本地 Embedding 需要 langchain-community 的 HuggingFaceEmbeddings。"
    ) from exc

# Embedding 固定使用本地 Qwen3-Embedding-0.6B（或 LOCAL_EMBED_MODEL 指定路径），
# 避免 online/offline 切换时向量库语义空间不一致。
qwen_embeddings = HuggingFaceEmbeddings(
    model_name=RUNTIME.local_embed_model or LOCAL_EMBED_MODEL,
    model_kwargs={"device": RUNTIME.local_embed_device or LOCAL_EMBED_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
