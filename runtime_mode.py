from __future__ import annotations

# 仅用于快速切换运行网络模式（无需 shell source，也无需额外启动脚本）。
#
# 可选值：
#   - "online"：Chat + Embeddings 都走远程/本地 OpenAI 兼容服务（但 embeddings 在本项目里仍会使用本地 sentence-transformers）
#   - "offline"：Chat 走本地 OpenAI 兼容服务；Embeddings 使用本地 sentence-transformers
#
# 建议：如果你不想写死，保持为空字符串 ""，此时按环境变量 `RAG_NETWORK_MODE` 生效。
RAG_NETWORK_MODE_OVERRIDE = "online"

