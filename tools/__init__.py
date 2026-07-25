"""
PaperSearchAssistant `tools` 包目录结构说明。

目前实现已按职责拆成三大子目录：
1) `tools/agent/`：Agent 运行/工具注册/路由/审批/回调等
2) `tools/rag/`：RAG/检索（向量存储、BM25/重排、文本/语言处理等）
3) `tools/storage/`：本地持久化（SQLite 论文库）与长期/对话记忆
"""

