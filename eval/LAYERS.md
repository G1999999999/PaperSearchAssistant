# PaperSearchAssistant2 分层评测说明

目标：**每一层**都有对应测法；能离线、确定性的在 CI 跑，需要向量库 / LLM 的单独打标与跳过。

## 层级与测试入口

| 层级 | 测什么 | 自动化 | 测试文件 / 命令 |
|------|--------|--------|------------------|
| L0 指标工具 | 关键词召回等辅助指标 | 纯本地 | `tests/layers/test_layer_00_answer_metrics.py` |
| L1 入口路由 | `build_query_route`、LangGraph `_route_node` 的 `selected_path` | 纯本地 | `eval/fixtures/golden_routing.json`、`test_layer_01_routing.py` |
| L2 查询理解 | `analyze_paper_query`、`subquestions_for_decomposition` | 纯本地 | `golden_query_understanding.json`、`test_layer_02_query_understanding.py` |
| L3 融合与排序 | `merge_ranked_lists`、`rrf_fusion`、`weighted_rrf_fusion` | 纯本地 | `test_layer_03_fusion_merge.py` |
| L4 检索评判契约 | `_parse_judge_json`、空上下文时的默认分支 | 纯本地（不调用 LLM） | `test_layer_04_retrieval_judge_contract.py` |
| L5 稀疏检索 BM25 | `build_bm25_index` + `bm25_top_k`（依赖 `rank_bm25`） | 纯本地 | `test_layer_05_bm25.py` |
| L6 Chroma recall | Top-K 是否命中「应召回」chunk（正文子串 + metadata） | 需 Chroma + 本地嵌入；默认 skip | `golden_retrieval_chunk_ids.json`、`retrieval_recall_eval.py`、`test_layer_06_*.py` |
| L7 检索评判实调用 | `judge_retrieval_context` | 需 LLM | 见下文 |
| L8 LangGraph 全链路 / Agent | `execute_chat_with_langgraph`、`RAGAgent.*` | 需 LLM + 可选向量库 | 见下文 |

## L6：`golden_retrieval_chunk_ids.json`

根字段：

- `version`：整数
- `chroma_persist_dir`：可选，覆盖 Chroma 目录（也可用环境变量 `CHROMA_EVAL_PERSIST_DIR`）
- `cases`：用例列表；**为空**则集成测试 skip

每条 `case`：

| 字段 | 含义 |
|------|------|
| `id` | 唯一标识 |
| `enabled` | 默认 `true`；`false` 时跳过该条 |
| `namespace` | 与 `NamespaceVectorStore` 的 namespace 一致 |
| `queries` | 检索 query 列表（多 query 会合并排序） |
| `k` | Top-K |
| `strategy` | 如 `vector`（非 hybrid）、`hybrid`、`hybrid_rerank`（与 `retrieve` 一致） |
| `score_threshold` | 距离过滤上界；评测可先设 `10.0` |
| `chroma_filter` | 可选，等价 `extra_chroma_filter` |
| `hits` | 「应命中 **hits 条中至少满足几条**」由 `min_hit_ratio` 控制 |
| `min_hit_ratio` | 默认 `1.0`，即全部 `hits` 都需在 Top-K 中找到对应 chunk |

每条 `hit`：Top-K 中存在**至少一个** chunk 同时满足——

- `match_any_text`：列表中**任一**子串出现在 `page_content`（大小写不敏感）
- `metadata_contains`：metadata 必须包含这些键值（数值与字符串宽松相等）

CLI（项目根、已配置嵌入）：

```bash
export RUN_LAYER_INTEGRATION=1
python -m eval.retrieval_recall_eval
python -m eval.retrieval_recall_eval --fixture eval/fixtures/golden_retrieval_chunk_ids.example.json
```

模板：`eval/fixtures/golden_retrieval_chunk_ids.example.json`（默认 `enabled: false`）。

## L6+（本仓库未默认开启的层）

这些层建议在**有数据副本**的机器上跑，或用 `pytest -m integration` + 环境变量门禁。

1. **重排**：对同一候选列表比较有/无 rerank 的名次变化（需下载 cross-encoder 模型）。
2. **`judge_retrieval_context`**：对固定 `(question, grouped)` 断言 `sufficient` / `score` 落在区间或做 mock LLM。
3. **端到端**：`RAGAgent.answer`，用 `keyword_recall` 或人工金标。

示例（仅当你准备好时再执行）：

```bash
export RUN_LAYER_INTEGRATION=1
pytest tests/layers -m integration -v
```

未设置 `RUN_LAYER_INTEGRATION=1` 时，集成层测试会 **skip**，避免 CI 因缺 key/库失败。

## 如何扩展黄金数据

- **L1 / L2**：在对应 `eval/fixtures/golden_*.json` 增加条目，无需大模型。
- **L6 检索 gold**：推荐 **人工** 或 **脚本从已入库 PDF 定位页码再映射 chunk_id**；若用大模型生成 query，**标准答案 / chunk 标注仍应对齐原文或库元数据**，避免幻觉 gold。
