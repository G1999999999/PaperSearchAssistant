# PaperSearchAssistant2 高级路由与混合检索改造方案

## 1. 目标

这份文档用于指导 `PaperSearchAssistant2` 的“路由层 + 检索层”重构。

目标不是单纯替换关键词判断，而是建立一套更稳定的链路：

- 先更准确地判断用户到底在问什么
- 再更精确地确定应该去哪里检索
- 最后在正确范围内使用混合 RAG 方法找证据

本方案强调一个核心原则：

**高级路由不能替代向量检索、BM25 和 reranker；高级路由只是让这些方法用在更正确的范围里。**

---

## 2. 为什么要改

当前项目的论文意图识别主要依赖关键词规则，例如：

- `论文`
- `paper`
- `arxiv`
- `作者`
- `摘要`
- `download`
- `pdf`

这种方法虽然简单，但有明显问题：

- 用户直接说论文标题时，不一定包含这些词
- 用户直接问“gaussian-grouping 的方法是什么”时，可能不会被稳定识别为论文问题
- 问题明明是“某篇本地论文的内容”，系统却可能走到 arXiv 搜索或泛化知识库检索
- 规则命中后，后续检索范围也没有被很好地缩小到对应论文

因此，需要把路由从“关键词判断”升级为：

1. 规则短路
2. 论文实体识别
3. SQL 候选召回
4. LLM 语义路由
5. 结构化路由输出

---

## 3. 总体设计原则

## 3.1 路由层负责什么

路由层负责：

- 判断用户的真实意图
- 判断是否在问论文
- 判断是在找论文，还是问论文内容
- 判断是否应优先走本地论文库
- 判断是否应补充联网搜索
- 判断是否应重点查表格或图片

## 3.2 检索层负责什么

检索层负责：

- 在正确范围里做向量检索
- 在正确范围里做 BM25 / FTS 关键词检索
- 合并多路召回结果
- 用 reranker 重排
- 取邻近 chunk 扩展上下文

## 3.3 结论

也就是说：

- 路由解决“去哪里搜”
- 检索解决“怎么搜”
- 重排解决“谁最相关”

这三者都必须保留。

---

## 4. 路由层改造方案

## 4.1 新路由层的四级结构

建议把当前路由升级成四级决策：

### 第一级：规则短路

继续保留简单规则，处理极明显问题。

适合用规则直接判定的场景：

- 出现 arXiv ID
- 出现 `pdf`
- 出现 `download`
- 出现 `table 3` / `图 2`
- 出现天气类明显词

作用：

- 速度快
- 成本低
- 对强结构问题非常有效

### 第二级：论文实体识别

识别输入里是否包含：

- 论文标题
- 论文别名
- 作者名
- arXiv 风格论文名
- 已入库论文的标题近似匹配

例如：

- `gaussian-grouping 的方法是什么`
- `attention is all you need 的核心贡献`
- `dpr 这篇论文讲了什么`

这类问题不一定包含“论文”“paper”等关键词，但仍然是典型论文问题。

### 第三级：SQL 候选召回

对提取出来的疑似论文标题，在本地论文库里先做结构化候选召回。

建议召回方式：

- 精确标题匹配
- 规范化标题匹配
- trigram 相似度匹配
- 别名匹配
- arXiv ID 匹配

如果本地候选命中高置信，就优先走本地论文检索，而不是直接联网。

### 第四级：LLM 语义路由

对于规则和实体识别都不够确定的问题，交给一个轻量语义路由器判断。

语义路由器输出应当是结构化结果，而不是单个字符串标签。

---

## 5. 新的路由输出格式

当前路由如果只输出：

- `RAG`
- `ARXIV`
- `WEATHER`

信息量太少，不足以支撑后续精准检索。

建议新的路由输出结构如下：

```json
{
  "intent": "paper_qa",
  "sub_intent": "method",
  "source_preference": "local_first",
  "paper_match_mode": "local_title_match",
  "paper_ids": [123],
  "paper_titles": ["Gaussian Grouping for ..."],
  "section_hint": "method",
  "needs_table": false,
  "needs_figure": false,
  "needs_web": false,
  "confidence": 0.93
}
```

建议字段：

- `intent`
- `sub_intent`
- `source_preference`
- `paper_match_mode`
- `paper_ids`
- `paper_titles`
- `section_hint`
- `needs_table`
- `needs_figure`
- `needs_web`
- `confidence`

---

## 6. 意图分类设计

## 6.1 主意图

建议主意图至少包含：

- `paper_search`：找论文
- `paper_qa`：问论文内容
- `local_rag`：问本地知识库
- `web_search`：需要联网补充
- `tool_task`：明显工具调用
- `casual_chat`：普通对话

## 6.2 论文子意图

针对 `paper_qa` 进一步细分：

- `summary`
- `method`
- `experiment`
- `result`
- `conclusion`
- `comparison`
- `table_lookup`
- `figure_lookup`
- `metric_lookup`

子意图的作用是给后续混合检索做加权和过滤。

---

## 7. 本地论文优先策略

当用户问论文内容时，应该优先判断“这是不是本地已经有的论文”。

推荐顺序：

1. 是否提到了 arXiv ID
2. 是否提到了本地已有论文标题或标题近似
3. 是否提到了已知论文别名
4. 如果本地没有高置信候选，再考虑 arXiv / 联网

推荐策略：

- 如果本地匹配置信度高，优先只在本地论文范围内检索
- 如果本地匹配不确定，但联网能补充，可先列候选让用户确认
- 如果用户明确说“联网搜索”，再走联网论文搜索

---

## 8. 为什么高级路由之后仍然要保留 Hybrid RAG

这是本方案最重要的结论之一。

高级路由只能解决：

- 该搜哪篇论文
- 该搜哪一类内容

但它不能直接找到答案所在的具体段落、图注或表格。

因此在确定范围后，仍然必须保留：

- 向量检索
- BM25 / FTS
- reranker 重排

原因如下。

## 8.1 向量检索仍然必要

适合：

- 用户问题和论文原文措辞不同
- 概念性提问
- 开放式问法
- 总结性问法

例如：

- `这篇论文的方法创新点是什么`
- `它是怎么做实例分组的`

这类问题很难只靠关键词。

## 8.2 BM25 / FTS 仍然必要

适合：

- 精确术语
- 缩写
- 指标名
- 数据集名
- 表格编号
- 图编号

例如：

- `Table 3`
- `mIoU`
- `DINOv2`
- `ablation`

这类词，BM25 往往比纯向量更稳。

## 8.3 Reranker 仍然必要

向量检索和 BM25 通常只能做“粗召回”。

论文里会出现大量“看起来都相关”的 chunk：

- 方法介绍
- 实验设置
- 结果分析
- 相关工作

如果没有 reranker，真正回答用户问题的证据不一定排在前面。

因此建议：

- 路由层缩小范围
- 混合召回找候选
- reranker 决定最终顺序

---

## 9. 推荐的论文问答完整链路

当用户在对话中输入：

`gaussian-grouping 的方法是什么`

推荐链路如下：

1. 路由层识别为 `paper_qa`
2. 论文实体识别发现 `gaussian-grouping` 像论文标题
3. SQL 在本地论文表中召回候选
4. 如果唯一高置信命中某篇本地论文，则锁定 `paper_id`
5. 根据问题内容识别 `sub_intent = method`
6. SQL 预过滤该论文中与 `method / approach / model` 相关的 section/chunk
7. 在这些候选上并行做：
   - 向量检索
   - BM25 / FTS
   - 如有必要，表格/图片召回
8. 多路结果融合
9. reranker 重排
10. 取 top 结果并扩展相邻 chunk
11. 组织为最终上下文
12. 生成答案

这个流程里，路由和混合 RAG 是协同关系，不是二选一。

---

## 10. 论文找寻和论文问答的区别

系统必须区分两种场景。

## 10.1 找论文

例如：

- `帮我找一篇关于 gaussian grouping 的论文`
- `找一下关于 RAG evaluation 的论文`

这类问题的核心是：

- 找到候选论文列表
- 返回论文元信息

此时优先使用：

- 本地 SQL 论文库搜索
- arXiv 搜索
- 标题/摘要匹配

不一定马上进入全文 chunk 检索。

## 10.2 问论文内容

例如：

- `gaussian-grouping 讲了什么`
- `这篇论文的方法是什么`

这类问题的核心是：

- 找到答案证据
- 需要进入 chunk 级别的混合检索

因此必须进入：

- SQL 预过滤
- 向量检索
- BM25
- 重排

---

## 11. 路由与检索协同的推荐策略

建议新增这些策略层次。

## 11.1 路由层策略

- `rule_first`
- `entity_first`
- `llm_assist`
- `hybrid_router`

推荐默认：

- `hybrid_router`

含义：

1. 先规则短路
2. 再做论文实体识别
3. 再做 SQL 候选召回
4. 最后必要时用 LLM 路由补判

## 11.2 检索层策略

- `vector_only`
- `bm25_only`
- `hybrid`
- `hybrid_rerank`
- `multi_query_hybrid`

推荐默认：

- `hybrid_rerank`

对复杂论文问答推荐：

- `multi_query_hybrid`

---

## 12. 建议增加的高级 RAG 方法

除了原有的向量检索、BM25 和 rerank，建议逐步加上这些方法。

## 12.1 多查询改写

对一个问题生成多个检索 query。

例如：

- `gaussian-grouping 的方法是什么`

可以生成：

- `gaussian-grouping method`
- `gaussian-grouping approach`
- `gaussian-grouping model architecture`

然后分别做召回，再合并。

## 12.2 复合问题拆分

例如：

- `这篇论文的方法是什么，实验结果怎么样，和 DPR 有什么区别`

拆成三个子问题分别检索，再合并。

## 12.3 分层检索

先找最相关 section，再在 section 内检索 chunk。

特别适合长论文。

## 12.4 Parent-Child Retrieval

先用较小 chunk 做召回，再返回它所在的更大父上下文。

这能兼顾：

- 召回精度
- 阅读完整性

## 12.5 MMR 上下文打包

不要直接拿 top-k chunk 拼 prompt。

应该用 MMR 或去冗余策略，保证：

- 高相关
- 低重复
- 覆盖多个角度

## 12.6 元数据感知检索

在召回和排序时考虑：

- `chunk_role`
- `section_title`
- `has_table`
- `has_figure`
- `paper_id`

---

## 13. 建议的代码改造点

## 13.1 新增路由模块

建议新增：

- `tools/retrieval/query_router.py`
- `tools/retrieval/paper_entity_matcher.py`
- `tools/retrieval/query_understanding.py`

职责分别为：

### `query_router.py`

- 统一入口
- 聚合规则、实体识别、SQL 候选、LLM 路由
- 输出结构化路由结果

### `paper_entity_matcher.py`

- 根据用户输入匹配本地论文标题
- 支持规范化、模糊匹配、trigram 相似度

### `query_understanding.py`

- 识别子意图
- 识别 section hint
- 识别是否偏向表格或图片
- 生成多 query

## 13.2 重构 `tools/agent/router.py`

目标：

- 不再只输出简单枚举值
- 改为调用新的结构化路由模块
- 保留规则短路逻辑作为第一层

## 13.3 新增 `tools/retrieval/paper_retriever.py`

职责：

- 接收结构化路由结果
- 基于 `paper_id / section_hint / needs_table / needs_figure` 发起混合检索
- 完成融合、重排、上下文扩展

---

## 14. 推荐新增配置项

建议新增这些配置项：

- `RAG_ROUTER_MODE`
- `RAG_ROUTER_ENABLE_ENTITY_MATCH`
- `RAG_ROUTER_ENABLE_SQL_CANDIDATE_MATCH`
- `RAG_ROUTER_ENABLE_LLM_ASSIST`
- `RAG_ROUTER_LOCAL_PAPER_CONFIDENCE_THRESHOLD`
- `RAG_PAPER_FORCE_LOCAL_FIRST`
- `RAG_PAPER_ENABLE_MULTI_QUERY`
- `RAG_PAPER_ENABLE_QUERY_DECOMPOSITION`
- `RAG_PAPER_ENABLE_SECTION_PREFILTER`
- `RAG_PAPER_ENABLE_TABLE_RECALL`
- `RAG_PAPER_ENABLE_FIGURE_RECALL`
- `RAG_PAPER_ENABLE_RERANK`

---

## 15. 推荐给 Cursor 的实施顺序

Cursor 应按以下顺序实施：

1. 新增结构化路由结果对象
2. 新增论文标题实体匹配器
3. 用 SQL 做本地论文候选召回
4. 改造 `router.py`，接入新路由层
5. 新增 `paper_retriever.py`
6. 把混合检索逻辑接到新检索器里
7. 在 `agent.py` 中把“论文问答”和“论文搜索”区分开
8. 增加配置项和日志
9. 增加评测脚本

---

## 16. 最终默认行为建议

改造完成后，论文相关问题默认行为建议如下：

### 场景一：用户直接问论文标题内容

例如：

- `gaussian-grouping 的方法是什么`

默认行为：

1. 识别为论文内容问题
2. 先匹配本地论文标题
3. 如果命中本地论文，则优先在本地该论文内做混合检索
4. 不直接联网

### 场景二：用户说要找论文

例如：

- `帮我找一篇关于 gaussian grouping 的论文`

默认行为：

1. 识别为论文搜索问题
2. 先查本地论文库
3. 本地没有高置信候选时，再去 arXiv 或联网补充

### 场景三：用户明确要求联网

例如：

- `请联网找一下 gaussian grouping 论文`

默认行为：

1. 尊重用户要求
2. 执行联网论文搜索
3. 若本地也有高相关论文，可合并展示

---

## 17. 最重要的结论

这一轮改造的核心不是把“关键词路由”简单换成“LLM 路由”，而是建立完整链路：

1. 高级路由负责识别目标论文和问题类型
2. SQL 负责缩小检索范围
3. 向量检索负责语义召回
4. BM25 负责精确术语命中
5. reranker 负责最终排序
6. 上下文扩展负责提升回答完整度

因此，最终推荐方案不是：

- “高级路由 or Hybrid RAG”

而是：

- **高级路由 + SQL 预过滤 + 向量检索 + BM25 + reranker + 上下文扩展**

这才是适合论文问答场景的完整方案。

