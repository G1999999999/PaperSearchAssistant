# PaperSearchAssistant2 本地论文问答优先方案

## 1. 文档目标

这份文档用于指导 `PaperSearchAssistant2` 修改“用户直接询问某篇论文内容”时的默认行为。

目标是确保：

- 在一个新对话中，用户直接输入某篇论文标题并提问时
- 系统优先在本地论文库中匹配这篇论文
- 如果本地命中，则直接进入本地论文问答流程
- 不要先返回论文候选列表
- 不要先走联网论文搜索
- 不要把“论文内容问答”误当成“论文搜索”

本方案主要解决以下类型请求：

- 说一说 Gaussian Grouping 这篇论文
- 说一说 Gaussian Grouping 这篇论文的方法部分
- 这篇论文讲了什么
- 它的实验结果如何

---

## 2. 当前问题

当前系统在遇到如下问题时：

- `说一说 Gaussian Grouping 这篇论文的方法部分`

可能会出现以下错误行为：

1. 把请求识别成 `paper_search`
2. 返回本地论文候选列表
3. 甚至触发联网论文补充
4. 没有直接进入“这篇本地论文的内容检索”
5. 甚至出现错误的 arXiv ID 绑定

这不符合论文阅读助手的预期。

正确行为应该是：

1. 识别为 `paper_qa`
2. 在本地论文库匹配目标论文
3. 本地命中后，直接绑定这篇论文
4. 在这篇论文内执行内容检索
5. 直接回答问题

---

## 3. 核心原则

## 3.1 “问某篇论文内容”不等于“找论文”

必须区分两类任务：

### 论文搜索

例如：

- 帮我找几篇 3DGS 的论文
- 推荐几篇 Gaussian Splatting 的论文

这是 `paper_search`。

### 论文内容问答

例如：

- 说一说 Gaussian Grouping 这篇论文
- Gaussian Grouping 的方法是什么
- 这篇论文实验部分讲了什么

这是 `paper_qa`。

`paper_qa` 的默认行为绝不能先返回论文列表。

## 3.2 本地命中后必须短路进入问答

当请求属于 `paper_qa`，且本地论文库高置信命中某篇论文时，必须：

- 直接绑定本地论文
- 直接进入内容检索
- 直接回答

不能再走：

- 论文列表展示
- 联网补充搜索
- 纯搜索结果返回

## 3.3 联网仅在本地缺失时才触发

如果：

- 用户问的是某篇论文内容
- 本地未命中该论文

则可以进入：

- 联网论文查找
- 下载
- 入库
- 再回答

也就是说，联网是 fallback，不是优先路径。

---

## 4. 正确的产品行为

对于：

- `说一说 Gaussian Grouping 这篇论文的方法部分`

正确行为应为：

1. 识别为 `paper_qa`
2. 抽取论文标题候选：`Gaussian Grouping`
3. 在本地论文库中匹配标题
4. 命中本地论文：
   - `Gaussian Grouping: Segment and Edit Anything in 3D Scenes`
   - `arXiv: 2312.00732`
5. 绑定这篇本地论文为当前论文
6. 识别问题子意图为 `method`
7. 在该论文范围内执行本地检索
8. 返回方法部分答案

而不是：

1. 列出本地候选论文
2. 再告诉用户“如果只想看本地结果可说只查本地”

---

## 5. 本地论文问答优先规则

建议明确写死以下规则。

## 5.1 规则一：识别 `paper_qa`

以下表达应优先识别为 `paper_qa`：

- 说一说某篇论文
- 这篇论文讲了什么
- 某篇论文的方法是什么
- 某篇论文的实验结果如何
- 帮我看看这篇论文

## 5.2 规则二：先做本地论文标题匹配

在 `paper_qa` 场景下，必须优先执行：

- 本地论文标题匹配
- 本地别名匹配
- 本地 arXiv ID 匹配

## 5.3 规则三：本地唯一高置信命中后直接短路

如果本地匹配结果满足：

- 唯一高置信命中

则直接：

- 绑定 `paper_id`
- 绑定 `arxiv_id`
- 绑定 `title`
- 进入本地论文问答

不再进入论文搜索列表流程。

## 5.4 规则四：本地未命中才进入联网 fallback

只有当：

- 本地未命中
- 或本地匹配非常不确定

才允许进入：

- 联网论文搜索
- 用户确认候选
- 下载与入库

## 5.5 规则五：当前论文状态必须持久化

当系统已经绑定某篇本地论文后，当前 session 需要保存：

- `current_paper_id`
- `current_arxiv_id`
- `current_title`
- `current_source = local`

后续用户继续问：

- 这篇论文的方法是什么
- 它的实验结果如何
- Table 3 说明了什么

默认应在这篇论文内继续检索。

---

## 6. 本地论文匹配算法优化

本地匹配算法必须加强，避免误判或漏判。

推荐匹配顺序如下：

## 6.1 标题精确匹配

优先匹配：

- 原标题
- 规范化标题

## 6.2 标题模糊匹配

支持：

- 大小写无关
- 连字符差异
- 空格差异
- 子串匹配

例如：

- `Gaussian Grouping`

可匹配：

- `Gaussian Grouping: Segment and Edit Anything in 3D Scenes`

## 6.3 trigram / 相似度匹配

如果用户输入是部分标题或略有误差，可用：

- trigram similarity
- 模糊匹配打分

## 6.4 arXiv ID 匹配

如果用户输入包含 arXiv ID，则直接匹配本地 arXiv 记录。

## 6.5 匹配结果置信度

建议给每个候选打分，并设置阈值：

- 高置信唯一命中 -> 直接进入本地问答
- 多个接近候选 -> 才列出候选让用户选

---

## 7. 本地论文问答检索流程

一旦本地论文命中并绑定成功，问答流程应如下：

## Step 1：识别子意图

识别用户问的是：

- `summary`
- `method`
- `experiment`
- `result`
- `conclusion`
- `table_lookup`
- `figure_lookup`

## Step 2：按 paper_id 限定检索范围

检索范围必须只限定在当前这篇本地论文内。

不能再全局乱搜其他论文。

## Step 3：section-aware 检索

根据子意图优先检索相关 section：

- method -> Method / Approach / Model
- experiment -> Experiment / Result / Evaluation
- conclusion -> Conclusion / Discussion

## Step 4：Hybrid RAG

在该论文内使用：

- 向量检索
- BM25 / FTS
- reranker

## Step 5：缺失补全

如果命中结果不完整，则：

- 补邻近 chunk
- 补对应 section
- 补 table / figure summary

## Step 6：生成回答

最终直接回答用户，而不是返回搜索列表。

---

## 8. 什么时候才应该先返回候选列表

只有以下情况才应该先返回候选列表，而不是直接问答：

## 8.1 本地有多个高相似候选

例如用户说：

- 看看 Gaussian 相关那篇论文

可能对应多篇本地论文。

此时可返回候选让用户选。

## 8.2 本地完全没有明确命中

此时可以：

- 联网搜索候选
- 再让用户选择

但这已经属于“本地问答失败后的 fallback”，不是默认主路径。

---

## 9. 不应出现的错误行为

## 9.1 错误行为一：本地命中后仍返回论文列表

用户已经明确问某篇论文内容时，本地又已命中唯一论文，不应再返回论文列表。

## 9.2 错误行为二：本地命中后仍优先联网

不应在本地已命中情况下，还先去联网补论文候选。

## 9.3 错误行为三：标题与 arXiv ID 错绑

例如：

- 问 Gaussian Grouping
- 却混入 `arXiv: 1706.03762`

这是严重错误。

必须保证：

- title
- `paper_id`
- `arxiv_id`

始终一致绑定。

## 9.4 错误行为四：没有复用当前论文状态

用户上一轮已经进入某篇论文阅读，下一轮追问时不能又回到全局搜索。

---

## 10. 推荐的结构化路由输出

对于：

- `说一说 Gaussian Grouping 这篇论文的方法部分`

推荐路由输出：

```json
{
  "intent": "paper_qa",
  "sub_intent": "method",
  "paper_match_mode": "local_title_match",
  "paper_ids": [123],
  "paper_titles": ["Gaussian Grouping: Segment and Edit Anything in 3D Scenes"],
  "source_preference": "local_first",
  "needs_web": false,
  "confidence": 0.97
}
```

关键点：

- `intent = paper_qa`
- 本地高置信命中
- `needs_web = false`

---

## 11. 推荐新增模块

建议新增：

- `tools/retrieval/local_paper_qa_resolver.py`
- `tools/retrieval/local_title_matcher.py`
- `tools/retrieval/session_current_paper.py`

职责如下：

## 11.1 `local_paper_qa_resolver.py`

- 接收 `paper_qa`
- 优先做本地论文匹配
- 命中后直接进入本地问答链路

## 11.2 `local_title_matcher.py`

- 做标题精确匹配
- 做标题模糊匹配
- 做 trigram 匹配

## 11.3 `session_current_paper.py`

- 维护当前会话绑定的本地论文
- 后续追问时直接复用

---

## 12. 推荐修改的现有模块

## 12.1 `router.py`

增加：

- `paper_search` 与 `paper_qa` 的明确区分

## 12.2 `agent.py`

增加：

- `paper_qa` 先走本地论文解析与绑定
- 本地命中后直接回答

## 12.3 `paper_search_service.py`

调整：

- 只有在本地论文问答未命中时，才作为 fallback 调用

## 12.4 `conversation.py`

支持保存：

- 当前 session 当前论文状态

---

## 13. 推荐给 Cursor 的实施顺序

建议 Cursor 按以下顺序修改：

1. 区分 `paper_search` 和 `paper_qa`
2. 增加本地标题匹配器
3. 增加“本地唯一命中后直接短路问答”规则
4. 增加当前 session 当前论文状态
5. 把论文问答流程限定到命中的 `paper_id`
6. 把联网搜索降级为本地未命中时的 fallback
7. 增加日志检查 title / paper_id / arxiv_id 一致性

---

## 14. 最重要的结论

对于新对话里直接问某篇本地论文内容的请求，系统必须做到：

1. 先识别为 `paper_qa`
2. 先匹配本地论文
3. 本地高置信命中后直接绑定该论文
4. 直接在该论文内做内容检索并回答

而不是：

1. 先返回论文候选列表
2. 再联网补搜索结果
3. 让用户重新确认

一句话总结：

- **本地论文内容问答必须优先于论文搜索列表流程**

