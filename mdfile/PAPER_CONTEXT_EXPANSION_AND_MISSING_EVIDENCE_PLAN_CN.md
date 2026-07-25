# PaperSearchAssistant2 论文上下文扩展与缺失证据补全方案

## 1. 文档目标

这份文档用于指导 `PaperSearchAssistant2` 在论文问答场景下处理以下问题：

- 检索命中的 chunk 被截断
- 命中的上下文不完整
- 摘要只返回一部分
- 用户问的是方法、实验、结论，但对应 section 没被召回
- 当前 top-k 证据不足以回答问题

这份方案的核心目标是：

- 让系统在“证据不完整”时自动补足上下文
- 避免因为只命中一小段 chunk 而直接回答不完整或拒答
- 避免粗暴把整篇论文塞进上下文

---

## 2. 核心结论

对于论文问答，推荐的检索与上下文组织方式是：

- **先粗排，再精排**

更具体地说，就是：

1. 先用便宜、快速的方法做大范围召回
2. 再用更精细、更昂贵的方法做排序和补全
3. 命中后按需扩展 chunk、section、表格、图片

因此，不推荐一开始就把整篇论文所有块都读取出来。

更好的策略是：

- 小粒度召回
- 中粒度扩展
- 按需补全

---

## 3. 为什么不能直接把整篇论文所有块都读出来

如果直接把一篇论文所有块全部读取出来，会有这些问题：

## 3.1 噪音太大

论文通常包含：

- 摘要
- 引言
- 相关工作
- 方法
- 实验
- 结论
- 附录
- 表格
- 图注

如果用户只问“方法是什么”，把整篇都塞进去会引入大量无关信息。

## 3.2 token 开销过高

整篇论文 chunk 数量可能很多，会带来：

- prompt 过长
- 成本过高
- 模型注意力分散

## 3.3 回答反而更不稳

上下文过多时，模型可能：

- 抓不到重点
- 混淆不同 section
- 优先复述摘要而非真正回答问题

因此，不建议“全篇全文读入”作为默认方案。

---

## 4. 推荐的整体策略：先粗排，再精排，再扩展

推荐链路如下：

```text
用户问题
  -> 粗排召回
  -> 精排重排序
  -> 命中结果分析
  -> 缺失检测
  -> 邻近 chunk 扩展
  -> section 扩展
  -> 表格 / 图片补充
  -> 最终上下文打包
  -> 回答生成
```

这里的关键思想是：

- 召回时细粒度
- 回答时中粒度

---

## 5. 第一阶段：粗排召回

粗排阶段的目标是：

- 用较低成本找到“可能相关”的候选证据

推荐使用：

- 向量检索
- BM25 / FTS
- 表格 summary 检索
- 图片 summary 检索

粗排的特点：

- 召回范围大
- 成本较低
- 允许有噪音

推荐粗排召回数量：

- 正文 chunk：`20-50`
- 表格：`5-15`
- 图片：`5-15`

---

## 6. 第二阶段：精排重排序

精排阶段的目标是：

- 从粗排结果中找出真正最能回答问题的证据

推荐使用：

- cross-encoder reranker

例如：

- `BAAI/bge-reranker-v2-m3`

推荐精排保留数量：

- top `5-10`

精排后的结果不能直接回答，还需要做“证据完整性分析”。

---

## 7. 第三阶段：命中结果分析

在精排完成后，需要分析当前命中的证据是否足够。

应重点检查：

- 命中的 chunk 是否被截断
- 命中的 chunk 是否只有摘要片段
- 问题所需的 section 是否未出现
- 命中的内容是否过于泛化
- 是否缺失表格或图片证据

例如，出现以下情况都属于“证据缺失”：

- 摘要只返回一句话
- 命中的内容末尾明显被截断
- 用户问方法，但没有任何 `method / approach / model` 相关 section
- 用户问实验，但只返回了摘要和引言

---

## 8. 缺失证据补全策略

当系统检测到证据缺失时，必须自动补全。

## 8.1 邻近 chunk 扩展

这是第一层补全。

规则：

- 如果命中 chunk 被截断
- 或当前 chunk 只是局部片段

则自动补：

- 前一个 chunk
- 后一个 chunk

适合场景：

- 正文解释跨 chunk
- 一段话被切断
- 命中在段落中间

推荐扩展深度：

- 前 1 个
- 后 1 个

---

## 8.2 Section 扩展

这是第二层补全。

当用户问的是某类内容，但当前精排结果中没有该类 section 时，应自动补相应 section。

例如：

### 问方法

如果问题是：

- 方法是什么
- 怎么做的
- 模型结构如何

而当前命中里没有：

- method
- approach
- architecture
- model

相关 section，则自动读取这些 section 的 chunks。

### 问实验

如果问题是：

- 实验结果如何
- 在什么数据集上测试
- 表现比 baseline 好多少

而当前结果里没有：

- experiment
- evaluation
- result
- ablation

相关 section，则自动补对应 section。

### 问结论

如果问题是：

- 这篇论文最终结论是什么

则应优先补：

- conclusion
- discussion

相关 section。

---

## 8.3 摘要补全

如果当前命中的摘要明显是截断的，则不能直接用这一小段摘要回答。

正确做法是：

- 重新检索 abstract chunk
- 或补 paper summary chunk
- 或读取摘要 section 的完整 chunk 集合

摘要补全适合场景：

- 用户问“这篇论文讲了什么”
- 当前只命中摘要前半句

---

## 8.4 表格补全

当问题涉及：

- Table 2
- 表 3
- 某个指标
- 哪个结果最好

如果当前命中的正文不够，应自动补表格证据：

- table caption
- table summary
- 关键数值
- 必要时少量 markdown

不能只靠正文猜测表格内容。

---

## 8.5 图片 / 图表补全

当问题涉及：

- Figure 2
- 图 3
- 结构图
- 流程图

如果当前结果没有图像证据，则自动补：

- figure caption
- vision summary
- OCR 关键词
- 必要时补相关正文段落

---

## 9. 推荐的分层补全顺序

建议系统严格按照以下顺序补全，不要一上来就读整篇论文：

### 第 1 层：命中 chunk

先保留精排命中 chunk。

### 第 2 层：邻近 chunk

如果被截断或明显上下文不足，补前后 chunk。

### 第 3 层：命中 section

如果仍不足，补命中 chunk 所在 section 的其余关键 chunks。

### 第 4 层：目标 section

如果用户问“方法 / 实验 / 结论”等，而结果中没有对应 section，则定向补这些 section。

### 第 5 层：表格 / 图片证据

如果问题类型涉及表格或图片，则额外补这些证据。

### 第 6 层：paper summary

如果问题是总览型问题，则补 paper summary / abstract / conclusion。

只有在极少数情况下，才考虑更大范围跨 section 扩展。

---

## 10. Parent-Child Retrieval 方案

这是非常推荐采用的方案。

## 10.1 核心思想

检索时使用较小的 child chunk，提高召回精度；
回答时使用较大的 parent context，提高可读性和完整性。

例如：

- child：一个段落级 chunk
- parent：该 chunk 所在 section 的压缩上下文

## 10.2 为什么适合论文

论文问题通常既需要：

- 精确定位一句话或一个段落

又需要：

- 看到它所属的上下文

Parent-child retrieval 正好兼顾这两点。

## 10.3 推荐实现

在数据库中记录：

- `chunk_id`
- `section_id`
- `prev_chunk_id`
- `next_chunk_id`

在检索命中后：

- 先定位 child chunk
- 再取 section summary 或 section 内邻近 chunk 作为 parent context

---

## 11. Section 级 fallback 方案

如果精排后的 top-k 命中虽然相关，但不足以回答问题，则可以触发 section 级 fallback。

推荐规则：

- 用户问“方法”，但命中没有 method section -> 直接补 method section
- 用户问“实验”，但命中没有 result/evaluation section -> 直接补实验 section
- 用户问“结论”，但命中没有 conclusion -> 直接补结论 section

Section fallback 比“整篇 fallback”更高效，也更稳定。

---

## 12. 多通道补全策略

用户问论文内容时，补全不应只针对正文。

应当按问题类型决定补全通道。

## 12.1 正文问题

优先补：

- 邻近 chunk
- 相关 section

## 12.2 表格问题

优先补：

- table summary
- table caption
- key metrics

## 12.3 图片问题

优先补：

- figure summary
- figure caption
- OCR text
- nearby explanation paragraph

## 12.4 综合问题

例如：

- 这篇论文的方法和实验结果说明了什么

应补：

- method section
- experiment section
- 若有关键表格，再补表格 summary

---

## 13. 最终上下文打包策略

补全后，不能把所有内容不加控制地塞给模型。

建议做上下文打包：

- 去重
- 保留高优先级证据
- 控制 token
- 避免多个 section 重复说同一件事

推荐优先级：

1. 精排命中 chunk
2. 邻近 chunk
3. 关键 section chunk
4. 表格 / 图片 summary
5. 摘要 / 结论 summary

---

## 14. 推荐的默认行为

## 14.1 用户问“这篇论文讲了什么”

默认流程：

1. 粗排召回摘要、summary、引言、结论
2. 精排
3. 如果摘要截断，补完整摘要或 paper summary
4. 再补少量 conclusion

## 14.2 用户问“方法是什么”

默认流程：

1. 粗排召回 method 相关 chunk
2. 精排
3. 如果没有 method section，触发 method section fallback
4. 邻近扩展

## 14.3 用户问“实验结果如何”

默认流程：

1. 粗排召回 result / evaluation / ablation chunk
2. 精排
3. 如果命中不足，补实验 section
4. 如有关键表格，补 table summary

## 14.4 用户问“Table 3 说明了什么”

默认流程：

1. 粗排召回 table summary
2. BM25 命中 table number
3. 精排
4. 补 table caption 和关键数值

## 14.5 用户问“Figure 2 讲了什么”

默认流程：

1. 粗排召回 figure summary
2. BM25 命中 figure number
3. 精排
4. 补 caption、vision summary 和 OCR

---

## 15. 需要新增的模块

建议新增：

- `tools/retrieval/missing_evidence_detector.py`
- `tools/retrieval/context_expander.py`
- `tools/retrieval/section_fallback.py`

职责如下：

## 15.1 `missing_evidence_detector.py`

- 判断当前证据是否完整
- 检测摘要截断
- 检测 method / experiment / conclusion 缺失
- 检测表格 / 图片证据缺失

## 15.2 `context_expander.py`

- 根据命中 chunk 扩展前后 chunk
- 读取同一 section 的邻近内容
- 按 token 限制打包

## 15.3 `section_fallback.py`

- 根据子意图补 method / result / conclusion section

---

## 16. 推荐修改的现有模块

## 16.1 `paper_retriever.py`

增加：

- 粗排 -> 精排 -> 缺失分析 -> 补全

## 16.2 `paper_content_qa.py`

增加：

- 回答前的证据完整性检测
- 表格 / 图片补全逻辑

## 16.3 `context_assembler.py`

增加：

- 邻近 chunk 扩展
- section fallback 结果合并
- 优先级排序

---

## 17. 推荐给 Cursor 的实施顺序

建议 Cursor 按以下顺序实施：

1. 增加粗排与精排的分离
2. 增加 rerank 后的命中分析
3. 增加邻近 chunk 扩展
4. 增加 method / experiment / conclusion section fallback
5. 增加 table / figure 补全
6. 增加上下文打包策略

---

## 18. 最重要的结论

对于论文问答中的“上下文缺失”问题，最好的方案不是：

- 直接把整篇论文所有块读出来

而是：

- **先粗排召回**
- **再精排重排序**
- **命中后做按需扩展**
- **缺什么补什么**

也就是说，推荐的主链路是：

- **先粗排，再精排，再补全**

而不是：

- **一次性全篇全文加载**

这样才能在保证回答完整度的同时，控制噪音、成本和上下文长度。

