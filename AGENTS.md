# AGENTS.md

本文件是这个仓库给后续 agent 的工程备忘录。目标不是重复 README，而是记录"这个项目现在实际上是怎么构建的、哪些默认值已经锁定、继续改时哪些地方最容易踩坑"。

## 1. 项目目标

这是一个本地可复用的 `agentic literature RAG` 系统，面向 Markdown 格式的科学文献语料。

系统分成两条主链路：

1. 离线索引链路（`agentic_rag.builder`）
   - 读取 `.md` Markdown 文件
   - 提取 `# ` 标题作为文献标题，并识别正文 section heading 作为 `section_hint`
   - 先清洗 front matter / references / 图片本体，再按 section 驱动做语义切块（默认最大长度 2200，轻量 overlap 100）
   - 调用 DashScope `text-embedding-v4` 生成 2048-dim 向量
   - 建本地混合索引（SQLite + FTS5 + 向量）
2. 在线问答链路（`agentic_rag.query`）
   - 用户提问
   - agent 分析问题和检索策略
   - **所有 chunk 参与 hybrid 检索**（不再截断）
   - 按文献聚合：每篇文献取 score_final 最高的 3 个 chunks 算平均分，排序取前 300 篇
   - 对这 300 篇做 qwen3-rerank，再取前 100 篇
   - 并发执行文献级 judge（每篇固定传 top-3 chunks）
   - judge 对 direct_support 文献直接生成基于 chunks 的 2-3 句文献级 answer line
   - 最终按排序顺序做确定性拼接并生成带引用回答

这个项目的目标不是"单次向量检索后直接生成"，而是"agent 参与检索规划和文献级证据判断"。

## 2. 运行环境

- conda 环境固定为：`/home/fanny/miniconda3/envs/literature_rag`
- 推荐 Python：`/home/fanny/miniconda3/envs/literature_rag/bin/python`
- 推荐 pip：`/home/fanny/miniconda3/envs/literature_rag/bin/pip`

安装：

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/pip install -e .
```

不要把 API key 写进代码或仓库。统一用环境变量（`.env` 文件，已 gitignore）。

## 3. 当前锁定默认值

这些值已经在代码里作为默认行为实现，继续迭代时不要无意改掉：

- `EMBEDDING_MODEL=text-embedding-v4`
- `EMBEDDING_DIMENSIONS=2048`
- `DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=deepseek-chat`
- `RERANK_MODEL=qwen3-rerank`
- `RERANK_DOCUMENT_CHUNK_LIMIT=3`
- `RERANK_DOCUMENT_TEXT_LIMIT=0`（默认不截断；正整数才启用兼容性截断）
- `RERANK_REQUEST_TOKEN_BUDGET=3600`
- `CHUNK_SIZE=2200`
- `CHUNK_OVERLAP=100`
- `TOP_K_VECTOR=30`
- `TOP_K_KEYWORD=30`
- `DOCUMENT_RECALL_LIMIT=300`
- `DOCUMENT_JUDGE_LIMIT=100`
- `DOCUMENT_JUDGE_INITIAL_CONCURRENCY=5`
- `EARLY_STOP_CONSECUTIVE_NON_SUPPORT=20`
- `RRF_K=60.0`
- `HYBRID_CO_OCCURRENCE_BONUS=0.15`

注意：agent 模式下 hybrid_search 的 `limit` 已经固定为 `None`（不截断），但公开 API `search()` 仍然支持 `top_k` 截断。

## 4. 数据输入和格式假设

当前输入只有 **纯 Markdown（`.md`）文件**，没有 JSON 解析环节。

格式约定：

- `# Paper Title` — 文献标题
- `## Abstract` / `## Introduction` 等 — 章节标题，会提取为 `section_hint`
- 正文段落、摘要段落、figure caption — 语义 chunk 的主要来源

当前 chunker 会做面向英文期刊论文的规则清洗：

- 删除：作者、单位、邮箱、`ARTICLE INFO`、`Keywords`、收到/接受/在线发表信息、图片 Markdown 本体、`References` 及其后全部内容
- 保留：标题、摘要、正文 section heading、正文段落、`Statement of significance` / `Highlights`、figure caption
- 若没有显式 `Abstract` 标题，则把标题后到首个正式 section 之前的首个实质性长段落视为摘要

## 5. 索引结构

SQLite 仍然是检索时唯一会被程序读取的真相源；另外会额外导出一份仅供人工检查的 chunk artifact 到 `.rag_store/chunks/`。

三张核心表：

- `documents` — 文献级元数据（doc_id, source_path, title, file_hash, chunk_count, indexed_at）
- `chunks` — chunk 级元数据 + 向量 BLOB（chunk_id, doc_id, page_start, page_end, text, normalized_text, vector_dim, vector）
- `chunk_fts` — FTS5 虚拟表（chunk_id, title, text, keywords_hint, source_path）

增量构建依赖：

- 原文件 `file_hash`
- 如果 chunking / embedding dimensions 有变化，需要 `rebuild`
- `rebuild` 也会重建 `.rag_store/chunks/` 下的导出文件，避免残留旧 chunk artifact

额外导出文件：

- `.rag_store/chunks/<doc_id>.chunks.json` — 清洗后、切块后的结构化结果
- `.rag_store/chunks/<doc_id>.chunks.md` — 方便人工阅读的 chunk 预览

## 6. Chunking 规则

核心实现：`src/agentic_rag/builder/chunker.py`

当前规则：

- 首块固定是 `Title + Abstract`，紧随其后的 `Statement of significance` / `Highlights` 也会并入首块
- 每个正文 section 从 `section heading + 第一段正文` 开始成块；首段太短时会继续并入下一段
- figure caption 不单独成块，而是优先并入相邻正文块
- 超长段落才按句子切分，切分后仍保留所属 `section_hint`
- 默认使用 `CHUNK_SIZE=2200` 作为最大长度，并按比例推导出较小的目标长度和最小长度；`CHUNK_OVERLAP=100` 只用于句子切分后的轻量续接

chunk 里保留的重要元数据：

- `chunk_id`
- `doc_id`
- `title`
- `source_path`
- `page_start` / `page_end`
- `block_start` / `block_end`
- `section_hint`
- `keywords_hint`
- `normalized_text`

## 7. 检索与 agent loop

### 7.1 为什么 agent 模式下不截断

之前 `hybrid_search` 截断到 `top_k_vector + top_k_keyword`（约 600 个 chunks），导致 150 篇文献只召回 80-90 篇。现在 agent 模式下 `limit=None`，所有 chunk 都参与 RRF 融合和排序。

### 7.2 文献级聚合逻辑

核心实现：`src/agentic_rag/query/agent.py` 的 `_retrieve_document_candidates`

1. 每个 query bundle 做 hybrid_search，返回**所有**命中 chunks
2. 用 `rerank_hits_for_query_plan` 做约束感知的 re-scoring
3. 按文献（doc_id）分组，每篇文献内部取 score_final 最高的 **3 个 chunks**
4. 文献排序分 = 这 3 个 chunks 的 score_final **平均分**
5. 按平均分排序，取前 `DOCUMENT_RECALL_LIMIT=300` 篇
6. 对前 300 篇做 qwen3-rerank（用 top-3 chunks 拼接成 document input，并按 `RERANK_REQUEST_TOKEN_BUDGET` 分批请求）
7. 按 rerank score 排序，取前 `DOCUMENT_JUDGE_LIMIT=100` 篇进入 LLM 判定

### 7.3 judge_document

- 每次 LLM 只看 1 篇文献
- 固定传入该文献的 top-3 chunks
- 如果判为 `direct_support`，同一次 judge 调用必须直接返回该文献的 `answer_line`
- `answer_line` 是基于已提供 chunks 的 2-3 句直接支持说明，不要求也不能编造 chunks 中没有的方法/证据来源
- 并发派发，但结果按文献排序顺序提交
- 如果已派发文献遇到 provider pressure / timeout，会降低并发后按同一 rank 重新派发，直到成功或遇到非临时错误

### 7.4 引用格式

citation 中的 chunk_id 已经缩短：

- 原始：`bacterial-s-layer-protein-inspired-multifunction-5b7a39b1:0026`
- 缩短后：`5b7a39b1:0026`

如果一篇文献有多个 supporting chunks，参考文献中会合并列出所有 chunk，例如：

```
Title (pp. 1-2, chunk 5b7a39b1:0026; pp. 5-6, chunk 5b7a39b1:0008)
```

### 7.5 停止条件

- 连续 `20` 篇已提交文献都没有新增 `direct_support`
- 或者判定队列自然耗尽

注意：提前停止的定义是"停止派发新文献"，不是立即丢弃已在飞的请求。

## 8. 存储实现

v1 没引入外部向量数据库。当前设计是轻量本地化：

- SQLite 存文档与 chunk 元数据
- SQLite FTS5 做关键词索引
- 向量保存在 SQLite `chunks` 表的 BLOB 列中

`fetch_all_vectors()` 会一次性加载所有 vectors 到内存做 cosine similarity。当前 150 篇文献约 60MB，完全可接受。

## 9. 对外入口

稳定 API 入口在 `src/agentic_rag/query/__init__.py` 和 `src/agentic_rag/builder/__init__.py`：

- `build_index(source_dir=..., index_path=...)`
- `add_documents(paths=..., index_path=...)`
- `search(query=..., mode=..., top_k=...)`
- `await answer(question=...)`
- `async for event in answer_stream(question=...)` （流式事件）

CLI 在 `src/agentic_rag/cli.py`：

- `literature-rag build ./papers` — 从目录构建索引
- `literature-rag add ./new_papers` — 增量添加目录中的 `.md` 文件
- `literature-rag` — 交互式查询（提示输入问题，检索后输出回答 + 引用 + 扫描状态）

如果以后给别的 agent 复用，优先走 Python API，不要要求对方直接拼 SQLite 查询。

## 10. 关键命令

### 10.1 安装

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/pip install -e .
```

### 10.2 测试

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/python -m pytest -q
```

### 10.3 构建索引

```bash
export DASHSCOPE_API_KEY="..."
/home/fanny/miniconda3/envs/literature_rag/bin/python -m agentic_rag.cli build \
  ./papers \
  --index-path .rag_store/literature_rag.sqlite3
```

### 10.4 增量添加

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/python -m agentic_rag.cli add \
  ./new_papers \
  --index-path .rag_store/literature_rag.sqlite3
```

### 10.5 交互式问答

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/python -m agentic_rag.cli \
  --index-path .rag_store/literature_rag.sqlite3
```

进入交互模式后，会提示 `请输入您的问题（直接回车退出）`，输入问题后执行完整 agentic 检索，输出：
1. 编号回答（每篇文献一条）
2. 参考文献列表（`[N] citation`，多 chunk 会合并）
3. 扫描状态统计

## 11. 后续修改守则

如果你是后续 agent，修改这个项目时优先遵守下面这些约束：

1. 不要把 API key 写进任何源码、测试、README。
2. 输入格式只有 `.md`，不要引入 JSON 解析。
3. SQLite 是检索和问答使用的唯一真相源；`.rag_store/chunks/` 里的导出文件只用于人工检查，不参与查询。
4. 改 chunking 后，必须验证 citation 仍然能定位页码和 chunk 范围。
5. 改 embedding 参数后，必须同步检查 `core/config.py`、`builder/embedder.py`、索引配置是否需要 rebuild。
6. 改检索逻辑后，必须跑 `pytest -q` 和手动验证召回文献数是否接近库中文献总数。
7. 改 agent loop 后，不要把系统退化回一次检索一次生成。
8. 改 citation 格式后，同步更新 `tests/test_query_agent.py` 里的断言。

## 12. 如果下一个 agent 需要快速恢复上下文

最少先读这些文件：

1. `README.md`
2. `AGENTS.md`
3. `src/agentic_rag/builder/store.py`
4. `src/agentic_rag/query/__init__.py`
5. `src/agentic_rag/query/agent.py`

如果要改检索或索引，再补读：

6. `src/agentic_rag/query/retriever.py`
7. `src/agentic_rag/query/reranker.py`
8. `src/agentic_rag/builder/embedder.py`

如果要改 chunking，再补读：

9. `src/agentic_rag/builder/chunker.py`
10. `tests/test_builder_chunker.py`
11. `tests/test_builder.py`

## 13. 一句总原则

这个项目的核心不是"把文献喂进 embedding"，而是：

让 agent 用**全量 chunk 混合检索**和**文献级证据判断**去取证回答，确保高召回、高可引用性。
