# Agentic Literature RAG

`agentic-literature-rag` 是一个面向 Markdown 科学文献语料的本地 agentic RAG 系统。它把离线建库和在线问答拆成两个独立链路：

1. `agentic_rag.builder`：清洗 Markdown 文献、提取标题和章节、语义切块、调用 DashScope embedding，并把 SQLite + FTS5 + 向量索引写入本地。
2. `agentic_rag.query`：分析用户问题、执行混合召回、调用 rerank，并根据选择的模式输出文献级证据回答或深度研究报告。

这个项目不是“一次向量检索后直接生成答案”。默认文献检索路线会让 agent 先规划检索，再对所有 chunks 做 hybrid recall，按文献聚合、重排、逐篇 judge，最后只基于有直接证据的文献生成可引用回答。

## 功能概览

- 本地 Markdown 文献库构建：递归读取 `.md` 文件，生成本地 SQLite 向量库。
- 增量添加文献：已入库且文件 hash 未变的文献会跳过，预览缺失时会从 SQLite 补写 chunk 预览。
- 文献删除：可按论文标题或原始 Markdown 文件名从库里删除。
- 文献检索模式：召回、rerank、逐篇 judge，并输出每篇支持文献的回答行和参考文献。
- 深度研究模式：召回后做 chunk 级 rerank，取最多 100 个完整 chunks，让聊天模型生成带引用的研究报告。
- 可替换聊天模型：通用聊天模型使用兼容 OpenAI Chat Completions 的 API，通过 `API_KEY`、`BASE_URL`、`MODEL` 配置。

## 安装

推荐在项目根目录执行：

```bash
pip install -e .
```

如果使用当前开发环境，可以显式指定 conda 环境里的 pip：

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/pip install -e .
```

要求 Python `>=3.11`。

安装后会得到命令行入口：

```bash
literature-rag
```

## 配置

仓库中直接包含一个空 key 的 `.env` 模板。安装后打开 `.env`，填入自己的 key 即可。

注意：`.env` 虽然作为模板提交到了仓库，但你填入真实 API key 后，不要把这次修改提交到 GitHub。

### 必填 key

```bash
DASHSCOPE_API_KEY=
API_KEY=
```

`DASHSCOPE_API_KEY` 用于 embedding 和 rerank。`API_KEY` 用于兼容 OpenAI API 的聊天模型，也就是 agent 分析、文献 judge 和深度研究报告生成。

### Embedding 与 rerank

```bash
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_RERANK_BASE_URL=https://dashscope.aliyuncs.com/compatible-api/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=2048
RERANK_MODEL=qwen3-rerank
RERANK_REQUEST_TOKEN_BUDGET=3600
```

默认 embedding 模型是 `text-embedding-v4`，维度是 `2048`。默认 rerank 模型是 `qwen3-rerank`。程序会按 token budget 分批请求 rerank；rerank 失败不会静默回退成本地排序，而是直接报错。

### 通用聊天模型

```bash
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
TIMEOUT_SECONDS=300
MAX_RETRIES=4
```

聊天模型只要求服务兼容 OpenAI Chat Completions API，因此可以换成不同服务：

```bash
# DeepSeek
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat

# DashScope OpenAI 兼容模式
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen-plus

# OpenAI
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4.1-mini
```

### 切块与检索默认值

```bash
CHUNK_SIZE=2200
CHUNK_OVERLAP=100
TOP_K_VECTOR=30
TOP_K_KEYWORD=30
DOCUMENT_RECALL_LIMIT=300
DOCUMENT_JUDGE_LIMIT=100
DOCUMENT_JUDGE_INITIAL_CONCURRENCY=5
RERANK_DOCUMENT_CHUNK_LIMIT=3
RERANK_DOCUMENT_TEXT_LIMIT=0
RESEARCH_RERANK_CHUNK_LIMIT=500
RESEARCH_FINAL_CHUNK_LIMIT=100
RESEARCH_CONTEXT_TOKEN_BUDGET=120000
RESEARCH_REPORT_MIN_CHARS=500
```

说明：

- `RERANK_DOCUMENT_TEXT_LIMIT=0` 表示文献级 rerank 不截断 chunk 文本。
- 文献检索路线每篇文献固定取 top-3 chunks 进入 rerank 和 judge。
- 深度研究路线默认对前 500 个 chunks 做 rerank，最多把 100 个完整 chunks 放入最终生成上下文。
- `RESEARCH_REPORT_MIN_CHARS=500` 按 Python 字符数计算，中文一个汉字算 1 个字符；报告没有固定上限，由聊天模型根据 chunks 内容决定长度。

## CLI 用法

### 重建向量库

```bash
literature-rag build ./papers
```

`build` 会清空旧索引，并用指定目录下的 Markdown 文献重新建库。

### 增量添加文献

```bash
literature-rag add ./new_papers
```

`add` 会递归扫描目录或添加单个 Markdown 文件。成功入库的文献会同步写入 `.rag_store/chunks/` 下的 chunk 预览文件。

### 删除文献

```bash
literature-rag delete --title "论文标题或md文件名"
```

删除操作会从 SQLite 索引中移除对应文献及其 chunks。

### 交互式查询

```bash
literature-rag
```

启动后程序会先加载默认向量库：

```text
.rag_store/literature_rag.sqlite3
```

然后显示上下键菜单：

```text
文献检索
深度研究
```

选择 `文献检索` 后，输入问题，程序会输出：

1. 编号回答，每条通常对应一篇 direct_support 文献。
2. 参考文献列表，同一文献多个 supporting chunks 会合并显示。
3. 扫描状态、judge 数量、并发和耗时统计。

选择 `深度研究` 后，输入研究问题，程序会输出：

1. 基于 evidence chunks 生成的研究报告。
2. 报告中实际引用到的参考文献。
3. recalled / reranked / context chunk 数量和耗时统计。

### 指定索引路径

```bash
literature-rag --index-path .rag_store/literature_rag.sqlite3
```

`build`、`add`、`delete` 也都支持 `--index-path`。

## Python API

### 建库与增量添加

```python
from agentic_rag.builder import add_documents, build_index, delete_document

build_index(
    source_dir="literature_md",
    index_path=".rag_store/literature_rag.sqlite3",
)

add_documents(
    paths=["new_papers"],
    index_path=".rag_store/literature_rag.sqlite3",
)

delete_document(
    title="Paper Title",
    index_path=".rag_store/literature_rag.sqlite3",
)
```

### 搜索、文献检索和深度研究

```python
import asyncio

from agentic_rag.query import answer, answer_stream, research, search

hits = search(
    query="hydroxyapatite molecular dynamics DFT",
    mode="hybrid",
    top_k=5,
)

async def main():
    result = await answer(
        question="使用分子动力学模拟和DFT分别研究羟基磷灰石，两者的异同在哪？",
    )
    print(result.answer)
    print(result.citations)

    async for event in answer_stream(
        question="早期矿化发生在胶原纤维内的文献内容",
    ):
        print(event.event)

    report = await research(
        question="羟基磷灰石在医学领域有哪些应用？",
    )
    print(report.report)
    print(report.citations)

asyncio.run(main())
```

Python API 支持注入自定义后端：

- `embedder=`：实现 `embed_texts()` 和 `embed_query()`
- `reranker=`：实现 `rerank()`
- `llm=`：实现 `await complete_json()`

## 输入 Markdown 格式

本项目当前只直接读取 Markdown 文件，不直接解析 PDF。如果原始文献是 PDF，可以先使用 MinerU 的网页版或 App 将 PDF 解析、转换为 Markdown，再把生成的 `.md` 文件放入文献目录中执行 `literature-rag build` 或 `literature-rag add`。

每篇文献应该是一个 `.md` 文件，并且必须有真实论文标题 H1：

```markdown
# Paper Title

Author names...

# ABSTRACT

Abstract text...

# INTRODUCTION

Introduction text...

## 2. Experimental section

### 2.1. Sample preparation

Body text...
```

标题约束：

- 论文标题必须来自 Markdown H1，也就是 `# Paper Title`。
- 不使用文件名兜底生成论文标题。
- `Just Accepted`、`Accepted Article`、`Article`、`Reuse`、`Takedown`、期刊 masthead、推荐文章、广告 OCR 等 front matter 不会作为论文标题。
- 如果同一个真实论文标题 H1 重复出现，默认选择最后一次作为正文起点，并丢弃它之前的网页头部噪声。
- 如果存在多个不同的真实论文标题候选，程序会报错并跳过该文件，避免把多篇文章混入同一条文献。

## 清洗与切块规则

核心实现位于 `src/agentic_rag/builder/chunker.py`。

主要规则：

- 删除网页头部、期刊 banner、accepted manuscript 声明、作者单位、邮箱、`ARTICLE INFO`、`Keywords`、received/accepted/published metadata、图片 Markdown 本体、`References` 及其后的参考文献。
- 保留摘要、正文段落、`Statement of significance`、`Highlights` 和 figure caption。
- 表格行和表格本体会在清洗时去掉，保留 caption；目标是让 chunk 内容尽量是自然语言句子。
- 不允许为了处理超长输入而随意切断句子；切分应优先按句子边界进行。
- 支持层级标题路径：`##`、`###`、`####` 等父子标题会被保留到 chunk 文本和 `section_hint` 中。
- 如果短 section 合并到同一个 chunk，每个 section 都会重新写入自己的完整标题路径，保证 chunk 独立可读。
- 如果长 section 被拆成多个 chunk，后续 chunk 也会重复携带完整标题路径。

建库或增量添加成功后，程序会额外写出人工检查用预览：

```text
.rag_store/chunks/<doc_id>.chunks.json
.rag_store/chunks/<doc_id>.chunks.md
```

注意：`.rag_store/chunks/` 只是预览文件，不参与检索。真正的查询源是 SQLite。

## 检索路线

### 文献检索

默认文献检索链路：

1. 聊天模型分析问题，生成 query bundles 和约束。
2. 对所有 chunks 执行 hybrid recall，不在 agent 模式下截断召回结果。
3. 对每个 query bundle 的命中结果做约束感知重评分。
4. 按 `doc_id` 聚合，每篇文献取最高分 top-3 chunks。
5. 按 top-3 chunk 平均分取前 300 篇文献。
6. 用 `qwen3-rerank` 对 300 篇文献级候选重排。
7. 取前 100 篇进入文献级 judge。
8. 每篇 judge 只看该文献 top-3 chunks，判断是否 direct_support。
9. 支持文献生成 2-3 句文献级 answer line。
10. 最终按排序顺序拼接答案并生成引用。

提前停止规则：如果连续 20 篇已提交文献都没有新增 direct_support，则停止派发新文献；已经派发的请求会完成并计入最终结果。

### 深度研究

深度研究链路：

1. 复用 agent 的问题分析和 query bundles。
2. 全量 hybrid recall。
3. 按 chunk 去重并按本地分数取前 500 chunks。
4. 用 `qwen3-rerank` 做 chunk 级重排。
5. 取最多 100 个完整 chunks 进入上下文，受 `RESEARCH_CONTEXT_TOKEN_BUDGET` 限制。
6. 聊天模型只基于这些 chunks 生成研究报告。
7. 正文引用会被归一化到最终参考文献编号，避免出现没有参考文献对应的 `[82]` 这类悬空编号。

深度研究不做逐篇 judge，它更适合“综述类、比较类、研究方向类”的问题。

## 项目结构

```text
agentic_rag/
├── builder/
│   ├── chunker.py      # Markdown 清洗与语义切块
│   ├── embedder.py     # DashScope embedding
│   ├── artifacts.py    # chunk 预览导出
│   └── store.py        # SQLite + FTS5 + 向量存储
├── query/
│   ├── retriever.py    # hybrid search
│   ├── reranker.py     # DashScope qwen3-rerank
│   └── agent.py        # 问题分析、judge、深度研究生成、引用合并
├── core/
│   ├── config.py       # 环境变量和默认设置
│   ├── llm.py          # OpenAI-compatible chat client
│   ├── models.py       # Pydantic 数据模型
│   └── utils.py        # retry、JSON 提取、归一化等工具
└── cli.py              # Typer 命令行入口
```

## 本地数据与 Git

以下内容不应该提交到 GitHub：

- 填入真实 API key 之后的 `.env` 修改
- `.rag_store/`
- `literature_md/`
- `literature_md_*/`
- `.pytest_cache/`
- `__pycache__/`
- `dist/`
- `build/`

GitHub 仓库只应该保存源码、测试、README、空 key 的 `.env` 模板和项目元数据。文献语料和向量库应由用户在本地通过 `literature-rag build` 或 `literature-rag add` 生成。

如果你已经在 `.env` 里填了真实 key，可以执行下面的命令让 Git 暂时忽略本机对 `.env` 的修改：

```bash
git update-index --skip-worktree .env
```

如果以后确实要更新 `.env` 模板，再恢复跟踪：

```bash
git update-index --no-skip-worktree .env
```

## 测试

```bash
python -m pytest -q
```

当前开发环境也可以使用：

```bash
/home/fanny/miniconda3/envs/literature_rag/bin/python -m pytest -q
```

## 常见问题

### 为什么别人从 GitHub 下载后没有 `literature-rag` 命令？

`literature-rag` 是通过 `pyproject.toml` 的 console script 安装出来的，不应该直接提交 conda 环境里的可执行文件。别人 clone 后执行：

```bash
pip install -e .
```

就会在自己的 Python 环境中生成 `literature-rag` 命令。

### `.rag_store/chunks/` 为什么不等于真实数据库？

`.rag_store/chunks/` 是人工检查用的 chunk 预览。真实检索只读取 SQLite 数据库：

```text
.rag_store/literature_rag.sqlite3
```

### 深度研究为什么可能很慢？

深度研究会全量召回、对最多 500 个 chunks 做 rerank，并把最多 100 个完整 chunks 交给聊天模型生成报告。它没有逐篇 judge，但上下文更大，速度主要受 rerank 和聊天模型服务影响。

### 没有合法 H1 标题的 Markdown 会怎样？

程序会跳过该文件并在 `failed_files` 中报告原因。当前设计坚持“标题必须来自 Markdown H1”，不会用文件名或 `Title:` 行兜底。
