# 初会RAG助手 Code Wiki

## 1. 项目概述

初会RAG助手是一个基于检索增强生成（RAG）技术的初级会计考试智能辅导系统。该项目利用阿里云通义千问的大语言模型和向量嵌入能力，结合Chroma向量数据库，为用户提供专业的初级会计考试知识库问答服务。

### 核心功能
- 多模态知识库管理（文本、图片、视频）
- 智能文档解析与切分
- 混合检索（向量检索 + BM25关键词检索）
- 查询智能改写与扩展
- 基于RAG的精准问答

---

## 2. 项目架构

### 2.1 目录结构

```
初会RAG助手/
├── knowledge_base/          # 知识库文件目录
│   ├── books/              # 教材文件
│   ├── notes/              # 笔记文件
│   ├── exercises/          # 习题文件
│   └── images/             # 图片资源
├── rag/                    # RAG核心模块
│   ├── __init__.py
│   ├── chroma_manager.py   # Chroma数据库管理
│   ├── embedding.py        # 多模态Embedding
│   ├── qa_chain.py         # QA问答链
│   └── retriever.py        # 检索器
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── bm25_retriever.py   # BM25检索器
│   ├── document_parser.py  # 文档解析
│   ├── query_rewriter_v2.py # 查询改写
│   └── text_splitter.py    # 文本切分
├── venv312/                # Python虚拟环境
├── build_index.py          # 构建索引入口
├── query.py                # 查询入口
├── config.py               # 配置文件
├── requirements.txt        # 依赖包
└── CODE_WIKI.md            # 本文档
```

### 2.2 架构设计图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询层                             │
│                      query.py (入口)                         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      QA问答链层 (qa_chain.py)                 │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ SmartQueryRewriter│──►  查询改写与扩展                    │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      检索层 (retriever.py)                    │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │  Chroma检索      │  │  BM25关键词检索                   │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
│               ↓ 混合检索 (HybridRetriever) ↓                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 向量数据库层 (chroma_manager.py)              │
│                 Chroma PersistentClient                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                 Embedding层 (embedding.py)                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ 文本Embedding   │  │ 图片Embedding    │  │ 视频Embedding│ │
│  │ text-embedding  │  │multimodal-       │  │multimodal-  │  │
│  │     -v4         │  │embedding-v1      │  │embedding-v1 │  │
│  └──────────────────┘  └──────────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      索引构建层                              │
│                   build_index.py (入口)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐  │
│  │ DocumentParser   │──► ChuHuiTextSplitter│──► ChromaManager│ │
│  │ 文档解析         │  │ 智能文本切分      │  │ 存入向量库  │  │
│  └──────────────────┘  └──────────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块说明

### 3.1 config.py - 配置模块

全局配置文件，管理所有项目参数。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| DASHSCOPE_API_KEY | 阿里云API密钥 | 环境变量 |
| CHROMA_DB_DIR | Chroma数据库目录 | ./chroma_db |
| TEXT_EMBEDDING_MODEL | 文本嵌入模型 | text-embedding-v4 |
| MULTIMODAL_EMBEDDING_MODEL | 多模态嵌入模型 | multimodal-embedding-v1 |
| LLM_MODEL | 大语言模型 | qwen-flash |
| CHUNK_SIZE | 默认文本块大小 | 600 |
| CHUNK_OVERLAP | 文本块重叠大小 | 100 |
| BM25_WEIGHT | BM25检索权重 | 0.15 |
| VECTOR_WEIGHT | 向量检索权重 | 0.85 |
| RETRIEVAL_K | 检索返回结果数 | 5 |

**知识类型分类：**
- 概念定义
- 会计分录
- 税法法条
- 计算公式
- 易错辨析
- 真题习题

**教材类型分类：**
- 初级会计实务
- 经济法基础

---

### 3.2 rag/chroma_manager.py - Chroma数据库管理

**类名：** `ChromaManager`

管理Chroma向量数据库的CRUD操作。

| 方法 | 说明 | 参数 |
|------|------|------|
| `__init__` | 初始化，创建或获取集合 | collection_name: str |
| `add_documents` | 添加文档到数据库 | documents, metadatas, ids |
| `add_image` | 添加图片到数据库 | image_path, metadata |
| `add_video` | 添加视频到数据库 | video_url, description, metadata |
| `query` | 查询向量数据库 | query_text, n_results, where |
| `get_collection_stats` | 获取集合统计信息 | - |
| `delete_by_ids` | 根据ID删除文档 | ids: list |
| `get_all_documents` | 获取所有文档 | - |
| `clear_collection` | 清空集合 | - |

**代码示例：**
```python
manager = ChromaManager()
manager.add_documents(["文档内容"], [{"source": "官方教材"}], ["doc_1"])
results = manager.query("什么是资产?", n_results=3)
```

---

### 3.3 rag/embedding.py - 多模态Embedding

**类名：** `MultiModalEmbedding`

封装阿里云通义千问的Embedding API，支持文本、图片、视频三种模态。

| 方法 | 说明 |
|------|------|
| `get_text_embedding` | 获取文本Embedding向量 |
| `get_image_embedding` | 获取图片Embedding向量 |
| `get_video_embedding` | 获取视频Embedding向量（多帧取平均） |
| `get_embedding` | 统一的Embedding接口 |

**关键代码：**
```python
# 文本Embedding使用 text-embedding-v4
resp = dashscope.TextEmbedding.call(
    model=TEXT_EMBEDDING_MODEL,
    input=text
)

# 图片Embedding使用 multimodal-embedding-v1
resp = dashscope.MultiModalEmbedding.call(
    model=MULTIMODAL_EMBEDDING_MODEL,
    input=[{'image': image_data}]
)
```

---

### 3.4 rag/retriever.py - 检索器

**类名：** `Retriever`

混合检索模块，结合向量检索和BM25关键词检索。

| 方法 | 说明 |
|------|------|
| `search` | 基础检索（支持过滤条件） |
| `search_with_filter` | 带自定义过滤条件的检索 |
| `search_all` | 不带过滤条件的检索 |
| `update_bm25_index` | 更新BM25索引 |

**检索策略：**
1. 自动检测查询的知识类型和教材类型
2. 支持严格过滤（仅检索匹配类型的知识）
3. 混合检索（向量检索 + BM25检索，权重可配置）

---

### 3.5 rag/qa_chain.py - QA问答链

**类名：** `QAChain`

完整的RAG问答流程实现。

| 方法 | 说明 |
|------|------|
| `ask` | 执行RAG问答（支持查询改写） |
| `_single_ask` | 执行单次问答 |
| `batch_ask` | 批量问答 |

**问答流程：**
1. 接收用户查询
2. 使用`SmartQueryRewriter`改写查询
3. 调用`Retriever`检索相关文档
4. 构建Prompt，包含检索到的上下文
5. 调用LLM生成回答
6. 返回回答及参考来源

---

### 3.6 utils/document_parser.py - 文档解析

**类名：** `DocumentParser`

支持PDF、DOCX、TXT三种文档格式的解析。

| 方法 | 说明 |
|------|------|
| `parse_pdf` | 解析PDF文件（带页码信息） |
| `parse_docx` | 解析DOCX文件（支持表格） |
| `parse_txt` | 解析TXT文件 |
| `parse_file` | 根据文件类型自动选择解析方法 |

---

### 3.7 utils/text_splitter.py - 智能文本切分

**类名：** `ChuHuiTextSplitter`

初级会计考试专属的文本切分器。

| 方法 | 说明 |
|------|------|
| `split_text` | 按初会规则切分文本 |
| `split_exercise` | 切分真题/习题 |
| `split_entry` | 切分会计分录 |
| `split_table` | 切分表格内容 |
| `split_by_type` | 根据知识类型选择切分策略 |
| `intelligent_split` | 智能切分（自动检测类型） |

**切分规则（CHUNK_RULES）：**
| 知识类型 | chunk_size | chunk_overlap |
|----------|------------|---------------|
| 概念定义 | 300 | 60 |
| 会计分录 | 200 | 40 |
| 税法法条 | 350 | 70 |
| 计算公式 | 250 | 50 |
| 易错辨析 | 350 | 70 |
| 真题习题 | 500 | 100 |
| 默认 | 350 | 70 |

---

### 3.8 utils/bm25_retriever.py - BM25检索器

**类名：** `BM25Retriever`, `HybridRetriever`

基于jieba分词和BM25算法的关键词检索。

| 方法 | 说明 |
|------|------|
| `BM25Retriever.build_index` | 构建BM25索引 |
| `BM25Retriever.search` | 执行BM25检索 |
| `HybridRetriever.hybrid_search` | 执行混合检索 |

**混合检索算法：**
1. 分别获取向量检索结果和BM25检索结果
2. 对两种结果的分数进行归一化
3. 按配置权重加权融合
4. 返回融合后的Top-K结果

---

### 3.9 utils/query_rewriter_v2.py - 查询改写

**类名：** `SimpleQueryRewriter`, `SmartQueryRewriter`

智能查询改写模块，提高检索匹配度。

| 方法 | 说明 |
|------|------|
| `SimpleQueryRewriter.rewrite` | 简单术语映射改写 |
| `SmartQueryRewriter.rewrite_for_retrieval` | 基于LLM的智能改写 |
| `SmartQueryRewriter.expand_query` | 查询扩展 |
| `SmartQueryRewriter.auto_rewrite` | 自动改写入口 |

**术语映射表示例：**
| 用户表达 | 知识库标准表达 |
|----------|----------------|
| 怎么做账 | 账务处理 |
| 进项税 | 进项税额 |
| 不能抵扣 | 不得抵扣 |
| 怎么算 | 计算公式 计算方法 |

---

### 3.10 build_index.py - 索引构建入口

**主函数：** `build_and_save()`

构建知识库索引的主流程：

1. 初始化`ChromaManager`和`ChuHuiTextSplitter`
2. 遍历`knowledge_base`目录下的`books/`、`notes/`、`exercises/`
3. 对每个文档文件：
   - 使用`DocumentParser`解析
   - 使用`infer_metadata`推断元数据（类型、章节、来源等）
   - 使用`ChuHuiTextSplitter`切分
   - 存入Chroma数据库
4. 处理`images/`目录下的图片
5. 输出构建统计信息

**元数据字段：**
- `book_type`: 教材类型（初级会计实务/经济法基础）
- `chapter`: 章节
- `knowledge_type`: 知识类型
- `exam_level`: 考试级别（必考/高频/了解）
- `source`: 来源
- `file_name`: 文件名
- `type`: 内容类型（text/image/video）
- `chunk_index`: chunk索引

---

### 3.11 query.py - 查询入口

**主函数：** `main()`

交互式查询入口：

1. 初始化`QAChain`
2. 循环接收用户输入
3. 调用`qa_chain.ask()`执行问答
4. 输出结果（原始问题、改写后问题、回答、参考来源）

---

## 4. 依赖关系

### 4.1 requirements.txt

```
chromadb>=0.4.0          # 向量数据库
dashscope>=1.14.0        # 阿里云通义千问SDK
pypdf2>=3.0.0            # PDF解析
python-docx>=0.8.11      # DOCX解析
langchain>=0.1.0         # LLM应用框架
numpy>=1.26.0            # 数值计算
pandas>=2.1.0            # 数据处理
Pillow>=10.0.0           # 图像处理
```

### 4.2 模块依赖图

```
build_index.py
├── config.py
├── rag/chroma_manager.py
│   ├── rag/embedding.py
│   └── config.py
└── utils/text_splitter.py
    ├── utils/document_parser.py
    └── config.py

query.py
├── config.py
└── rag/qa_chain.py
    ├── rag/retriever.py
    │   ├── rag/chroma_manager.py
    │   ├── utils/bm25_retriever.py
    │   └── config.py
    └── utils/query_rewriter_v2.py
        └── config.py
```

---

## 5. 运行指南

### 5.1 环境准备

1. 安装Python 3.12+
2. 设置环境变量：
   ```bash
   export DASHSCOPE_API_KEY=your_api_key_here
   ```
   (Windows: `set DASHSCOPE_API_KEY=your_api_key_here`)
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 5.2 构建知识库

1. 将文档放入 `knowledge_base/` 对应的子目录：
   - `books/`: 教材文件
   - `notes/`: 笔记文件
   - `exercises/`: 习题文件
   - `images/`: 图片资源
2. 运行构建脚本：
   ```bash
   python build_index.py
   ```

### 5.3 运行查询

```bash
python query.py
```

然后按提示输入问题即可。

---

## 6. 关键数据结构

### 6.1 文档元数据

```python
{
    "book_type": "初级会计实务",
    "chapter": "第一章 会计概述",
    "section": "",
    "sub_topic": "",
    "knowledge_type": "概念定义",
    "exam_level": "了解",
    "difficulty": "简单",
    "source": "2025官方教材",
    "file_name": "初级会计实务教材.pdf",
    "type": "text",
    "page_number": 1,
    "chunk_index": 0,
    "total_chunks": 10,
    "content_length": 250
}
```

### 6.2 检索结果格式

```python
{
    "content": "文档内容...",
    "metadata": {...},  # 上述元数据
    "distance": 0.35,   # 余弦距离（越小越相似）
    "id": "doc_0"
}
```

### 6.3 QA回答格式

```python
{
    "query": "用户问题",
    "answer": "LLM生成的回答",
    "sources": [
        {
            "source": "2025官方教材",
            "chapter": "第一章",
            "knowledge_type": "概念定义",
            "similarity": "65.00%"
        }
    ],
    "context": [...]  # 完整的检索结果列表
}
```

---

## 7. 扩展开发指南

### 7.1 添加新的文档格式

在 `utils/document_parser.py` 中的 `DocumentParser` 类添加新的解析方法：

```python
@staticmethod
def parse_md(md_path: str) -> str:
    """解析Markdown文件"""
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()

# 在 parse_file 中添加分支
elif ext == '.md':
    return DocumentParser.parse_md(md_path), {}
```

### 7.2 添加新的知识类型

1. 在 `config.py` 中的 `KNOWLEDGE_TYPES` 列表添加
2. 在 `CHUNK_RULES` 中添加对应的切分规则
3. 在 `infer_metadata` 中添加识别逻辑
4. 在 `_detect_knowledge_type` 中添加检测逻辑
5. 在 `ChuHuiTextSplitter` 中添加切分策略（如需要）

### 7.3 自定义检索权重

修改 `config.py` 中的 `BM25_WEIGHT` 和 `VECTOR_WEIGHT`：

```python
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7
```

---

## 8. 常见问题

### Q: 如何清空并重建知识库？

A: 删除 `chroma_db/` 目录，然后重新运行 `build_index.py`。

### Q: 如何调整检索结果数量？

A: 修改 `config.py` 中的 `RETRIEVAL_K` 参数，或在调用 `ask()` 时传入 `k` 参数。

### Q: 支持哪些文件格式？

A: 目前支持 PDF、DOCX、TXT 三种文本格式，以及 PNG、JPG、JPEG、GIF、BMP 等图片格式。

### Q: 如何提高检索准确率？

A: 可以尝试：
1. 调整混合检索权重
2. 添加更多相关文档到知识库
3. 调整文本切分参数
4. 使用查询扩展功能（`use_expansion=True`）

---

## 9. 技术栈总结

| 类别 | 技术选型 |
|------|----------|
| 编程语言 | Python 3.12 |
| 向量数据库 | ChromaDB |
| Embedding模型 | 阿里云 text-embedding-v4 / multimodal-embedding-v1 |
| LLM模型 | 阿里云 qwen-flash |
| 文档解析 | PyPDF2, python-docx |
| 关键词检索 | jieba + rank_bm25 |

---

## 10. 版本历史

- **v1.0.0**: 初始版本，支持基础RAG功能
  - 文本知识库管理
  - 向量检索
  - 智能问答

