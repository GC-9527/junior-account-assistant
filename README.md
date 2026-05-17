
# 初会RAG助手

智能初级会计考试备考助手，基于RAG技术构建。

## 功能特性

- 📚 智能检索 - 基于Chroma的向量检索 + BM25混合检索
- 🎯 精准重排序 - 通义千问LLM Rerank提升相关性
- 🔍 智能改写 - 自动优化查询，支持复合问题识别
- 🌐 联网搜索 - Tavily搜索增强时效性内容
- 💡 术语保护 - 防止会计术语混淆（专项扣除vs专项附加扣除）

## 技术栈

- **向量化**: DashScope text-embedding-v4
- **LLM**: 通义千问 Qwen-Flash
- **向量库**: Chroma
- **检索方式**: 混合检索（向量+BM25）
- **重排序**: 通义千问 Rerank
- **联网搜索**: Tavily Search

## 快速开始

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置API Key
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
$env:TAVILY_API_KEY="your-tavily-api-key"  # 可选
```

### 构建索引

```bash
python build_index.py
```

### 启动查询

```bash
python query.py
```

## 目录结构

```
初会RAG助手/
├── knowledge_base/          # 知识库
│   ├── books/              # 教材和题库
│   └── notes/              # 笔记
├── rag/                    # RAG核心模块
│   ├── chroma_manager.py   # Chroma索引管理
│   ├── embedding.py        # 向量化
│   ├── qa_chain.py         # QA链
│   └── retriever.py        # 检索器
├── utils/                  # 工具模块
│   ├── bm25_retriever.py   # BM25检索
│   ├── document_parser.py  # 文档解析
│   ├── query_rewriter_v2.py # 查询改写
│   ├── qwen_reranker.py    # 通义千问Rerank
│   ├── tavily_search.py    # Tavily搜索
│   └── text_splitter.py    # 文本切分
├── config.py               # 配置文件
├── build_index.py          # 构建索引
├── query.py                # 查询入口
└── requirements.txt        # 依赖
```

## 配置说明

主要配置项在 `config.py` 中：

- `LLM_MODEL`: LLM模型（默认: qwen-flash）
- `CHUNK_SIZE`: 分块大小（默认: 600）
- `RETRIEVAL_K`: 检索结果数量（默认: 5）
- `USE_RERANK`: 是否启用Rerank（默认: True）
- `RERANK_TOP_K`: Rerank后返回数量（默认: 3）

## 查询示例

```
请输入问题: 应收账款坏账准备的会计分录如何编制？

请输入问题: 专项扣除和专项附加扣除有什么区别？

请输入问题: 2024年最新的个人所得税税率表是怎样的？
```

## 特色功能

### 1. 术语混淆防护

自动识别并保护会计核心术语：
- 专项扣除 vs 专项附加扣除
- 进项税额 vs 销项税额
- 权责发生制 vs 收付实现制

### 2. 智能复合问题处理

识别多部分问题，保持原意检索。

### 3. 联网搜索增强

自动检测时效性需求，使用Tavily搜索最新政策。

## License

MIT License

