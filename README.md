# LangChain Learn

基于 **LangChain 1.x** 的 Python 学习项目，以 Jupyter Notebook 形式系统讲解 LangChain 生态的核心概念与实战用法：从模型接入、消息与提示词、工具调用、结构化输出，到 Agent、中间件、记忆机制，最终以 RAG（检索增强生成）实战收尾。

> 配套的生态导读见 [Tutorial/01-LangChain家族四大支柱](Tutorial/01-LangChain家族四大支柱/01-LangChain家族四大支柱.md)。

## 技术栈

| 类别 | 组件 |
| --- | --- |
| 核心框架 | LangChain 1.4.x、LangGraph、LangSmith |
| 模型接入 | OpenAI、DeepSeek、Anthropic、Ollama（本地） |
| 向量数据库 | Milvus（Milvus Lite）、ChromaDB |
| 文档处理 | unstructured、docling、PyMuPDF4LLM、markitdown |
| 其他 | Tavily 搜索、SQLite/Postgres Checkpoint |
| 运行环境 | Python ≥ 3.11（项目使用 3.11）、uv 包管理 |

## 目录结构

| 章节 | 内容 |
| --- | --- |
| [Tutorial](Tutorial/01-LangChain家族四大支柱/01-LangChain家族四大支柱.md) | LangChain 家族四大支柱：LangChain / LangGraph / Deep Agents / LangSmith |
| [Chapter02_model](Chapter02_model) | 模型接入：在线/本地初始化、参数配置、invoke、流式/批量/异步调用 |
| [Chapter03_langsmith](Chapter03_langsmith) | LangSmith 追踪平台基本用法 |
| [Chapter04_messages](Chapter04_messages) | 消息对象、消息管理器、ChatPromptTemplate 基础与高级特性 |
| [Chapter05_tools](Chapter05_tools) | 工具使用、定义、装饰器、参数 Schema、应用场景、工具选择 |
| [Chapter06_structure_output](Chapter06_structure_output) | 结构化输出：Pydantic、TypedDict、JSON Schema、dataclass |
| [Chapter07_AGENT](Chapter07_AGENT) | Agent 基本用法、工具调用流程、高级用法、结构化输出、ToolStrategy、错误处理、流式输出、实战多功能智能助手 |
| [Chapter08_Middleware](Chapter08_Middleware) | 中间件：概述、摘要、Human-in-the-Loop、PII 脱敏、TodoList、自定义中间件、Hook 执行顺序 |
| [Chapter09_memory](Chapter09_memory) | 记忆机制：概述、短期记忆、SQL 存储、记忆策略、长期记忆（API/搜索/Agent/异步） |
| [Chapter10_RAG](Chapter10_RAG) | RAG 实战：概述、文档加载、文本切分、向量化（Embedding）、向量数据库、客服知识库助手 |

## 环境准备

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，Python 版本 3.11。

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
# 在 conf/ 目录下创建 .env 文件，内容结构见下方说明
```

## 环境变量配置

在 `conf/.env` 中配置所需模型与服务的 API Key（该文件已被 `.gitignore` 忽略，不会提交到仓库）：

```dotenv
# 模型服务（按需配置）
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

# LangSmith 追踪
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=...

# 其他服务
TAVILY_API_KEY=...
```

## 运行 Notebook

```bash
# 启动 Jupyter 并运行指定章节
uv run jupyter notebook Chapter10_RAG
```

各章节 Notebook 依赖对应服务，运行前请确保已完成上述环境变量配置。

## 参考资源

- [LangChain 文档](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith 文档](https://docs.langchain.com/langsmith/)
- [LangChain 开发栈概览](https://www.langchain.com/oss-overview)
