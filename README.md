<div align="center">

# 🛍️ RAG 智能服装客服系统

**基于 RAG（检索增强生成）的智能客服，专为服装电商场景打造**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--chat-6C5CE7.svg)](https://deepseek.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目概述

本项目是一个轻量级、易于部署的 **RAG 智能客服系统**，面向服装零售场景，帮助顾客解决三大核心问题：

| 场景 | 说明 |
|------|------|
| **📏 尺码推荐** | 根据身高、体重自动推荐 S~5XL 的合适尺码 |
| **🧺 洗涤养护** | 按面料材质（棉/麻/丝/羊毛/羽绒等）和季节提供专业洗护指导 |
| **🎨 颜色选择** | 根据肤色、场合、体型、季节推荐服装颜色搭配 |

**核心工作流：** 用户提问 → 向量库检索相关文档 → 组装 Prompt → DeepSeek 大模型生成专业回答，支持流式输出与多轮对话。

---

## ✨ 功能亮点

| 功能 | 说明 |
|------|------|
| **🔍 语义检索** | 基于 Embedding 向量相似度，精准匹配最相关的知识片段（top-k） |
| **💬 多轮对话** | 自动保存/加载聊天历史，支持上下文连续追问 |
| **⚡ 流式输出** | 回答逐字生成显示，无需等待完整结果 |
| **📚 动态知识库** | 通过 Web 页面上传 .txt 文件，自动切片、去重、入库 |
| **🧹 MD5 去重** | 相同内容不会重复入库，节省存储和 Token 成本 |
| **🔧 配置分离** | 所有参数集中管理，修改一处全局生效 |
| **💾 持久化存储** | ChromaDB 存磁盘，程序重启数据不丢失 |
| **🖥️ 纯 Python UI** | 基于 Streamlit，无需写 HTML/CSS/JS |

---

## 🏗️ 技术栈

```
┌─────────────────────────────────────────────────────┐
│                    前端层 (Streamlit)                 │
│           app_qa.py  │  app_file_uploader.py         │
├─────────────────────────────────────────────────────┤
│                   RAG 编排层 (LangChain LCEL)         │
│              rag.py  │  vector_stores.py             │
├─────────────────────────────────────────────────────┤
│              数据层 (ChromaDB + JSON)                │
│     knowledge_besa.py  │  file_history_store.py     │
├─────────────────────────────────────────────────────┤
│                   外部服务 (Cloud APIs)               │
│     DeepSeek API (LLM)  │  阿里云百炼 (Embedding)    │
└─────────────────────────────────────────────────────┘
```

### 核心依赖

| 层级 | 技术/库 | 用途 |
|------|---------|------|
| 编程语言 | **Python 3.10+** | 全部代码 |
| Web 框架 | **Streamlit** | 聊天 UI + 管理后台 |
| LLM 编排 | **LangChain (LCEL)** | RAG 管道链式编排 |
| 大语言模型 | **DeepSeek** (`deepseek-chat`) | 对话生成（兼容 OpenAI 接口） |
| 向量数据库 | **ChromaDB** | 文本向量存储与相似度检索 |
| Embedding | **阿里云百炼 DashScope** (`text-embedding-v4`) | 文本向量化 |
| 文本切片 | **RecursiveCharacterTextSplitter** | 长文本智能分段 |
| 去重 | **hashlib (MD5)** | 内容指纹去重 |
| 对话历史 | **JSON 本地文件** | 按会话 ID 存储 |

### 安装依赖

```bash
pip install streamlit langchain langchain-core langchain-community \
            langchain-openai langchain-chroma chromadb
```

---

## 📁 项目结构

```
RAG/
├── README.md                         # 项目说明（本文件）
├── LICENSE                           # MIT 开源协议
├── .gitignore                        # Git 忽略规则
│
└── 项目案例RAG/                      # 主项目目录
    ├── config_data.py                # 全局配置（API Key、路径、切片参数）
    ├── rag.py                        # RAG 对话服务（检索 + 生成 + 多轮对话）
    ├── vector_stores.py              # 向量库检索封装（ChromaDB 查询）
    ├── knowledge_besa.py             # 知识库入库服务（切片 + 去重 + 向量化写入）
    ├── file_history_store.py         # 聊天历史本地存储（JSON 文件读写）
    │
    ├── app_qa.py                     # 【页面 1】智能客服问答页面
    ├── app_file_uploader.py          # 【页面 2】知识库文件上传页面
    │
    ├── data/                         # 知识库原始文档
    │   ├── 尺码推荐.txt
    │   ├── 洗涤养护.txt
    │   └── 颜色选择.txt
    │
    ├── chroma_db/                    # ChromaDB 向量数据库（持久化）
    ├── chat_history/                 # 聊天历史存储（按会话 ID 分文件）
    └── .md5.text                     # MD5 去重指纹记录
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/lumfei/RAG.git
cd RAG/项目案例RAG

# 安装依赖
pip install streamlit langchain langchain-core langchain-community \
            langchain-openai langchain-chroma chromadb
```

### 2. 配置 API Key

编辑 `config_data.py`，填入你的 API Key：

```python
# 阿里云百炼 DashScope（用于文本向量化）
dashscope_api_key = "sk-你的key"

# DeepSeek（用于大模型对话生成）
deepseek_api_key = "sk-你的key"
```

> 💡 **推荐使用环境变量**（更安全）：
> ```bash
> export DASHSCOPE_API_KEY="sk-你的key"
> export DEEPSEEK_API_KEY="sk-你的key"
> ```

### 3. 上传知识库

```bash
streamlit run app_file_uploader.py
```

浏览器打开 `http://localhost:8501`，上传 `data/` 目录下的 .txt 文件：
- `尺码推荐.txt` — 身高体重与尺码对照
- `洗涤养护.txt` — 面料洗护知识
- `颜色选择.txt` — 肤色场合颜色搭配

### 4. 启动智能客服

```bash
streamlit run app_qa.py
```

在聊天框中提问，例如：

```
我身高172cm，体重120斤，应该穿什么尺码？
真丝连衣裙怎么洗？
黄皮肤适合什么颜色的衣服？
```

---

## ⚙️ 配置说明

所有可调参数集中在 `config_data.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | `1000` | 文本切片最大字符数 |
| `chunk_overlap` | `100` | 相邻切片重叠字符数 |
| `similarity_threshold` | `2` | 检索返回 top-k 数量 |
| `collection_name` | `"rag"` | ChromaDB 集合名称 |
| `persist_directory` | `"./chroma_db"` | 向量库存储路径 |
| `embedding_model_name` | `"text-embedding-v4"` | Embedding 模型 |
| `deepseek_model_name` | `"deepseek-chat"` | LLM 模型 |
| `deepseek_base_url` | `"https://api.deepseek.com/v1"` | LLM API 地址 |
| `session_config` | `{"session_id": "user_001"}` | 默认会话 ID |

> 💡 **切换 LLM：** 修改 `deepseek_model_name` 和 `deepseek_base_url` 即可对接任何兼容 OpenAI 接口的模型（如 OpenAI GPT、本地 vLLM 等）。

---

## 🏛️ 架构详解

### 知识入库流程

```
用户上传 .txt 文件
    │
    ▼
[1] MD5 去重检查 ──→ 已存在 → 跳过
    │ 新内容
    ▼
[2] RecursiveCharacterTextSplitter 文本切片
    chunk_size=1000, chunk_overlap=100
    按语义边界切分：段落 → 换行 → 句号 → 空格 → 逐字
    │
    ▼
[3] DashScope text-embedding-v4 生成向量
    │
    ▼
[4] 向量 + 元数据（来源、时间、操作人）→ ChromaDB
    │
    ▼
[5] 记录 MD5 指纹 → .md5.text
```

### RAG 问答流程

```
用户输入问题
    │
    ▼
[1] 问题向量化 → ChromaDB 相似度检索 → Top-K 文档片段
    │
    ▼
[2] 组装 Prompt：
    System: 角色设定 + 检索到的参考文档
    History: 当前会话聊天历史（多轮上下文）
    User: 用户提问
    │
    ▼
[3] DeepSeek LLM 流式生成
    │
    ▼
[4] 保存聊天历史 + 流式展示回答
```

### LCEL 管道链（rag.py 核心）

```python
RunnableLambda(提取问题)
  → retriever (向量检索)
  → format_document (格式化检索结果)
  ├─ 同时透传 input (RunnablePassthrough)
  → format_for_prompt_template (重组数据)
  → ChatPromptTemplate (填入提示词)
  → ChatOpenAI (DeepSeek 流式生成)
  → StrOutputParser (解析文本)
```

外层包裹 `RunnableWithMessageHistory` 实现多轮对话历史自动管理。

---

## 🔧 常见问题

<details>
<summary><b>Q: 上传文件时报 API 错误？</b></summary>

检查 `config_data.py` 中的 `dashscope_api_key` 是否正确配置，推荐使用环境变量。
</details>

<details>
<summary><b>Q: 问答时没有返回相关知识？</b></summary>

确保已通过 `app_file_uploader.py` 上传了知识文档，且 `chroma_db/` 目录存在并包含数据。
</details>

<details>
<summary><b>Q: 想清空向量库重新开始？</b></summary>

删除 `chroma_db/` 文件夹和 `.md5.text` 文件，然后重新上传知识文档。
</details>

<details>
<summary><b>Q: 如何切换会话/用户？</b></summary>

修改 `config_data.py` 中的 `session_id`：
```python
session_config = {"configurable": {"session_id": "user_002"}}
```
</details>

<details>
<summary><b>Q: 可以换成其他 LLM 吗？</b></summary>

可以。只要兼容 OpenAI 接口（`/v1/chat/completions`），修改 `deepseek_model_name` 和 `deepseek_base_url` 即可。
</details>

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📄 License

本项目基于 MIT 协议开源 — 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**Built with ❤️ using Python · Streamlit · LangChain · DeepSeek · ChromaDB**

</div>
