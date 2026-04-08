# RAG Chat - PDF 智能问答助手

基于 **LlamaIndex** + **Streamlit** 实现的 RAG（检索增强生成）应用。用户上传 PDF 文件后，可以用自然语言提问，系统根据文档内容给出准确、可靠的回答。

支持 **阿里云通义千问（DashScope）** 和 **OpenRouter** 多种大语言模型，并内置评估数据集生成与 RAG 质量评测功能。

## 项目目标与价值

- 支持单份 PDF 的智能问答
- 同时使用向量检索（适合找细节、引用、事实）和摘要检索（适合总结、整体理解）
- 通过路由机制（Router Retriever）让 LLM 自动判断使用哪种检索方式
- 使用重排序（Reranker）提升检索结果的精确性
- 支持流式输出 + 模型推理过程展示，提升用户体验
- 内置数据集生成与评估管线，便于量化 RAG 效果
- 适用场景：论文阅读、合同审查、技术手册查询、内部知识库问答等

## 技术栈一览

| 类别           | 技术 / 库                            | 主要用途                          |
|----------------|--------------------------------------|-----------------------------------|
| 前端界面       | Streamlit                           | 快速构建交互式聊天界面            |
| RAG 核心框架   | LlamaIndex                          | 文档加载、分块、索引、检索、对话引擎 |
| 大模型         | 通义千问 (qwen-max)、OpenRouter (Step 3.5 Flash, GPT-oss 20B) | 文本生成、路由选择 |
| 向量嵌入       | DashScope text-embedding-v2         | 生成文本向量                      |
| PDF 解析       | SimpleDirectoryReader               | 加载 PDF 文档                     |
| 索引类型       | VectorStoreIndex + SummaryIndex     | 向量检索 + 文档摘要检索           |
| 检索路由       | RouterRetriever + LLMSingleSelector | 智能选择检索工具                  |
| 重排序         | DashScopeRerank (gte-rerank)        | 对检索结果重排，提升 Top-N 精确性  |
| 对话记忆       | ChatMemoryBuffer (可配置 token 上限) | 支持多轮上下文对话               |
| 数据集生成     | RagDatasetGenerator                 | 从文档自动生成问答对用于评估      |
| 评估指标       | FaithfulnessEvaluator + RelevancyEvaluator | 忠实度与相关性评测           |
| 环境变量管理   | python-dotenv                       | 安全管理 API Key                  |

## 核心实现逻辑

### 1. 文档处理流程
- 用户上传 PDF → 保存到临时目录
- 使用 SimpleDirectoryReader 解析文档
- SentenceSplitter（块大小 1024，交叠 150）进行分块

### 2. 双索引设计
- **VectorStoreIndex**：用于精确定位细节、引用、具体事实
- **SummaryIndex**：用于回答概括性、整体理解类问题

### 3. 智能路由检索
```python
router_retriever = RouterRetriever(
    selector=LLMSingleSelector.from_defaults(),
    retriever_tools=[summary_tool, vector_tool],
    verbose=True
)
```
LLM 根据问题自动判断：
- 用 summary_tool（问主题、总结、大方向）
- 用 vector_tool（问具体段落、数据、引用）

### 4. 重排序（Reranker）
检索结果经过 DashScope gte-rerank 模型重排，从 Top-K（默认 15）中筛选出最相关的 Top-N（默认 5）节点，提升回答质量。

### 5. 对话引擎
- 使用 ContextChatEngine（保留完整上下文）
- 自定义 system prompt，强制模型基于上传的 PDF 内容回答，不编造内容
- 支持 streaming 流式输出（逐 token 显示）
- 内置 ChatMemoryBuffer 支持多轮对话

### 6. 思考过程可视化
- 自动识别模型输出中的 `` 标签
- 用 `st.expander` 折叠展示推理过程，主回答保持简洁

## 模块说明

| 文件                  | 功能                                     |
|-----------------------|------------------------------------------|
| `demo.py`             | Streamlit 前端应用入口                   |
| `utils.py`            | 核心工具函数：LLM 路由、索引构建、检索引擎配置 |
| `dataset_generate.py` | 从 PDF 文档自动生成评估数据集（问答对）    |
| `eval.py`             | 对生成的数据集进行评估，输出忠实度与相关性指标 |

## 如何运行

### 环境要求

- Python 3.10+
- 可联网访问 dashscope.aliyuncs.com 和 openrouter.ai

### 安装依赖

```bash
pip install streamlit llama-index llama-index-llms-openai-like llama-index-llms-openrouter llama-index-embeddings-dashscope llama-index-postprocessor-dashscope-rerank pymupdf python-dotenv
```

推荐创建 `requirements.txt`：

```text
streamlit>=1.38.0
llama-index>=0.10.0
llama-index-llms-openai-like
llama-index-llms-openrouter
llama-index-embeddings-dashscope
llama-index-postprocessor-dashscope-rerank
llama-index-readers-file
pymupdf
python-dotenv
pandas
```

### 配置 API Key

项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-你的通义千问API密钥
OPENROUTER_API_KEY=sk-你的OpenRouter API密钥
```

> 两个 Key 均需配置，DashScope 用于嵌入模型和 Qwen 模型，OpenRouter 用于免费模型。

### 启动应用

```bash
streamlit run demo.py
```

浏览器会自动打开：http://localhost:8501

## 使用步骤

1. 侧边栏选择模型（Qwen Max / Step 3.5 Flash / GPT-oss 20B）
2. 上传一份 PDF 文件
3. 等待「PDF 加载完成」和「索引构建完成」
4. 在下方输入框提问
5. 可点击「清空对话」重置所有状态

## 数据集生成与评估

### 生成评估数据集

将待评估的 PDF 文件放入 `Files/` 目录，然后运行：

```bash
python dataset_generate.py
```

脚本会：
1. 加载 `Files/` 目录下的文档
2. 使用 Qwen Max 为每个文本块生成 2 个问答对
3. 保存为 `datasets/my_dataset.json`

### 运行评估

将 PDF 文件放入 `Files/` 目录，确保 `datasets/my_dataset.json` 已生成，然后运行：

```bash
python eval.py
```

脚本会：
1. 加载评估数据集
2. 使用 ChatEngine 逐一生成回答
3. 分别评估 **忠实度（Faithfulness）** 和 **相关性（Relevancy）**
4. 输出汇总结果到 `eval/result.csv`，包含每个问题的回答、各项指标的通过情况及模型反馈

评估指标说明：
- **忠实度（Faithfulness）**：回答是否基于检索到的上下文，是否存在幻觉/编造
- **相关性（Relevancy）**：回答是否与问题相关

## 当前限制 & 未来改进方向

- 仅支持单份 PDF（可扩展为多文件/文件夹）
- 未显示引用来源/页码（可添加 Citation 后处理）
- 未处理表格、图片内容（可引入 LlamaParse 或 Unstructured）
- 可进一步加入 hybrid search 提升检索召回率
- 评估数据集的生成质量依赖 LLM，可加入人工校验环节
