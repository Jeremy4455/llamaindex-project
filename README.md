# RAG Chat - PDF 智能问答助手

基于 **LlamaIndex** + **Streamlit** 实现的 RAG（检索增强生成）应用。用户上传 PDF 文件后，可以用自然语言提问，系统根据文档内容给出准确、可靠的回答。

目前使用 **阿里云通义千问（DashScope）** 作为大语言模型和 Embedding 模型。

## 项目目标与价值

- 支持单份 PDF 的智能问答
- 同时使用向量检索（适合找细节、引用、事实）和摘要检索（适合总结、整体理解）
- 通过路由机制（Router Retriever）让 LLM 自动判断使用哪种检索方式
- 支持流式输出 + 模型推理过程展示，提升用户体验
- 适用场景：论文阅读、合同审查、技术手册查询、内部知识库问答等

## 技术栈一览

| 类别           | 技术 / 库                            | 主要用途                          |
|----------------|--------------------------------------|-----------------------------------|
| 前端界面       | Streamlit                           | 快速构建交互式聊天界面            |
| RAG 核心框架   | LlamaIndex                          | 文档加载、分块、索引、检索、对话引擎 |
| 大模型         | 通义千问 (qwen-plus / qwen-max)     | 文本生成、路由选择                |
| 向量嵌入       | DashScope text-embedding-v2         | 生成文本向量                      |
| PDF 解析       | PyMuPDFReader                       | 高质量提取 PDF 文本与结构         |
| 索引类型       | VectorStoreIndex + SummaryIndex     | 向量检索 + 文档摘要检索           |
| 检索路由       | RouterRetriever + LLMSingleSelector | 智能选择检索工具                  |
| 对话记忆       | ChatMemoryBuffer (约 10 万 token)   | 支持多轮上下文对话                |
| 环境变量管理   | python-dotenv                       | 安全管理 API Key                  |

## 核心实现逻辑

1. **文档处理流程**  
   - 用户上传 PDF → 保存到临时目录  
   - 使用 PyMuPDFReader 解析 → 得到 LlamaIndex Document 对象  
   - SentenceSplitter（块大小 1024，交叠 100）进行分块

2. **双索引设计**  
   - **VectorStoreIndex**：用于精确定位细节、引用、具体事实  
   - **SummaryIndex**：用于回答概括性、整体理解类问题

3. **智能路由检索（核心创新点）**  
   ```python
   router_retriever = RouterRetriever(
       selector=LLMSingleSelector.from_defaults(),
       retriever_tools=[summary_tool, vector_tool]
   )
	```
   LLM 根据问题自动判断：  
   - 用 summary_tool（问主题、总结、大方向）  
   - 用 vector_tool（问具体段落、数据、引用）

4. **对话引擎**  
   - 使用 ContextChatEngine（保留完整上下文）  
   - 自定义 system prompt，强制模型只基于上传的 PDF 内容回答  
   - 支持 streaming 流式输出（逐 token 显示）

5. **思考过程可视化**  
   - 自动识别模型输出中的 `<think>...</think>` 标签  
   - 用 st.expander 折叠展示推理过程，主回答保持简洁

## 如何运行

### 环境要求

- Python 3.10+
- 可联网访问 dashscope.aliyuncs.com

### 安装依赖

```bash
pip install streamlit llama-index llama-index-llms-openai-like llama-index-embeddings-dashscope pymupdf python-dotenv
```

推荐创建 `requirements.txt`：

```text
streamlit>=1.38.0
llama-index>=0.10.0
llama-index-llms-openai-like
llama-index-embeddings-dashscope
llama-index-readers-file
pymupdf
python-dotenv
```

### 配置 API Key

项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-你的通义千问API密钥
```

### 启动应用

```bash
streamlit run demo.py
```

浏览器会自动打开：http://localhost:8501

## 使用步骤

1. 侧边栏选择模型（qwen-plus 或 qwen-max）
2. 上传一份 PDF 文件（当前仅支持单文件）
3. 等待「索引构建完成」
4. 在下方输入框提问
5. 可点击「清空对话」重置所有状态

## 当前限制 & 未来改进方向

- 仅支持单份 PDF（可扩展为多文件/文件夹）
- 未显示引用来源/页码（可添加 Citation 后处理）
- 未处理表格、图片内容（可引入 LlamaParse 或 Unstructured）
- 可进一步加入 hybrid search、重排序（reranker）提升检索精度
