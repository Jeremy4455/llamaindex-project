import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, SummaryIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import RetrieverTool
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv
import tempfile
import shutil
import re

from llama_index.readers.file import PyMuPDFReader

# Load environment variables
load_dotenv()


def format_reasoning_response(thinking_content):
    """清理 <think> 标签（如果模型输出带有）"""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )


def display_assistant_message(content):
    """显示 assistant 回复，自动展开 thinking 部分（如果有）"""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)

    if think_match:
        think_block = think_match.group(0)
        main_response = content.replace(think_block, "").strip()
        think_clean = format_reasoning_response(think_block)

        with st.expander("模型推理过程"):
            st.markdown(think_clean)
        st.markdown(main_response)
    else:
        st.markdown(content)


def main():
    st.set_page_config(page_title="RAG Chat", layout="wide")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # 标题
    st.title("RAG Chat with LlamaIndex")

    # 清除聊天记录按钮
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.docs_loaded = False
        if st.session_state.temp_dir:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
        st.session_state.current_pdf = None
        st.rerun()

    # 侧边栏 - 配置 & PDF 上传
    with st.sidebar:
        # 模型选择（目前只保留常用模型，可自行扩展）
        model_options = ["qwen-plus", "qwen-max"]
        selected_model = st.selectbox(
            "选择生成模型",
            model_options,
            index=0
        )

        st.divider()

        st.subheader("上传 PDF")
        uploaded_file = st.file_uploader(
            "选择 PDF 文件",
            type="pdf",
            accept_multiple_files=False
        )

        if uploaded_file is not None:
            if uploaded_file != st.session_state.current_pdf:
                st.session_state.current_pdf = uploaded_file

                try:
                    if not os.getenv("DASHSCOPE_API_KEY"):
                        st.error("请在 .env 文件中设置 DASHSCOPE_API_KEY")
                        st.stop()

                    # 清理旧的临时目录
                    if st.session_state.temp_dir:
                        shutil.rmtree(st.session_state.temp_dir)

                    st.session_state.temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("正在加载 PDF..."):
                        file_extractor = {".pdf": PyMuPDFReader()}
                        documents = SimpleDirectoryReader(
                            st.session_state.temp_dir,
                            file_extractor=file_extractor
                        ).load_data()

                        st.session_state.docs_loaded = True
                        st.session_state.documents = documents
                        st.success("PDF 加载完成")

                    if st.session_state.docs_loaded and st.session_state.chat_engine is None:
                        with st.spinner("正在构建索引..."):
                            llm = OpenAILike(
                                model=selected_model,
                                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                api_key=os.getenv("DASHSCOPE_API_KEY"),
                                is_chat_model=True,
                                temperature=0.7,
                                max_tokens=2048,
                                context_window=1000000,
                            )
                            embed_model = DashScopeEmbedding(
                                model_name="text-embedding-v2"
                            )
                            Settings.llm = llm
                            Settings.embed_model = embed_model

                            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
                            nodes = splitter.get_nodes_from_documents(st.session_state.documents)

                            summary_index = SummaryIndex(nodes)
                            vector_index = VectorStoreIndex(nodes)

                            summary_retriever = summary_index.as_retriever()
                            vector_retriever = vector_index.as_retriever()

                            summary_tool = RetrieverTool.from_defaults(
                                retriever=summary_retriever,
                                description="Useful for summarization questions and getting a high-level overview."
                            )

                            vector_tool = RetrieverTool.from_defaults(
                                retriever=vector_retriever,
                                description="Useful for retrieving specific context, details, quotes, or facts from the document."
                            )

                            router_retriever = RouterRetriever(
                                selector=LLMSingleSelector.from_defaults(),
                                retriever_tools=[summary_tool, vector_tool],
                                verbose=True
                            )

                            memory = ChatMemoryBuffer.from_defaults(
                                token_limit=100000,
                            )

                            chat_engine = ContextChatEngine.from_defaults(
                                retriever=router_retriever,
                                memory=memory,
                                system_prompt=(
                                    "你是一个专业的PDF问答助手，要根据提供的文件内容进行回答，"
                                    "不要编造内容，优先基于提供的PDF内容。"
                                ),
                                verbose=True
                            )

                            st.session_state.chat_engine = chat_engine
                        st.success("索引构建完成")

                except Exception as e:
                    st.error(f"加载失败：{str(e)}")

    # 显示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_message(message["content"])
            else:
                st.markdown(message["content"])

    # 用户输入框
    if prompt := st.chat_input("请问关于这份 PDF 的任何问题..."):
        if not st.session_state.docs_loaded:
            st.error("请先上传 PDF 文件")
            st.stop()

        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回复
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    message_placeholder = st.empty()  # 用于逐字更新
                    full_response = ""

                    stream_response = st.session_state.chat_engine.stream_chat(prompt)

                    for token in stream_response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")  # 光标效果

                    message_placeholder.markdown(full_response)

                    # 保存完整回复
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"生成失败：{str(e)}")


if __name__ == "__main__":
    main()
