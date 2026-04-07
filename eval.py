import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, context_recall
from ragas.integrations.llama_index import evaluate
from ragas.metrics.collections import context_precision
from utils import get_chat_engine

dataset_path = "datasets/my_dataset.json"

chat_engine = get_chat_engine("Step 3.5 Flash", docs=SimpleDirectoryReader('Files').load_data())

# 裁判 LLM (建议用强模型如 qwen-max 或 gpt-4o)
eval_llm = OpenAILike(
    model='qwen-max',
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True,
    temperature=0
)

embed_model = DashScopeEmbedding(model_name="text-embedding-v2")

if os.path.exists(dataset_path):
    rag_dataset = LabelledRagDataset.from_json(dataset_path)
    print(f"成功读取数据集，包含 {len(rag_dataset.examples)} 条测试用例")
else:
    raise FileNotFoundError(f"未找到数据集文件: {dataset_path}，请先运行生成脚本。")

test_df = rag_dataset.to_pandas()

result = evaluate(
    query_engine=chat_engine,
    metrics=[faithfulness, answer_relevancy, context_precision],
    dataset=test_df,
    llm=eval_llm,
    embeddings=embed_model
)

# --- 4. 结果展现 ---
print("\n--- 评估结果 ---")
print(result)
