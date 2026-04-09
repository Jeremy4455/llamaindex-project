import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.llms.openai_like import OpenAILike
from utils import get_chat_engine, ALL_MODELS
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
import pandas as pd

dataset_path = "datasets/my_dataset.json"

load_dotenv()

model_name = "Llama 3.1 8B"
chat_engine = get_chat_engine(model_name, docs=SimpleDirectoryReader('Files').load_data(), m_size=ALL_MODELS[model_name]['context_window'] * 0.5)

eval_llm = OpenAILike(
    model='qwen-max',
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True,
    temperature=0
)

faith_evaluator = FaithfulnessEvaluator(llm=eval_llm)
relev_evaluator = RelevancyEvaluator(llm=eval_llm)

rag_dataset = LabelledRagDataset.from_json(dataset_path)
results_list = []

print("开始评估...")
for i, example in enumerate(rag_dataset.examples):
    print(f"[{i + 1}/{len(rag_dataset.examples)}] 正在处理问题: {example.query[:30]}...")

    try:
        response = chat_engine.chat(example.query)

        faith_result = faith_evaluator.evaluate_response(response=response)
        relev_result = relev_evaluator.evaluate_response(query=example.query, response=response)

        results_list.append({
            "Question": example.query,
            "Response": response.response,
            "Faithfulness": faith_result.passing,
            "Relevancy": relev_result.passing,
            "Faith_Feedback": faith_result.feedback,
            "Relev_Feedback": relev_result.feedback,
        })

    except Exception as e:
        print(f"Error processing example {i + 1}: {str(e)}")

        results_list.append({
            "Question": example.query,
            "Response": "ERROR",
            "Faithfulness": None,
            "Relevancy": None,
            "Faith_Feedback": None,
            "Relev_Feedback": None,
        })
        continue

df = pd.DataFrame(results_list)
faith_score = df["Faithfulness"].mean()
relev_score = df["Relevancy"].mean()

df.to_csv(f"eval/{model_name.lower().replace(" ", "_")}_result.csv")

print(f"\n评估完成！\n忠实度 (Faithfulness): {faith_score:.2%}")
print(f"相关性 (Relevancy): {relev_score:.2%}")
