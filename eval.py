import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.llms.openai_like import OpenAILike
from utils import get_chat_engine
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
import pandas as pd

dataset_path = "datasets/my_dataset.json"

load_dotenv()

chat_engine = get_chat_engine("Qwen Max", docs=SimpleDirectoryReader('Files').load_data())

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
for example in rag_dataset.examples:
    response = chat_engine.chat(example.query)

    # 运行评估
    faith_result = faith_evaluator.evaluate_response(response=response)
    relev_result = relev_evaluator.evaluate_response(query=example.query, response=response)

    results_list.append({
        "Question": example.query,
        "Response": response.response,
        "Faithfulness": faith_result.passing,  # True/False
        "Relevancy": relev_result.passing,  # True/False
        "Faith_Feedback": faith_result.feedback,
        "Relev_Feedback": relev_result.feedback
    })

df = pd.DataFrame(results_list)
faith_score = df["Faithfulness"].mean()
relev_score = df["Relevancy"].mean()

df.to_csv("eval/result.csv")

print(f"\n评估完成！\n忠实度 (Faithfulness): {faith_score:.2%}")
print(f"相关性 (Relevancy): {relev_score:.2%}")
