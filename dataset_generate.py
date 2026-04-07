from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai_like import OpenAILike
import os


generator_llm = OpenAILike(
    model='qwen-max',
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True,
)

documents = SimpleDirectoryReader(
    'Files',
).load_data()

dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=generator_llm,
    num_questions_per_chunk=2,
    show_progress=True
)

rag_dataset = dataset_generator.generate_dataset_from_nodes()

rag_dataset.save_json("datasets/my_dataset.json")

