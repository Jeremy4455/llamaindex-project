import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from utils import get_chat_engine, ALL_MODELS

load_dotenv()

async def run_evaluation():
    source_dir = "./datasets/paul_graham/source_files"
    dataset_json = "./datasets/paul_graham/rag_dataset.json"

    print("Loading local text and dataset...")
    try:
        documents = SimpleDirectoryReader(source_dir).load_data()
        rag_dataset = LabelledRagDataset.from_json(dataset_json)
    except Exception as e:
        print(f"Failed to load files: {e}")
        return

    # 1. Setup Llama 3.1 8B
    print("Initializing Llama 3.1 8B...")
    llm = OpenRouter(
        model="meta-llama/llama-3.1-8b-instruct",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    Settings.llm = llm

    print("Setting up embedding model...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 2. Build Standard Vector Index
    print("Building RAG engine...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 3. Initialize Evaluators (Faithfulness & Relevancy)
    print("Initializing evaluators...")
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
    relevancy_evaluator = RelevancyEvaluator(llm=llm)

    eval_results = []
    output_dir = "eval"
    output_csv_path = f"{output_dir}/pg_hallucination_report.csv"
    os.makedirs(output_dir, exist_ok=True)

    # Variables for final score calculation
    total_queries = 0
    faithfulness_pass_count = 0
    relevancy_pass_count = 0

    print("\n--- Starting Evaluation ---")
    
    test_examples = rag_dataset.examples 
    total_queries = len(test_examples)
    
    for i, example in enumerate(test_examples):
        query = example.query
        print(f"\n[Test {i+1}/{total_queries}] Query: {query}")

        # RAG generates answer
        response = query_engine.query(query)
        print(f"Response: {str(response)[:100]}...")
        
        # Evaluate Faithfulness
        eval_faith = await faithfulness_evaluator.aevaluate_response(response=response)
        if eval_faith.passing:
            faithfulness_pass_count += 1
            
        # Evaluate Relevancy
        eval_rel = await relevancy_evaluator.aevaluate_response(query=query, response=response)
        if eval_rel.passing:
            relevancy_pass_count += 1

        print(f"Faithfulness: {'Pass' if eval_faith.passing else 'Fail'}, Relevancy: {'Pass' if eval_rel.passing else 'Fail'}")

        eval_results.append({
            "ID": i + 1,
            "Query": query,
            "Response": str(response),
            "Faithfulness_Pass": eval_faith.passing,
            "Relevancy_Pass": eval_rel.passing,
            "Source_Context": [n.get_content()[:150] + "..." for n in response.source_nodes]
        })

    # 4. Print Final Metrics
    print("\n==========================================")
    print("           FINAL EVALUATION SCORES          ")
    print("==========================================")
    faithfulness_score = (faithfulness_pass_count / total_queries) * 100
    relevancy_score = (relevancy_pass_count / total_queries) * 100
    
    print(f"Total Queries Evaluated : {total_queries}")
    print(f"Faithfulness    : {faithfulness_score:.2f}% ({faithfulness_pass_count}/{total_queries} passed)")
    print(f"Relevancy       : {relevancy_score:.2f}% ({relevancy_pass_count}/{total_queries} passed)")
    print("==========================================\n")

    # 5. Export to CSV
    df = pd.DataFrame(eval_results)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Detailed evaluation report saved to: {output_csv_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())