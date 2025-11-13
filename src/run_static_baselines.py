"""
Runs the "Static Arm" baselines.

This script iterates through each arm defined in config.py,
runs the entire dataset using only that arm, and reports
its average score and latency.
"""

from logging import config
import numpy as np
import argparse
import weaviate
from sentence_transformers import SentenceTransformer
import time
import json
import os
from tqdm import tqdm

from config import (
    HOST,
    PORT,
    GRPC_PORT,
    MS_MARCO_COLLECTION_NAME,
    NQ_COLLECTION_NAME,
    QA_EMBEDDING_MODEL_NAME,
    ARM_CONFIG,
    MS_MARCO_QUERIES_WITH_QRELS_PATH,
    NQ_QUERIES_WITH_QRELS_PATH,
    RESULTS_DIR,
    SEARCH_LIMIT
)
from helpers import setup_logging, load_queries_and_qrels
from environment import RetrievalEnvironment
from metrics import get_metric_fn

# Set the current script as the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory set to: {os.getcwd()}")

# Setup logging
logger = setup_logging("static_baseline")

def main(args):
    # --- Select Dataset ---
    if args.dataset_name == "ms_marco":
        QUERIES_WITH_QRELS_PATH = MS_MARCO_QUERIES_WITH_QRELS_PATH
        COLLECTION_NAME = MS_MARCO_COLLECTION_NAME
    elif args.dataset_name == "nq":
        QUERIES_WITH_QRELS_PATH = NQ_QUERIES_WITH_QRELS_PATH
        COLLECTION_NAME = NQ_COLLECTION_NAME
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    logger.info(f"Using dataset: {args.dataset_name}")
    
    # --- 1. Load Data and Models ---
    logger.info("Loading queries, qrels, and embedding model...")
    queries, qrels = load_queries_and_qrels(QUERIES_WITH_QRELS_PATH)
    query_ids = list(queries.keys())
    model = SentenceTransformer(QA_EMBEDDING_MODEL_NAME)
    logger.info("Data and models loaded.")

    # --- 2. Connect to Weaviate ---
    try:
        client = weaviate.connect_to_local(
            host=HOST,
            port=PORT,
            grpc_port=GRPC_PORT
        )
        logger.info("Connected to Weaviate successfully.")
        
        # --- 3. Initialize Environment ---
        env = RetrievalEnvironment(
            client=client,
            collection_name=COLLECTION_NAME,
            model=model,
            queries=queries,
            qrels=qrels,
            arm_config=ARM_CONFIG
        )
        
        metric_fn = get_metric_fn(args.metric)
        logger.info(f"Using metric: {args.metric}")

        # --- 4. Run Baseline for Each Arm ---
        all_arm_results = {}
        
        for arm_index, arm in enumerate(ARM_CONFIG):
            arm_name = arm['name']
            logger.info(f"--- Running baseline for Arm {arm_index}: {arm_name} ---")
            
            all_scores = []
            all_latencies = []
            
            # Use tqdm for a progress bar
            for qid in tqdm(query_ids, desc=f"Testing {arm_name}"):
                
                # Pull the arm
                retrieved_pids, latency = env.pull_arm(
                    arm_index, qid
                )
                relevant_pids = env.qrels.get(qid, set())

                # Calculate score
                score = metric_fn(retrieved_pids, relevant_pids, k=SEARCH_LIMIT)
                
                all_scores.append(score)
                all_latencies.append(latency)
            
            # --- Calculate and Save Arm Results ---
            avg_score = float(np.mean(all_scores))
            avg_latency = float(np.mean(all_latencies))
            
            arm_result = {
                "arm_index": arm_index,
                "arm_name": arm_name,
                "arm_type": arm['type'],
                "metric": args.metric,
                "avg_score": avg_score,
                "avg_latency_ms": avg_latency,
                "total_queries": len(query_ids)
            }
            
            all_arm_results[arm_name] = arm_result
            logger.info(f"Finished {arm_name}: Avg Score = {avg_score:.4f}, Avg Latency = {avg_latency:.2f}ms")

            # # Save individual arm result
            # save_path = os.path.join(RESULTS_DIR, f"baseline_static_arm_{args.dataset_name}_{arm_name}.json")
            # with open(save_path, "w") as f:
            #     json.dump(arm_result, f, indent=2)
                
        # --- Save Summary ---
        summary_path = os.path.join(RESULTS_DIR, f"baseline_static_summary_{args.dataset_name}_{int(time.time())}.json")
        with open(summary_path, "w") as f:
            json.dump(all_arm_results, f, indent=2)
        logger.info(f"All static baseline results saved to {summary_path}")


    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()
            logger.info("Weaviate client closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Static Arm Baselines.")
    
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        choices=["recall", "precision", "f1"],
        help="Accuracy metric to use for score calculation."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ms_marco",
        choices=["ms_marco", "nq"],
        help="Name of the dataset to use."
    )
    
    args = parser.parse_args()
    main(args)