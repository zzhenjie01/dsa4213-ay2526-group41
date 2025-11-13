"""
Training script for the UCB bandit.
"""
import numpy as np
import argparse
import weaviate
from sentence_transformers import SentenceTransformer
import time
import os

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
    DEFAULT_LAMBDA,
    DEFAULT_SEED
)
from helpers import setup_logging, load_queries_and_qrels
from environment import RetrievalEnvironment
from experiment import run_experiment
from mab_algorithms import UCB
from metrics import get_metric_fn

# Set the current script as the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory set to: {os.getcwd()}")

# Setup logging
logger = setup_logging("train_ucb")

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

        # --- 4. Run Experiments for each 'c' value ---
        for c_val in args.c_values:
            # Create a new RNG for each run for reproducibility
            rng = np.random.default_rng(args.seed)
            
            bandit = UCB(
                k_arms=env.k_arms,
                rng=rng,
                c=c_val
            )
            
            run_name = (
                f"UCB_c={c_val}_lambda={args.lambda_param}_"
                f"metric={args.metric}_seed={args.seed}_dataset={args.dataset_name}"
                f"_{int(time.time())}"
            )
            
            run_experiment(
                bandit=bandit,
                env=env,
                query_ids=query_ids,
                metric_fn=metric_fn,
                lambda_param=args.lambda_param,
                run_name=run_name
            )

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'client' in locals() and client.is_connected():
            client.close()
            logger.info("Weaviate client closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UCB MAB.")
    parser.add_argument(
        "--c_values",
        type=float,
        nargs='+',
        default=[2.0],
        help="List of 'c' (exploration) values to try."
    )
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Reward trade-off parameter (score vs. latency)."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="recall",
        choices=["recall", "precision", "f1"],
        help="Accuracy metric to use for reward calculation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility."
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