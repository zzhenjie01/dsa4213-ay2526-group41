"""
Central configuration file for the MAB project.
"""
import os

# -----------------------------------------------
# Paths and Directories
# -----------------------------------------------
# Assumes scripts are run from the project's root directory
# Project root directory is 2 levels up from this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS_MARCO_QUERIES_WITH_QRELS_PATH = os.path.join(PROJECT_ROOT, "data/ms_marco/processed/queries_with_qrels.jsonl")
NQ_QUERIES_WITH_QRELS_PATH = os.path.join(PROJECT_ROOT, "data/nq/processed/queries_with_qrels.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------------------------
# Weaviate/Retrieval Configuration
# -----------------------------------------------
HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
MS_MARCO_COLLECTION_NAME = "MS_MARCO"
NQ_COLLECTION_NAME = "NQ"
QA_EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
SEARCH_LIMIT = 10  # k value for retrieval

# -----------------------------------------------
# Arm Configuration
# -----------------------------------------------
# Define all available arms (retrieval strategies) here.
# 'name' is a unique identifier.
# 'type' corresponds to a method in RetrievalEnvironment.
# 'params' are passed directly to the Weaviate query.
ARM_CONFIG = [
    {
        "name": "bm25",
        "type": "bm25",
        "params": {}  # Weaviate's defaults
    },
    {
        "name": "dense",
        "type": "dense",
        "params": {}
    },
    {
        "name": "hybrid_alpha(0.25)",
        "type": "hybrid",
        "params": {"alpha": 0.25}
    },
    {
        "name": "hybrid_alpha(0.5)",
        "type": "hybrid",
        "params": {"alpha": 0.5}
    },
    {
        "name": "hybrid_alpha(0.75)",
        "type": "hybrid",
        "params": {"alpha": 0.75}
    }
]

# -----------------------------------------------
# Experiment Parameters
# -----------------------------------------------
# Default reward trade-off parameter (accuracy vs. latency)
DEFAULT_LAMBDA = 0.5
# Default seed for reproducibility
DEFAULT_SEED = 42

# -----------------------------------------------
# Plotting Configuration
# -----------------------------------------------
# Window size for calculating moving averages in plots
MOVING_AVG_WINDOW = 1000