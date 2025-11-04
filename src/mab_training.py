import numpy as np
import matplotlib.pyplot as plt
import time
import random
import jsonlines
import json
import weaviate
from mab_algorithms import EpsilonGreedy, UCB, ThompsonSampling
from sentence_transformers import SentenceTransformer
import os
import logging
from helpers import moving_average

# -----------------------------------------------
# Set random seed for reproducibility
# -----------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------
# Configure Script to be Current Directory
# -----------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Current working directory set to: {os.getcwd()}")

# -----------------------------------------------
# Global Constants
# -----------------------------------------------
HOST = "127.0.0.1"
PORT = 8080
GRPC_PORT = 50051
COLLECTION_NAME = "MS_MARCO"
QUERIES_QREL_PATH = "../data/processed/formatted_train_data.jsonl"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
ARM_METHODS = ["bm25", "dense", "hybrid"]
HYBRID_ALPHA = 0.5  # Weight for hybrid retrieval
SEARCH_LIMIT = 5    # Number of results to retrieve
LAMBDA_PARAMS = 0.5 # Trade-off parameter for reward calculation
MOVING_AVG_WINDOW = 1000
SAVE_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------
# Configure Logging
# -----------------------------------------------
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"mab_training_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------------------------
# Retrieval Environment
# -----------------------------------------------
class RetrievalEnvironment:
    def __init__(self, client, collection_name, model, queries, qrels, arm_methods):
        self.client = client
        self.collection = client.collections.use(collection_name)
        self.model = model
        self.queries = queries
        self.qrels = qrels
        self.arm_methods = arm_methods
        self.k_arms = len(arm_methods)

    def pull_arm(self, arm_index, query_id):
        """Perform retrieval using the selected method and return accuracy + latency"""
        query_text = self.queries[query_id]
        relevant_pids = self.qrels.get(query_id, set())
        method = self.arm_methods[arm_index]

        time_taken = 0.0

        # --- Perform retrieval based on the arm ---
        if method == "bm25":
            start_time = time.time()
            response = self.collection.query.bm25(
                query=query_text,
                limit=SEARCH_LIMIT,
                return_properties=["pid"]
            )
            time_taken = time.time() - start_time
        elif method == "dense":
            start_time = time.time()
            query_embedding = self.model.encode(query_text, show_progress_bar=False).tolist()

            response = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=SEARCH_LIMIT,
                return_properties=["pid"]
            )
            time_taken = time.time() - start_time
        elif method == "hybrid":
            start_time = time.time()
            query_embedding = self.model.encode(query_text, show_progress_bar=False).tolist()

            response = self.collection.query.hybrid(
                query=query_text,
                vector=query_embedding,
                alpha=HYBRID_ALPHA,
                limit=SEARCH_LIMIT,
                return_properties=["pid"]
            )
            time_taken = time.time() - start_time
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        latency_ms = time_taken * 1000

        # Compute accuracy
        if not response.objects:
            return 0, latency_ms

        retrieved_pids = [
            obj.properties["pid"] for obj in response.objects
        ]
        correct_retrievals = len(set(retrieved_pids) & relevant_pids)
        accuracy = correct_retrievals / len(relevant_pids) if relevant_pids else 0
        
        return accuracy, latency_ms
    
# -----------------------------------------------
# Load Queries and Qrels
# -----------------------------------------------
def load_queries_and_qrels(queries_qrels_path):
    queries, qrels = {}, {}

    with jsonlines.open(queries_qrels_path) as reader:
        for obj in reader:
            queries[obj["qid"]] = obj["query"]
            qrels[obj["qid"]] = set(obj["qrels"])

    return queries, qrels

# -----------------------------------------------
# Run MAB Training
# -----------------------------------------------
def run_experiment(bandit, env, query_id, lambda_params=LAMBDA_PARAMS):
    """Run MAB training for a given bandit and environment."""
    rewards = []
    total_reward = 0
    all_latencies, all_accuracies = [], []
    selected_arms = []

    # Keep track of min/max latency for stable O(1) normalization
    min_latency = float('inf')
    max_latency = float('-inf')

    for qid in query_id:
        arm_index = bandit.select_arm()
        selected_arms.append(arm_index)
        accuracy, cost = env.pull_arm(arm_index, qid) # cost is latency

        all_latencies.append(cost)
        all_accuracies.append(accuracy)

        # Update running min/max for normalization
        min_latency = min(min_latency, cost)
        max_latency = max(max_latency, cost)
        latency_range = max_latency - min_latency + 1e-6  # Avoid division by zero

        # Normalize reward to penalize latency
        norm_cost = (cost - min_latency) / latency_range
        reward = accuracy - lambda_params * norm_cost

        bandit.update(arm_index, reward)
        total_reward += reward
        rewards.append(reward)

        if len(rewards) % 1000 == 0:
            logger.info(f"[{bandit}] Step: {len(rewards)} | Arm: {arm_index} | Acc: {accuracy:.4f} | Latency: {cost:.4f} ms | Reward: {reward:.4f}")

    return rewards, all_accuracies, all_latencies, selected_arms

# -----------------------------------------------
# Main Execution
# -----------------------------------------------
if __name__ == "__main__":
    # Connect to Weaviate
    client = weaviate.connect_to_local(
        host=HOST,
        port=PORT,
        grpc_port=GRPC_PORT
    )
    logger.info("Connected to Weaviate successfully.")
    try:
        # Load queries and qrels
        queries, qrels = load_queries_and_qrels(QUERIES_QREL_PATH)
        logger.info(f"Loaded {len(queries)} queries and qrels.")

        # Initialize embedding model
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Loaded pre-trained embedding model.")

        # Initialize retrieval environment
        env = RetrievalEnvironment(
            client=client,
            collection_name=COLLECTION_NAME,
            model=model,
            queries=queries,
            qrels=qrels,
            arm_methods=ARM_METHODS
        )
        logger.info("Initialized Retrieval Environment.")
        
        query_ids = list(queries.keys())

        # Initialize MAB algorithms
        bandits = [
            EpsilonGreedy(len(ARM_METHODS), epsilon=0.1),
            UCB(len(ARM_METHODS), c=2),
            ThompsonSampling(len(ARM_METHODS))
        ]

        HOT_LATENCIES = {
            0: 1.7604,  # avg_bm25_latency
            1: 7.3123,  # avg_dense_latency
            2: 13.2463   # avg_hybrid_latency ms
        }
        ARM_NAMES = {
            0: "bm25",
            1: "dense",
            2: "hybrid"
        }

        plt.figure(figsize=(8, 5))

        # Run experiments for each bandit
        for bandit in bandits:
            logger.info(f"Starting experiment with {bandit}")
            rewards, all_accuracies, all_latencies, selected_arms = run_experiment(bandit, env, query_ids, LAMBDA_PARAMS)
            # Calculate inference latency (theoretical)
            num_queries = len(selected_arms)
            arm_counts = np.bincount(selected_arms, minlength=len(ARM_METHODS))

            theoretical_latency = 0.0
            proportions = {}
            for arm_index, count in enumerate(arm_counts):
                proportion = count / num_queries
                arm_name = ARM_NAMES.get(arm_index, f"arm_{arm_index}")
                
                proportions[arm_name] = proportion
                
                theoretical_latency += proportion * HOT_LATENCIES.get(arm_index, 0)
            
            logger.info(f"[{bandit}] Final Policy Proportions: {proportions}")
            logger.info(f"[{bandit}] Actual Avg Latency (Training): {np.mean(all_latencies):.4f} ms")
            logger.info(f"[{bandit}] Theoretical Avg Latency (Inference): {theoretical_latency:.4f} ms")
            
            # Calculate moving average of rewards
            moving_avg_rewards = moving_average(rewards, MOVING_AVG_WINDOW)

            plt.plot(moving_avg_rewards, label=str(bandit))

            result_dict = {
                "bandit": str(bandit),
                "rewards": list(rewards),
                "moving_avg_rewards": list(moving_avg_rewards),
                "accuracies": list(all_accuracies),
                "avg_accuracy": float(np.mean(all_accuracies)),
                "latencies": list(all_latencies),
                "total_latency": float(np.sum(all_latencies)),
                "actual_avg_latency": float(np.mean(all_latencies)),
                "theoretical_avg_latency": float(theoretical_latency),
                "selected_arms": [int(arm) for arm in selected_arms], # Convert NumPy int to native int
                "policy_proportions": proportions,
                "params": bandit.__dict__ # Save bandit internal parameters
            }

            # Convert any NumPy arrays inside params to lists
            for key, value in result_dict.get("params", {}).items():
                if isinstance(value, np.ndarray):
                    result_dict["params"][key] = value.tolist()

            # Convert np.float64 in proportions dictionary to native float
            for key, value in result_dict["policy_proportions"].items():
                if isinstance(value, np.float64):
                    result_dict["policy_proportions"][key] = float(value)

            save_path = os.path.join(SAVE_DIR, f"{str(bandit).replace(' ', '_')}_results_{timestamp}.json")

            with open(save_path, "w") as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"Results saved to {save_path}")

        plt.title("Average Rewards of Multi-Armed Bandits")
        plt.xlabel("Queries")
        plt.ylabel("Average Reward")
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(SAVE_DIR, f"mab_average_rewards_{timestamp}.png"),
            dpi=300,
            bbox_inches="tight"
        )
    
    finally:
        # Close Weaviate client
        client.close()