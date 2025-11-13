"""
Contains the core experiment running logic.
"""
import numpy as np
import time
import json
import os
from config import SEARCH_LIMIT, RESULTS_DIR
from environment import RetrievalEnvironment
from mab_algorithms import BaseBandit
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def run_experiment(
    bandit: BaseBandit,
    env: RetrievalEnvironment,
    query_ids: list,
    metric_fn: Callable,
    lambda_param: float,
    run_name: str
    ):
    """
    Run MAB training for a given bandit and environment.
    
    Parameters:
    - bandit: The MAB algorithm instance.
    - env: The RetrievalEnvironment instance.
    - query_ids: List of query IDs to iterate through.
    - metric_fn: Function to calculate accuracy (e.g., recall_at_k).
    - lambda_param: Trade-off parameter for reward (score vs. latency).
    - run_name: A unique name for this experiment run for saving.
    """
    logger.info(f"Starting experiment: {run_name} ...")
    start_time = time.time()
    
    rewards = []
    all_latencies = []
    all_scores = [] # "score" is the raw metric (e.g., recall)
    selected_arms = []
    
    # Keep track of min/max latency for stable O(1) normalization
    min_latency = float('inf')
    max_latency = float('-inf')

    for i, qid in enumerate(query_ids):
        # 1. Select arm
        arm_index = bandit.select_arm()
        selected_arms.append(arm_index)
        
        # 2. Pull arm (get results and latency)
        retrieved_pids, latency = env.pull_arm(arm_index, qid)
        relevant_pids = env.qrels.get(qid, set())

        # 3. Calculate score (accuracy)
        score = metric_fn(retrieved_pids, relevant_pids, k=SEARCH_LIMIT)
        
        all_latencies.append(latency)
        all_scores.append(score)

        # 4. Calculate reward
        # Update running min/max for normalization
        min_latency = min(min_latency, latency)
        max_latency = max(max_latency, latency)
        latency_range = max(max_latency - min_latency, 1e-6) # Avoid division by zero

        # Normalize cost (latency)
        norm_cost = (latency - min_latency) / latency_range
        # Reward = score (e.g., recall) - penalty * normalized_cost
        reward = score - lambda_param * norm_cost
        
        # 5. Update bandit
        bandit.update(arm_index, reward)
        
        rewards.append(reward)

        if (i + 1) % 1000 == 0:
            logger.info(
                f"[{run_name}] Step: {i+1}/{len(query_ids)} | "
                f"Arm: {env.arm_config[arm_index]['name']} | "
                f"Score: {score:.4f} | "
                f"Latency: {latency:.2f}ms | "
                f"Reward: {reward:.4f}"
            )

    end_time = time.time()
    logger.info(f"Experiment {run_name} finished in {end_time - start_time:.2f} seconds.")
    
    # --- Collect and Save Results ---
    avg_score = float(np.mean(all_scores))
    avg_latency = float(np.mean(all_latencies))
    
    logger.info(f"[{run_name}] Avg Score: {avg_score:.4f} | Avg Latency: {avg_latency:.4f} ms")

    # Calculate final policy proportions
    arm_counts = np.bincount(selected_arms, minlength=env.k_arms)
    policy_proportions = {
        env.arm_config[i]['name']: count / len(selected_arms)
        for i, count in enumerate(arm_counts)
    }
    logger.info(f"[{run_name}] Final Policy: {policy_proportions}")

    result_data = {
        "run_name": run_name,
        "bandit": str(bandit),
        "params": bandit.__dict__,
        "lambda_param": lambda_param,
        "total_queries": len(query_ids),
        "avg_score": avg_score,
        "avg_latency": avg_latency,
        "policy_proportions": policy_proportions,
        "rewards": rewards,
        "scores": all_scores,
        "latencies": all_latencies,
        "selected_arms": [int(a) for a in selected_arms], # For JSON
        "arm_map": {i: arm['name'] for i, arm in enumerate(env.arm_config)}
    }

    # Clean up params for JSON serialization
    if 'rng' in result_data['params']:
        del result_data['params']['rng'] # Don't save the generator
    for key, value in result_data["params"].items():
        if isinstance(value, np.ndarray):
            result_data["params"][key] = value.tolist()

    save_path = os.path.join(RESULTS_DIR, f"{run_name}.json")
    try:
        with open(save_path, "w") as f:
            json.dump(result_data, f, indent=2)
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results for {run_name}: {e}")

    return result_data