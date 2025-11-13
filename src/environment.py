"""
Contains the RetrievalEnvironment class that interacts with Weaviate.
"""
import time
from config import SEARCH_LIMIT
import logging

logger = logging.getLogger(__name__)

class RetrievalEnvironment:
    """
    Handles all interactions with the Weaviate retrieval backend.
    """
    def __init__(self, client, collection_name, model, queries, qrels, arm_config):
        self.client = client
        self.collection = client.collections.get(collection_name)
        self.model = model
        self.queries = queries
        self.qrels = qrels
        self.arm_config = arm_config
        self.k_arms = len(arm_config)
        logger.info(f"RetrievalEnvironment initialized with {self.k_arms} arms.")
        for i, arm in enumerate(self.arm_config):
            logger.info(f"  Arm {i}: {arm['name']} (type: {arm['type']})")

    def pull_arm(self, arm_index, query_id):
        """
        Perform retrieval using the selected arm's method and parameters.
        
        Returns:
        - retrieved_pids (set[str]): The set of retrieved document IDs.
        - latency_ms (float): The time taken for the retrieval in milliseconds.
        """
        query_text = self.queries[query_id]
        arm = self.arm_config[arm_index]
        method_type = arm["type"]
        method_params = arm["params"]
        
        time_taken = 0.0
        response = None
        
        try:
            # --- Perform retrieval based on the arm type ---
            if method_type == "bm25":
                start_time = time.perf_counter()
                response = self.collection.query.bm25(
                    query=query_text,
                    limit=SEARCH_LIMIT,
                    return_properties=["pid"],
                )
                time_taken = time.perf_counter() - start_time
            
            elif method_type == "dense":
                start_time = time.perf_counter()
                query_embedding = self.model.encode(query_text, show_progress_bar=False).tolist()
                response = self.collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=SEARCH_LIMIT,
                    return_properties=["pid"],
                    **method_params # (e.g., distance)
                )
                time_taken = time.perf_counter() - start_time

            elif method_type == "hybrid":
                start_time = time.perf_counter()
                query_embedding = self.model.encode(query_text, show_progress_bar=False).tolist()
                response = self.collection.query.hybrid(
                    query=query_text,
                    vector=query_embedding,
                    limit=SEARCH_LIMIT,
                    return_properties=["pid"],
                    **method_params # e.g., alpha
                )
                time_taken = time.perf_counter() - start_time
            
            else:
                raise ValueError(f"Unknown retrieval method type: {method_type}")

            latency_ms = time_taken * 1000

            # --- Process response ---
            if not response or not response.objects:
                return set(), latency_ms

            retrieved_pids = {
                obj.properties["pid"] for obj in response.objects
            }
            return retrieved_pids, latency_ms

        except Exception as e:
            logger.error(f"Error pulling arm {arm['name']} for query {query_id}: {e}")
            return set(), 0.0 # Return empty set and 0 latency on failure