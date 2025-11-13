"""
Defines evaluation metrics for retrieval.

All metric functions follow the same signature:
(retrieved_pids: set[str], relevant_pids: set[str], k: int) -> float
"""

def recall_at_k(retrieved_pids: set, relevant_pids: set, **kwargs) -> float:
    """
    Calculates Recall@k.
    Fraction of relevant documents that were retrieved.
    """
    if not relevant_pids:
        return 0.0  # Cannot calculate recall if there are no relevant documents
    
    correct_retrievals = len(retrieved_pids & relevant_pids)
    return correct_retrievals / len(relevant_pids)

def precision_at_k(retrieved_pids: set, relevant_pids: set, **kwargs) -> float:
    """
    Calculates Precision@k.
    Fraction of retrieved documents that are relevant.
    
    'k' is passed via kwargs to match the signature but is implied
    by the length of retrieved_pids.
    """
    k = len(retrieved_pids)
    if k == 0:
        return 0.0 # Cannot calculate precision if nothing was retrieved
        
    correct_retrievals = len(retrieved_pids & relevant_pids)
    return correct_retrievals / k

def f1_at_k(retrieved_pids: set, relevant_pids: set, **kwargs) -> float:
    """
    Calculates F1-Score@k.
    The harmonic mean of precision and recall.
    """
    precision = precision_at_k(retrieved_pids, relevant_pids, **kwargs)
    recall = recall_at_k(retrieved_pids, relevant_pids, **kwargs)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * (precision * recall) / (precision + recall)


# --- Metric Mapping ---
# Convenience dictionary to select metrics by name
METRIC_FUNCTIONS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "f1": f1_at_k,
}

def get_metric_fn(metric_name: str):
    """Retrieves a metric function by its name."""
    metric_fn = METRIC_FUNCTIONS.get(metric_name.lower())
    if metric_fn is None:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_FUNCTIONS.keys())}")
    return metric_fn