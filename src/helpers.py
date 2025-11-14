"""
Utility functions for the MAB project.
"""
import numpy as np
import logging
import time
import os
import jsonlines
from config import LOG_DIR

def moving_average(data: list, window_size: int, mode: str ='valid') -> list:
    """
    Calculates the moving average of a list.
    
    Args:
        data: Input list or array
        window_size: Size of the moving window
        mode: 'valid', 'same', or 'full' (convolution mode)
    
    Returns:
        Moving average list
    """
    if len(data) == 0:
        return []
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    if window_size > len(data) and mode == 'valid':
        raise ValueError("Window size cannot be larger than data length for 'valid' mode")
    
    # Convert to numpy array for consistency
    data_array = np.array(data)
    result = np.convolve(data_array, np.ones(window_size)/window_size, mode=mode)

    return result.tolist()


def setup_logging(script_name: str) -> logging.Logger:
    """Configures and returns a logger."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f"{script_name}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(script_name)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger


def load_queries_and_qrels(queries_with_qrels_path: str) -> tuple[dict, dict]:
    """Loads queries and qrels from a jsonl file."""
    queries, qrels = {}, {}
    try:
        with jsonlines.open(queries_with_qrels_path) as reader:
            for obj in reader:
                queries[obj["qid"]] = obj["query"]
                qrels[obj["qid"]] = set(obj["qrels"])
    except FileNotFoundError:
        logging.error(f"Data file not found at: {queries_with_qrels_path}")
        raise
    return queries, qrels