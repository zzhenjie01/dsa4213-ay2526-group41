import numpy as np

def moving_average(data, window_size):
    """Calculates the moving average of a list."""
    if len(data) == 0:
        return []
    # Use convolution to efficiently calculate the moving average
    return np.convolve(data, np.ones(window_size), 'valid') / window_size