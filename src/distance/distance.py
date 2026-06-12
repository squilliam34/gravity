import numpy as np

def sigmoid(vix: np.ndarray, k: int = 0.05, threshold: int=20) -> np.ndarray:
    """
    Apply the sigmoid transformation to the VIX to fit it between 0 and 1 as
    the lambda weight for my distance metric.

    Args:
    vix (np.ndarray): An array of closing VIX values.
    k (int): The tuning parameter. Determines how "steep" the sigmoid is.
    threshold (int): The long run average of the VIX (~20). Used to scale the current value.

    Returns:
    np.ndarray: An array containing the transformed VIX value at each point in time.
    """
    return 1/(1 + np.exp(-k*(vix-threshold)))