import numpy as np


def binary_entropy(probabilities):
    p = np.clip(np.asarray(probabilities), 1e-8, 1 - 1e-8)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def probability_uncertainty(probabilities):
    """Decision uncertainty; not rigorous epistemic uncertainty."""
    p = np.asarray(probabilities)
    return 1 - 2 * np.abs(p - 0.5)
