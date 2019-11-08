import numpy as np


def sign_labels(labels):
    """Used to transform 0/1 labels to -1/1 labels"""
    if set(labels) == (0, 1):
        return np.array([label if label != 0 else -1 for label in labels])
    else:
        return labels
