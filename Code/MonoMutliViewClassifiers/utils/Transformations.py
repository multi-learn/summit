import numpy as np

def signLabels(labels):
    if set(labels) == (0,1):
        return np.array([label if label != 0 else -1 for label in labels])
    else:
        return labels
