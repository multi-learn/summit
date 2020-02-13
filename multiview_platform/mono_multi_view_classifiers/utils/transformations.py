import numpy as np
from sklearn.preprocessing import LabelEncoder

def sign_labels(labels):
    """
    Returns a label array with (-1,1) as labels.
    If labels was already made of (-1,1), returns labels.
    If labels is made of (0,1), returns labels with all
    zeros transformed in -1.

    Parameters
    ----------
    labels

    The original label numpy array

    Returns
    -------
    A np.array with labels made of (-1,1)
    """
    if 0 in labels:
        return np.array([label if label != 0 else -1 for label in labels])
    else:
        return labels

def unsign_labels(labels):
    """
    The inverse function

    Parameters
    ----------
    labels

    Returns
    -------

    """
    if len(labels.shape)==2:
        labels = labels.reshape((labels.shape[0], ))
    if -1 in labels:
        return np.array([label if label != -1 else 0 for label in labels])
    else:
        return labels

