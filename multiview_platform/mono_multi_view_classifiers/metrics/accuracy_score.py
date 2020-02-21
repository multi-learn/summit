"""Functions :
 score: to get the accuracy score
 get_scorer: returns a sklearn scorer for grid search
"""

import warnings

from sklearn.metrics import accuracy_score as metric
from sklearn.metrics import make_scorer

warnings.warn("the accuracy_score module  is deprecated", DeprecationWarning,
              stacklevel=2)

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    """Arguments:
    y_true: real labels
    y_pred: predicted labels

    Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    Weighted accuracy score for y_true, y_pred"""
    score = metric(y_true, y_pred, **kwargs)
    return score


def get_scorer(**kwargs):
    """Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    A weighted sklearn scorer for accuracy"""
    return make_scorer(metric, greater_is_better=True,
                       **kwargs)


def get_config(**kwargs):
    config_string = "Accuracy score using {}, (higher is better)".format(kwargs)
    return config_string
