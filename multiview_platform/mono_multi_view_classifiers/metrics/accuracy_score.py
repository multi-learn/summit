"""Functions :
 score: to get the accuracy score
 get_scorer: returns a sklearn scorer for grid search
"""

from sklearn.metrics import accuracy_score as metric
from sklearn.metrics import make_scorer
import warnings

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
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    score = metric(y_true, y_pred, sample_weight=sample_weight)
    return score


def get_scorer(**kwargs):
    """Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    A weighted sklearn scorer for accuracy"""
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    return make_scorer(metric, greater_is_better=True,
                       sample_weight=sample_weight)


def get_config(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    config_string = "Accuracy score using " + str(
        sample_weight) + " as sample_weights (higher is better)"
    return config_string
