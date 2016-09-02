"""Functions :
 score: to get the accuracy score
 get_scorer: returns a sklearn scorer for grid search
"""

from sklearn.metrics import accuracy_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def score(y_true, y_pred, **kwargs):
    """Arguments:
    y_true: real labels
    y_pred predicted labels

    Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    Weighted accuracy score for y_true, y_pred"""
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    score = metric(y_true, y_pred, sample_weight=sample_weight)
    return score


def get_scorer(**kwargs):
    """Keyword Arguments:
    "0": weights to compute accuracy

    Returns:
    A weighted sklearn scorer for accuracy"""
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    return make_scorer(metric, greater_is_better=True, sample_weight=sample_weight)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    configString = "Accuracy score using "+str(sample_weight)+" as sample_weights (higher is better)"
    return configString