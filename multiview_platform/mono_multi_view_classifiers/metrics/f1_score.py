"""Functions :
 score: to get the f1 score
 get_scorer: returns a sklearn scorer for grid search
"""

import warnings

from sklearn.metrics import f1_score as metric
from sklearn.metrics import make_scorer

warnings.warn("the f1_score module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=True, average='micro', **kwargs):
    score = metric(y_true, y_pred, average=average, **kwargs)
    return score


def get_scorer(average="micro", **kwargs):
    return make_scorer(metric, greater_is_better=True, average=average,
                       **kwargs)


def get_config(average="micro", **kwargs, ):
    config_string = "F1 score using average: {}, {} (higher is better)".format(
        average, kwargs)
    return config_string
