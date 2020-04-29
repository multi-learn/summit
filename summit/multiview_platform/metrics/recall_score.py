import warnings

from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score as metric

warnings.warn("the recall_score module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, average='micro', **kwargs):
    score = metric(y_true, y_pred, average=average, **kwargs)
    return score


def get_scorer(average='micro', **kwargs):
    return make_scorer(metric, greater_is_better=True,
                       average=average, **kwargs)


def get_config(average="micro", **kwargs):
    configString = "Recall score using average: {}, {} (higher is better)".format(
        average, kwargs)
    return configString
