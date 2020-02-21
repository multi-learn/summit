import warnings

from sklearn.metrics import make_scorer
from sklearn.metrics import zero_one_loss as metric

warnings.warn("the zero_one_loss module  is deprecated", DeprecationWarning,
              stacklevel=2)

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    score = metric(y_true, y_pred, **kwargs)
    return score


def get_scorer(**kwargs):
    return make_scorer(metric, greater_is_better=False,
                       **kwargs)


def get_config(**kwargs):
    configString = "Zero_one loss using {} (lower is better)".format(kwargs)
    return configString
