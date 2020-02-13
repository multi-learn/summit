from sklearn.metrics import hamming_loss as metric
from sklearn.metrics import make_scorer
import warnings

warnings.warn("the hamming_loss module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    score = metric(y_true, y_pred, **kwargs)
    return score


def get_scorer(**kwargs):
    return make_scorer(metric, greater_is_better=False, **kwargs)


def get_config(**kwargs):
    config_string = "Hamming loss using {} (lower is better)".format(kwargs)
    return config_string
