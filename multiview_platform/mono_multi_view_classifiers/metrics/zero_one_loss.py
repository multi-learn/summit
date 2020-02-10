from sklearn.metrics import make_scorer
from sklearn.metrics import zero_one_loss as metric
import warnings

warnings.warn("the zero_one_loss module  is deprecated", DeprecationWarning,
              stacklevel=2)

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    score = metric(y_true, y_pred, sample_weight=sample_weight)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    return make_scorer(metric, greater_is_better=False,
                       sample_weight=sample_weight)


def get_config(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    configString = "Zero_one loss using " + str(
        sample_weight) + " as sample_weights (lower is better)"
    return configString
