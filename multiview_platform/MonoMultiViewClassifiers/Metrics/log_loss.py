from sklearn.metrics import log_loss as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except Exception:
        eps = 1e-15
    score = metric(y_true, y_pred, sample_weight=sample_weight, eps=eps)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except Exception:
        eps = 1e-15
    return make_scorer(metric, greater_is_better=False,
                       sample_weight=sample_weight, eps=eps)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except Exception:
        eps = 1e-15
    config_string = "Log loss using " + str(
        sample_weight) + " as sample_weights, " + str(
        eps) + " as eps (lower is better)"
    return config_string
