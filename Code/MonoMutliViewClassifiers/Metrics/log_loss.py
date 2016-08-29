from sklearn.metrics import log_loss as metric
from sklearn.metrics import make_scorer


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except:
        eps = 1e-15
    score = metric(y_true, y_pred, sample_weight=sample_weight, eps=eps)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except:
        eps = 1e-15
    return make_scorer(metric, greater_is_better=False, sample_weight=sample_weight, eps=eps)
