from sklearn.metrics import hamming_loss as metric
from sklearn.metrics import make_scorer


def score(y_true, y_pred, **kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes=None
    score = metric(y_true, y_pred, classes=classes)
    return score


def get_scorer(**kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes=None
    return make_scorer(metric, greater_is_better=False, classes=classes)
