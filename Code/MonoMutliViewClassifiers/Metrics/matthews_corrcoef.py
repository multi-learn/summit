from sklearn.metrics import matthews_corrcoef as metric
from sklearn.metrics import make_scorer


def score(y_true, y_pred, **kwargs):
    score = metric(y_true, y_pred)
    return score


def get_scorer(**kwargs):
    return make_scorer(metric, greater_is_better=True)