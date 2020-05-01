from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef as metric

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    score = metric(y_true, y_pred)
    return score


def get_scorer(**kwargs):
    return make_scorer(metric, greater_is_better=True)


def get_config(**kwargs):
    config_string = "Matthews correlation coefficient (higher is better)"
    return config_string
