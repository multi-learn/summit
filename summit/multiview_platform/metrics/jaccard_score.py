from sklearn.metrics import jaccard_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    score = metric(y_true, y_pred, **kwargs)
    return score


def get_scorer(**kwargs):
    return make_scorer(metric, greater_is_better=True,
                       **kwargs)


def get_config(**kwargs):
    config_string = "Jaccard_similarity score using {} (higher is better)".format(
        kwargs)
    return config_string
