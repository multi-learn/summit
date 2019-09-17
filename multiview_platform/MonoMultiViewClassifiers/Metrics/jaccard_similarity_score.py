from sklearn.metrics import jaccard_similarity_score as metric
from sklearn.metrics import make_scorer
import warnings

warnings.warn("the jaccard_similarity_score module  is deprecated", DeprecationWarning,
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
    return make_scorer(metric, greater_is_better=True,
                       sample_weight=sample_weight)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    config_string = "Jaccard_similarity score using " + str(
        sample_weight) + " as sample_weights (higher is better)"
    return config_string
