from sklearn.metrics import fbeta_score as metric
from sklearn.metrics import make_scorer
import warnings

warnings.warn("the fbeta_score module is deprecated", DeprecationWarning,
              stacklevel=2)

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, beta=2.0, average="micro", **kwargs):
    score = metric(y_true, y_pred, beta=beta, average=average, **kwargs)
    return score


def get_scorer(beta=2.0, average="micro", **kwargs):
    return make_scorer(metric, greater_is_better=True, beta=beta,
                       average=average, **kwargs)


def get_config(beta=2.0, average="micro", **kwargs):
    config_string = "F-beta score using beta: {}, average: {}, {} (higher is better)".format(beta, average, kwargs)
    return config_string
