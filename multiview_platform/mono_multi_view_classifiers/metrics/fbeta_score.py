from sklearn.metrics import fbeta_score as metric
from sklearn.metrics import make_scorer
import warnings

warnings.warn("the fbeta_score module is deprecated", DeprecationWarning,
              stacklevel=2)

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        beta = kwargs["1"]
    except Exception:
        beta = 10.0
    try:
        labels = kwargs["2"]
    except Exception:
        labels = None
    try:
        pos_label = kwargs["3"]
    except Exception:
        pos_label = 1
    try:
        average = kwargs["4"]
    except Exception:
        if multiclass:
            average = "micro"
        else:
            average = "binary"
    score = metric(y_true, y_pred, beta, sample_weight=sample_weight,
                   labels=labels, pos_label=pos_label,
                   average=average)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        beta = kwargs["1"]
    except Exception:
        beta = 1.0
    try:
        labels = kwargs["2"]
    except Exception:
        labels = None
    try:
        pos_label = kwargs["3"]
    except Exception:
        pos_label = 1
    try:
        average = kwargs["4"]
    except Exception:
        average = "binary"
    return make_scorer(metric, greater_is_better=True, beta=beta,
                       sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label, average=average)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except Exception:
        sample_weight = None
    try:
        beta = kwargs["1"]
    except Exception:
        beta = 1.0
    try:
        labels = kwargs["1"]
    except Exception:
        labels = None
    try:
        pos_label = kwargs["2"]
    except Exception:
        pos_label = 1
    try:
        average = kwargs["3"]
    except Exception:
        average = "binary"
    config_string = "F-beta score using " + str(
        sample_weight) + " as sample_weights, " + str(
        labels) + " as labels, " + str(pos_label) \
                   + " as pos_label, " + average + " as average, " + str(
        beta) + " as beta (higher is better)"
    return config_string
