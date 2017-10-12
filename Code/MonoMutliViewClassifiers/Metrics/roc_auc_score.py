from sklearn.metrics import roc_auc_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        average = kwargs["1"]
    except:
        average = "micro"
    score = metric(y_true, y_pred, sample_weight=sample_weight, average=average)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        average = kwargs["1"]
    except:
        average = "micro"
    return make_scorer(metric, greater_is_better=True, sample_weight=sample_weight, average=average)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        average = kwargs["3"]
    except:
        average = "micro"
    configString = "ROC AUC score using " + str(
        sample_weight) + " as sample_weights, " + average + " as average (higher is better)"
    return configString
