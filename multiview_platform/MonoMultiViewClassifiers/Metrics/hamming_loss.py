from sklearn.metrics import hamming_loss as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes = None
    score = metric(y_true, y_pred)
    return score


def get_scorer(**kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes = None
    return make_scorer(metric, greater_is_better=False, classes=classes)


def getConfig(**kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes = None
    configString = "Hamming loss using " + str(classes) + " as classes (lower is better)"
    return configString
