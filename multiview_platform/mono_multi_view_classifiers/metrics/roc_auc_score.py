import warnings

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score as metric
from sklearn.preprocessing import MultiLabelBinarizer

warnings.warn("the roc_auc_score module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    if multiclass:
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform([(label) for label in y_true])
        y_pred = mlb.fit_transform([(label) for label in y_pred])

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
    return make_scorer(metric, greater_is_better=True,
                       sample_weight=sample_weight, average=average)


def get_config(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        average = kwargs["3"]
    except Exception:
        average = "micro"
    configString = "ROC_AUC score using " + str(
        sample_weight) + " as sample_weights, " + average + " as average (higher is better)"
    return configString
