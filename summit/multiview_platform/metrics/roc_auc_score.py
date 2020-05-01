from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score as metric

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
    configString = "ROC_AUC score using {}".format(kwargs)
    return configString
