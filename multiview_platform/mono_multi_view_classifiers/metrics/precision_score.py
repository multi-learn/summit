from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score as metric
import warnings
warnings.warn("the precision_score module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

def score(y_true, y_pred, average='micro', multiclass=False, **kwargs):
    score = metric(y_true, y_pred, average=average, **kwargs)
    return score


def get_scorer(average='micro', **kwargs):
    return make_scorer(metric, greater_is_better=True,
                       average=average, **kwargs)


def get_config(average='micro', **kwargs):
    config_string = "Precision score using average: {}, {} (higher is better)".format(average, kwargs)
    return config_string
