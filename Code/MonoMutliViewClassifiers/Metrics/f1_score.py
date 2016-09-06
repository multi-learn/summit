"""Functions :
 score: to get the f1 score
 get_scorer: returns a sklearn scorer for grid search
"""

from sklearn.metrics import f1_score as metric
from sklearn.metrics import make_scorer

# Author-Info
__author__ 	= "Baptiste Bauvin"
__status__ 	= "Prototype"                           # Production, Development, Prototype


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        labels = kwargs["1"]
    except:
        labels=None
    try:
        pos_label = kwargs["2"]
    except:
        pos_label = 1
    try:
        average = kwargs["3"]
    except:
        average = "micro"
    score = metric(y_true, y_pred, sample_weight=sample_weight, labels=labels, pos_label=pos_label, average=average)
    return score


def get_scorer(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        labels = kwargs["1"]
    except:
        labels=None
    try:
        pos_label = kwargs["2"]
    except:
        pos_label = 1
    try:
        average = kwargs["3"]
    except:
        average = "micro"
    return make_scorer(metric, greater_is_better=True, sample_weight=sample_weight, labels=labels,
                       pos_label=pos_label, average=average)


def getConfig(**kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        labels = kwargs["1"]
    except:
        labels=None
    try:
        pos_label = kwargs["2"]
    except:
        pos_label = 1
    try:
        average = kwargs["3"]
    except:
        average = "micro"
    configString = "F1 score using "+str(sample_weight)+" as sample_weights, "+str(labels)+" as labels, "+str(pos_label)\
                   +" as pos_label, "+average+" as average (higher is better)"
    return configString