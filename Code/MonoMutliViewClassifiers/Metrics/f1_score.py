from sklearn.metrics import f1_score as metric


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
        average = "binary"
    score = metric(y_true, y_pred, sample_weight=sample_weight, labels=labels, pos_label=pos_label, average=average)
    return score
