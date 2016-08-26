from sklearn.metrics import roc_auc_score as metric


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    try:
        average = kwargs["1"]
    except:
        average = "binary"
    score = metric(y_true, y_pred, sample_weight=sample_weight, average=average)
    return score
