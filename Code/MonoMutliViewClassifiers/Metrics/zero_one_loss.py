from sklearn.metrics import zero_one_loss as metric


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight=None
    score = metric(y_true, y_pred, sample_weight=sample_weight)
    return score
