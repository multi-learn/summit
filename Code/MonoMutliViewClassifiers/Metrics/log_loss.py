from sklearn.metrics import log_loss as metric


def score(y_true, y_pred, **kwargs):
    try:
        sample_weight = kwargs["0"]
    except:
        sample_weight = None
    try:
        eps = kwargs["1"]
    except:
        eps = 1e-15
    score = metric(y_true, y_pred, sample_weight=sample_weight, eps=eps)
    return score
