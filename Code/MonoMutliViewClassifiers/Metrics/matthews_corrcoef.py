from sklearn.metrics import matthews_corrcoef as metric


def score(y_true, y_pred, **kwargs):
    score = metric(y_true, y_pred)
    return score