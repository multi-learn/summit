from sklearn.metrics import hamming_loss as metric


def score(y_true, y_pred, **kwargs):
    try:
        classes = kwargs["0"]
    except:
        classes=None
    score = metric(y_true, y_pred, classes=classes)
    return score
