from sklearn.linear_model import SGDClassifier


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    loss = kwargs['0']
    penalty = kwargs['1']
    try:
        alpha = int(kwargs['2'])
    except:
        alpha = 0.15
    classifier = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier

def getConfig(config):
    return "\n\t\t- SGDClassifier with loss : "+config[0]+", penalty : "+config[1]