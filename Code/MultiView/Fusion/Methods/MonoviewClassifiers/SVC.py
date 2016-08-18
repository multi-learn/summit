from sklearn.svm import SVC


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    C = int(kwargs['0'])
    kernel = kwargs['1']
    classifier = SVC(C=C, kernel=kernel, probability=True)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def getConfig(config):
    return "\n\t\t- SVM with C : "+config[0]+", kernel : "+config[1]