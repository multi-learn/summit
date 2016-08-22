from sklearn.neighbors import KNeighborsClassifier


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    nNeighbors = int(kwargs['0'])
    classifier = KNeighborsClassifier(n_neighbors=nNeighbors)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier

def getConfig(config):
    return "\n\t\t- K nearest Neighbors with  n_neighbors: "+config[0]