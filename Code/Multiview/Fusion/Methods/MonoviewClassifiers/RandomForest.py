from sklearn.ensemble import RandomForestClassifier


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    num_estimators = int(kwargs['0'])
    maxDepth = int(kwargs['1'])
    classifier = RandomForestClassifier(n_estimators=num_estimators, max_depth=maxDepth, n_jobs=NB_CORES)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier

def getConfig(config):
    return "\n\t\t- Random Forest with num_esimators : "+config[0]+", max_depth : "+config[1]