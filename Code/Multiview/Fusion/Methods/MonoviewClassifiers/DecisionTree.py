from sklearn.tree import DecisionTreeClassifier


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    maxDepth = int(kwargs['0'])
    classifier = DecisionTreeClassifier(max_depth=maxDepth)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier

def getConfig(config):
    return "\n\t\t- Decision Tree with max_depth : "+config[0]