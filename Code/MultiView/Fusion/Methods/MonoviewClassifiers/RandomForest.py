from sklearn.ensemble import RandomForestClassifier


def train(DATASET, CLASS_LABELS, monoviewClassifierConfig):
    num_estimators, maxDepth, NB_CORES = map(int, monoviewClassifierConfig)
    classifier = RandomForestClassifier(n_estimators=num_estimators, max_depth=maxDepth, n_jobs=NB_CORES)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def predict(DATASET, classifier):
    predictedLabels = classifier.predict(DATASET)
    return predictedLabels