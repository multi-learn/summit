from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import GridSearchCV


def fit(DATASET, CLASS_LABELS, NB_CORES=1, **kwargs):
    maxDepth = int(kwargs['0'])
    classifier = DecisionTreeClassifier(max_depth=maxDepth)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
    pipeline_DT = Pipeline([('classifier', DecisionTreeClassifier())])
    param_DT = {"classifier__max_depth":map(int, kwargs['0'])}

    grid_DT = GridSearchCV(pipeline_DT, param_grid=param_DT, refit=True, n_jobs=nbCores, scoring='accuracy',
                           cv=nbFolds)
    DT_detector = grid_DT.fit(X_train, y_train)
    desc_params = [DT_detector.best_params_["classifier__max_depth"]]
    description = "Classif_" + "DT" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str,desc_params))
    return description, DT_detector


def getConfig(config):
    return "\n\t\t- Decision Tree with max_depth : "+config[0]