from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import GridSearchCV
import numpy as np

def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    nNeighbors = int(kwargs['0'])
    classifier = KNeighborsClassifier(n_neighbors=nNeighbors)
    classifier.fit(DATASET, CLASS_LABELS)
    return "No desc", classifier


def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
    pipeline_KNN = Pipeline([('classifier', KNeighborsClassifier())])
    param_KNN = {"classifier__n_neighbors": map(int, kwargs['0'])}
    grid_KNN = GridSearchCV(pipeline_KNN, param_grid=param_KNN, refit=True, n_jobs=nbCores, scoring='accuracy',
                            cv=nbFolds)
    KNN_detector = grid_KNN.fit(X_train, y_train)
    desc_params = [KNN_detector.best_params_["classifier__n_neighbors"]]
    description = "Classif_" + "Lasso" + "-" + "CV_" + str(nbFolds) + "-" + "-".join(map(str,desc_params))
    return description, KNN_detector


def gridSearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
    pipeline_KNN = Pipeline([('classifier', KNeighborsClassifier())])
    param_KNN = {"classifier__n_neighbors": np.random.randint(1, 30, 10)}
    grid_KNN = GridSearchCV(pipeline_KNN, param_grid=param_KNN, refit=True, n_jobs=nbCores, scoring='accuracy',
                            cv=nbFolds)
    KNN_detector = grid_KNN.fit(X_train, y_train)
    desc_params = [KNN_detector.best_params_["classifier__n_neighbors"]]
    return desc_params


def getConfig(config):
    return "\n\t\t- K nearest Neighbors with  n_neighbors: "+config[0]