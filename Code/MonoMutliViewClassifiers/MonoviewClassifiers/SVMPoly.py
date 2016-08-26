from sklearn.svm import SVC
from sklearn.pipeline import Pipeline                   # Pipelining in classification
from sklearn.grid_search import GridSearchCV
import numpy as np


def fit(DATASET, CLASS_LABELS, NB_CORES=1,**kwargs):
    C = int(kwargs['0'])
    degree = int(kwargs['1'])
    classifier = SVC(C=C, kernel='poly', degree=degree, probability=True)
    classifier.fit(DATASET, CLASS_LABELS)
    return "No desc", classifier


def fit_gridsearch(X_train, y_train, nbFolds=4, nbCores=1, **kwargs):
    pipeline_SVMPoly = Pipeline([('classifier', SVC(kernel="poly"))])
    param_SVMPoly= {"classifier__C": np.random.randint(1,2000,30), "classifier__degree": np.random.randint(1,10,5)}
    grid_SVMPoly = GridSearchCV(pipeline_SVMPoly, param_grid=param_SVMPoly, refit=True, n_jobs=nbCores, scoring='accuracy',
                                  cv=nbFolds)
    SVMPoly_detector = grid_SVMPoly.fit(X_train, y_train)
    desc_params = [SVMPoly_detector.best_params_["classifier__C"], SVMPoly_detector.best_params_["classifier__degree"]]
    return desc_params




def getConfig(config):
    return "\n\t\t- SVM with C : "+config[0]+", kernel : "+config[1]