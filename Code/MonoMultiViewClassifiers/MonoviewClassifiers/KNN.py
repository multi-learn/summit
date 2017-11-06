from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline  # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    nNeighbors = int(kwargs['0'])
    weights = kwargs["1"]
    algorithm = kwargs["2"]
    p = int(kwargs["3"])
    classifier = KNeighborsClassifier(n_neighbors=nNeighbors, weights=weights, algorithm=algorithm, p=p,
                                      n_jobs=NB_CORES, )
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 50), randomState.choice(["uniform", "distance"]),
                          randomState.choice(["auto", "ball_tree", "kd_tree", "brute"]), randomState.choice([1, 2])])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_KNN_neigh":
            kwargsDict['0'] = int(kwargValue)
        if kwargName == "CL_KNN_weights":
            kwargsDict['1'] = kwargValue
        if kwargName == "CL_KNN_algo":
            kwargsDict['2'] = kwargValue
        if kwargName == "CL_KNN_p":
            kwargsDict['3'] = int(kwargValue)
        else:
            raise ValueError("Wrong arguments served to KNN")
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, nbCores=1,
                     metric=["accuracy_score", None], nIter=30):
    pipeline_KNN = Pipeline([('classifier', KNeighborsClassifier())])
    param_KNN = {"classifier__n_neighbors": randint(1, 50),
                 "classifier__weights": ["uniform", "distance"],
                 "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                 "classifier__p": [1, 2],
                 }
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_KNN = RandomizedSearchCV(pipeline_KNN, n_iter=nIter, param_distributions=param_KNN, refit=True, n_jobs=nbCores,
                                  scoring=scorer,
                                  cv=KFolds, random_state=randomState)
    KNN_detector = grid_KNN.fit(X_train, y_train)
    desc_params = [KNN_detector.best_params_["classifier__n_neighbors"],
                   KNN_detector.best_params_["classifier__weights"],
                   KNN_detector.best_params_["classifier__algorithm"],
                   KNN_detector.best_params_["classifier__p"],
                   ]

    scoresArray = KNN_detector.cv_results_['mean_test_score']
    params = [("nNeighbors", np.array(KNN_detector.cv_results_['param_classifier__n_neighbors'])),
              ("weights", np.array(KNN_detector.cv_results_['param_classifier__weights'])),
              ("algorithm", np.array(KNN_detector.cv_results_['param_classifier__algorithm'])),
              ("p", np.array(KNN_detector.cv_results_['param_classifier__p']))]

    genHeatMaps(params, scoresArray, outputFileName)

    return desc_params


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- K nearest Neighbors with  n_neighbors : " + str(
            config.n_neighbors) + ", weights : " + config.weights + ", algorithm : " + config.algorithm + ", p : " + str(
            config.p)
    else:
        try:
            return "\n\t\t- K nearest Neighbors with  n_neighbors : " + str(config[0]) + ", weights : " + config[
                1] + ", algorithm : " + config[2] + ", p : " + str(config[3])
        except:
            return "\n\t\t- K nearest Neighbors with  n_neighbors : " + str(config["0"]) + ", weights : " + config[
                "1"] + ", algorithm : " + config["2"] + ", p : " + str(config["3"])

def getInterpret(classifier, directory):
    return ""