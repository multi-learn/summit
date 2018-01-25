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
    classifier = KNeighborsClassifier(n_neighbors=kwargs["n_neighbors"],
                                      weights=kwargs["weights"],
                                      algorithm=kwargs["algorithm"],
                                      p=kwargs["p"],
                                      n_jobs=NB_CORES, )
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 20), randomState.choice(["uniform", "distance"]),
                          randomState.choice(["auto", "ball_tree", "kd_tree", "brute"]), randomState.choice([1, 2])])
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"n_neighbors": args.KNN_neigh,
                  "weights":args.KNN_weights,
                  "algorithm":args.KNN_algo,
                  "p":args.KNN_p}
    return kwargsDict

def genPipeline():
    return Pipeline([('classifier', KNeighborsClassifier())])


def genParamsDict(randomState):
    return {"classifier__n_neighbors": np.arange(1, 20),
            "classifier__weights": ["uniform", "distance"],
            "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "classifier__p": [1, 2]}


def genBestParams(detector):
    return {"n_neighbors": detector.best_params_["classifier__n_neighbors"],
            "weights": detector.best_params_["classifier__weights"],
            "algorithm": detector.best_params_["classifier__algorithm"],
            "p": detector.best_params_["classifier__p"]}


def genParamsFromDetector(detector):
    return [("nNeighbors", np.array(detector.cv_results_['param_classifier__n_neighbors'])),
              ("weights", np.array(detector.cv_results_['param_classifier__weights'])),
              ("algorithm", np.array(detector.cv_results_['param_classifier__algorithm'])),
              ("p", np.array(detector.cv_results_['param_classifier__p']))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- K nearest Neighbors with  n_neighbors : " + str(config.n_neighbors) + \
               ", weights : " + config.weights + ", algorithm : " + config.algorithm + ", p : " + \
               str(config.p)
    else:
        return "\n\t\t- K nearest Neighbors with  n_neighbors : " + str(config["n_neighbors"]) + \
               ", weights : " + config["weights"] + ", algorithm : " + config["algorithm"] + \
               ", p : " + str(config["p"])

def getInterpret(classifier, directory):
    return ""