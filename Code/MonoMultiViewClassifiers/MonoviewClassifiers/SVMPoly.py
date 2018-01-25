from sklearn.svm import SVC
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
    classifier = SVC(C=kwargs['C'], kernel='poly', degree=kwargs["degree"], probability=True, max_iter=1000, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"C": randomState.randint(1, 10000), "degree": randomState.randint(1, 30)})
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"C": args.SVMPoly_C, "degree": args.SVMPoly_deg}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', SVC(kernel="poly", max_iter=1000))])


def genParamsDict(randomState):
    return {"classifier__C": np.arange(1, 10000),
                     "classifier__degree": np.arange(1, 30)}


def genBestParams(detector):
    return {"C": detector.best_params_["classifier__C"],
                   "degree": detector.best_params_["classifier__degree"]}


def genParamsFromDetector(detector):
    return [("c", np.array(detector.cv_results_['param_classifier__C'])),
              ("degree", np.array(detector.cv_results_['param_classifier__degree']))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SVM Poly with C : " + str(config.C) + ", degree : " + str(config.degree)
    else:
        return "\n\t\t- SVM Poly with C : " + str(config["C"]) + ", degree : " + str(config["degree"])

def getInterpret(classifier, directory):
    return ""
