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
    classifier = SVC(C=kwargs['C'], kernel='rbf', probability=True, max_iter=1000, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 10000), ])
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"C": args.SVMRBF_C}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', SVC(kernel="rbf", max_iter=1000))])


def genParamsDict(randomState):
    return {"classifier__C": np.arange(1, 10000)}


def genBestParams(detector):
    return {'C': detector.best_params_["classifier__C"]}


def genParamsFromDetector(detector):
    nIter = len(detector.cv_results_['param_classifier__C'])
    return [("c", np.array(detector.cv_results_['param_classifier__C'])),
              ("control", np.array(["control" for _ in range(nIter)]))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SVM RBF with C : " + str(config.C)
    else:
        return "\n\t\t- SVM RBF with C : " + str(config["C"])


def getInterpret(classifier, directory):
    return ""