from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline  # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import numpy as np

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    classifier = SGDClassifier(loss=kwargs['loss'],
                               penalty=kwargs['penalty'],
                               alpha=kwargs['alpha'],
                               random_state=randomState, n_jobs=NB_CORES)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"loss": randomState.choice(['log', 'modified_huber']),
                          "penalty": randomState.choice(["l1", "l2", "elasticnet"]),
                          "alpha": randomState.random_sample()})
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"loss": args.SGD_loss,
                  "penalty": args.SGD_penalty,
                  "alpha": args.SGD_alpha}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', SGDClassifier())])


def genParamsDict(randomState):
    losses = ['log', 'modified_huber']
    penalties = ["l1", "l2", "elasticnet"]
    alphas = uniform()
    return {"classifier__loss": losses, "classifier__penalty": penalties,
                 "classifier__alpha": alphas}


def genBestParams(detector):
    return {"loss": detector.best_params_["classifier__loss"],
                  "penalty": detector.best_params_["classifier__penalty"],
                  "alpha": detector.best_params_["classifier__alpha"]}


def genParamsFromDetector(detector):
    return [("loss", np.array(detector.cv_results_['param_classifier__loss'])),
              ("penalty", np.array(detector.cv_results_['param_classifier__penalty'])),
              ("aplha", np.array(detector.cv_results_['param_classifier__alpha']))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SGDClassifier with loss : " + config.loss + ", penalty : " + \
               config.penalty + ", alpha : " + str(config.alpha)
    else:
        return "\n\t\t- SGDClassifier with loss : " + config["loss"] + ", penalty : " + \
               config["penalty"] + ", alpha : " + str(config["alpha"])

def getInterpret(classifier, directory):
    # TODO : coeffs
    return ""
# 