from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np
# import cPickle

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps
from ..utils.Interpret import getFeatureImportance

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    classifier = RandomForestClassifier(n_estimators=kwargs['n_estimators'],
                                        max_depth=kwargs['max_depth'],
                                        criterion=kwargs['criterion'],
                                        n_jobs=NB_CORES, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": randomState.randint(1, 300),
                          "max_depth": randomState.randint(1, 300),
                          "criterion": randomState.choice(["gini", "entropy"])})
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"n_estimators": args.RF_trees,
                  "max_depth": args.RF_max_depth,
                  "criterion": args.RF_criterion}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', RandomForestClassifier())])


def genParamsDict(randomState):
    return {"classifier__n_estimators": np.arange(1, 300),
                "classifier__max_depth": np.arange(1, 300),
                "classifier__criterion": ["gini", "entropy"]}


def genBestParams(detector):
    return {"n_estimators": detector.best_params_["classifier__n_estimators"],
            "max_depth": detector.best_params_["classifier__max_depth"],
            "criterion": detector.best_params_["classifier__criterion"]}


def genParamsFromDetector(detector):
    return [("nEstimators", np.array(detector.cv_results_['param_classifier__n_estimators'])),
              ("maxDepth", np.array(detector.cv_results_['param_classifier__max_depth'])),
              ("criterion", np.array(detector.cv_results_['param_classifier__criterion']))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- Random Forest with num_esimators : " + str(config.n_estimators) + ", max_depth : " + str(
            config.max_depth) + ", criterion : " + config.criterion
    else:
        return "\n\t\t- Random Forest with num_esimators : " + str(config["n_estimators"]) + \
               ", max_depth : " + str(config["max_depth"]) + ", criterion : " + config["criterion"]


def getInterpret(classifier, directory):
    interpretString = getFeatureImportance(classifier, directory)
    return interpretString
