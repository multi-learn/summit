from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
    num_estimators = int(kwargs['0'])
    maxDepth = int(kwargs['1'])
    criterion = kwargs["2"]
    classifier = RandomForestClassifier(n_estimators=num_estimators, max_depth=maxDepth, criterion=criterion,
                                        n_jobs=NB_CORES, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 300), randomState.randint(1, 300),
                          randomState.choice(["gini", "entropy"])])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_RandomForest_trees":
            kwargsDict['0'] = int(kwargValue)
        elif kwargName == "CL_RandomForest_max_depth":
            kwargsDict['1'] = kwargValue
        elif kwargName == "CL_RandomForest_criterion":
            kwargsDict['2'] = kwargValue
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, nbCores=1,
                     metric=["accuracy_score", None], nIter=30):
    pipeline_rf = Pipeline([('classifier', RandomForestClassifier())])
    param_rf = {"classifier__n_estimators": randint(1, 300),
                "classifier__max_depth": randint(1, 300),
                "classifier__criterion": ["gini", "entropy"]}
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_rf = RandomizedSearchCV(pipeline_rf, n_iter=nIter, param_distributions=param_rf, refit=True, n_jobs=nbCores,
                                 scoring=scorer, cv=KFolds, random_state=randomState)
    rf_detector = grid_rf.fit(X_train, y_train)

    desc_estimators = [rf_detector.best_params_["classifier__n_estimators"],
                       rf_detector.best_params_["classifier__max_depth"],
                       rf_detector.best_params_["classifier__criterion"]]

    scoresArray = rf_detector.cv_results_['mean_test_score']
    params = [("nEstimators", np.array(rf_detector.cv_results_['param_classifier__n_estimators'])),
              ("maxDepth", np.array(rf_detector.cv_results_['param_classifier__max_depth'])),
              ("criterion", np.array(rf_detector.cv_results_['param_classifier__criterion']))]

    genHeatMaps(params, scoresArray, outputFileName)
    return desc_estimators


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- Random Forest with num_esimators : " + str(config.n_estimators) + ", max_depth : " + str(
            config.max_depth) + ", criterion : " + config.criterion
    else:
        try:
            return "\n\t\t- Random Forest with num_esimators : " + str(config[0]) + ", max_depth : " + str(
                config[1]) + ", criterion : " + config[2]
        except:
            return "\n\t\t- Random Forest with num_esimators : " + str(config["0"]) + ", max_depth : " + str(
                config["1"]) + ", criterion : " + config["2"]


def getInterpret(classifier, directory):
    pass
