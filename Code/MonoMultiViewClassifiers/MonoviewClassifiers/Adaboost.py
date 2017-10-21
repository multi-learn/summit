from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import numpy as np
# import cPickle
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps
from ..utils.Interpret import getFeatureImportance

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    num_estimators = int(kwargs['0'])
    base_estimators = DecisionTreeClassifier()  # kwargs['1']
    classifier = AdaBoostClassifier(n_estimators=num_estimators, base_estimator=base_estimators,
                                    random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 15), DecisionTreeClassifier()])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_Adaboost_n_est":
            kwargsDict['0'] = int(kwargValue)
        elif kwargName == "CL_Adaboost_b_est":
            kwargsDict['1'] = kwargValue
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, metric=["accuracy_score", None], nIter=30,
                     nbCores=1):
    pipeline = Pipeline([('classifier', AdaBoostClassifier())])

    param = {"classifier__n_estimators": randint(1, 150),
             "classifier__base_estimator": [DecisionTreeClassifier()]}
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid = RandomizedSearchCV(pipeline, n_iter=nIter, param_distributions=param, refit=True, n_jobs=nbCores,
                              scoring=scorer, cv=KFolds, random_state=randomState)
    detector = grid.fit(X_train, y_train)
    desc_estimators = [detector.best_params_["classifier__n_estimators"],
                       detector.best_params_["classifier__base_estimator"]]

    scoresArray = detector.cv_results_['mean_test_score']
    params = [("baseEstimators", np.array(["DecisionTree" for _ in range(nIter)])),
              ("nEstimators", np.array(detector.cv_results_['param_classifier__n_estimators']))]

    genHeatMaps(params, scoresArray, outputFileName)
    return desc_estimators


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- Adaboost with num_esimators : " + str(config.n_estimators) + ", base_estimators : " + str(
            config.base_estimator)
    else:
        try:
            return "\n\t\t- Adaboost with num_esimators : " + str(config[0]) + ", base_estimators : " + str(config[1])
        except:
            return "\n\t\t- Adaboost with num_esimators : " + str(config["0"]) + ", base_estimators : " + str(
                config["1"])

def getInterpret(classifier, directory):
    interpretString = getFeatureImportance(classifier, directory)
    return interpretString