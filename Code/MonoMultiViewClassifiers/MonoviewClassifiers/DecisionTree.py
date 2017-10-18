from sklearn.tree import DecisionTreeClassifier
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
    maxDepth = int(kwargs['0'])
    criterion = kwargs['1']
    splitter = kwargs['2']
    classifier = DecisionTreeClassifier(max_depth=maxDepth, criterion=criterion, splitter=splitter,
                                        random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.randint(1, 300), randomState.choice(["gini", "entropy"]),
                          randomState.choice(["best", "random"])])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_DecisionTree_depth":
            kwargsDict['0'] = int(kwargValue)
        if kwargName == "CL_DecisionTree_criterion":
            kwargsDict['1'] = kwargValue
        if kwargName == "CL_DecisionTree_splitter":
            kwargsDict['2'] = kwargValue
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, nbCores=1,
                     metric=["accuracy_score", None], nIter=30):
    pipeline_DT = Pipeline([('classifier', DecisionTreeClassifier())])
    param_DT = {"classifier__max_depth": randint(1, 300),
                "classifier__criterion": ["gini", "entropy"],
                "classifier__splitter": ["best", "random"]}
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_DT = RandomizedSearchCV(pipeline_DT, n_iter=nIter, param_distributions=param_DT, refit=True, n_jobs=nbCores,
                                 scoring=scorer,
                                 cv=KFolds, random_state=randomState)
    DT_detector = grid_DT.fit(X_train, y_train)
    desc_params = [DT_detector.best_params_["classifier__max_depth"], DT_detector.best_params_["classifier__criterion"],
                   DT_detector.best_params_["classifier__splitter"]]

    scoresArray = DT_detector.cv_results_['mean_test_score']
    params = [("maxDepth", np.array(DT_detector.cv_results_['param_classifier__max_depth'])),
              ("criterion", np.array(DT_detector.cv_results_['param_classifier__criterion'])),
              ("splitter", np.array(DT_detector.cv_results_['param_classifier__splitter']))]

    genHeatMaps(params, scoresArray, outputFileName)
    return desc_params


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- Decision Tree with max_depth : " + str(
            config.max_depth) + ", criterion : " + config.criterion + ", splitter : " + config.splitter
    else:
        try:
            return "\n\t\t- Decision Tree with max_depth : " + str(config[0]) + ", criterion : " + config[
                1] + ", splitter : " + config[2]
        except:
            return "\n\t\t- Decision Tree with max_depth : " + str(config["0"]) + ", criterion : " + config[
                "1"] + ", splitter : " + config["2"]
