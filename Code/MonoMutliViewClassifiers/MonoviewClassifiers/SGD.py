from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline  # Pipelining in classification
from sklearn.model_selection import RandomizedSearchCV
import Metrics
from scipy.stats import uniform
import numpy as np
from utils.HyperParameterSearch import genHeatMaps

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def canProbas():
    return True


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    loss = kwargs['0']
    penalty = kwargs['1']
    try:
        alpha = float(kwargs['2'])
    except:
        alpha = 0.15
    classifier = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, random_state=randomState, n_jobs=NB_CORES)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.choice(['log', 'modified_huber']),
                          randomState.choice(["l1", "l2", "elasticnet"]), randomState.random_sample()])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_SGD_loss":
            kwargsDict['0'] = kwargValue
        elif kwargName == "CL_SGD_penalty":
            kwargsDict['1'] = kwargValue
        elif kwargName == "CL_SGD_alpha":
            kwargsDict['2'] = float(kwargValue)
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, nbCores=1,
                     metric=["accuracy_score", None], nIter=30):
    pipeline_SGD = Pipeline([('classifier', SGDClassifier())])
    losses = ['log', 'modified_huber']
    penalties = ["l1", "l2", "elasticnet"]
    alphas = uniform()
    param_SGD = {"classifier__loss": losses, "classifier__penalty": penalties,
                 "classifier__alpha": alphas}
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid_SGD = RandomizedSearchCV(pipeline_SGD, n_iter=nIter, param_distributions=param_SGD, refit=True,
                                  n_jobs=nbCores, scoring=scorer, cv=KFolds, random_state=randomState)
    SGD_detector = grid_SGD.fit(X_train, y_train)
    desc_params = [SGD_detector.best_params_["classifier__loss"], SGD_detector.best_params_["classifier__penalty"],
                   SGD_detector.best_params_["classifier__alpha"]]

    scoresArray = SGD_detector.cv_results_['mean_test_score']
    params = [("loss", np.array(SGD_detector.cv_results_['param_classifier__loss'])),
              ("penalty", np.array(SGD_detector.cv_results_['param_classifier__penalty'])),
              ("aplha", np.array(SGD_detector.cv_results_['param_classifier__alpha']))]

    genHeatMaps(params, scoresArray, outputFileName)

    return desc_params


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SGDClassifier with loss : " + config.loss + ", penalty : " + config.penalty + ", alpha : " + str(
            config.alpha)
    else:
        try:
            return "\n\t\t- SGDClassifier with loss : " + config[0] + ", penalty : " + config[1] + ", alpha : " + str(
                config[2])
        except:
            return "\n\t\t- SGDClassifier with loss : " + config["0"] + ", penalty : " + config[
                "1"] + ", alpha : " + str(config["2"])
