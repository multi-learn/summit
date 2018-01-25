import numpy as np

from pyscm.scm import SetCoveringMachineClassifier as scm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals.six import iteritems
from scipy.stats import uniform, randint

from .. import Metrics
from ..utils.HyperParameterSearch import genHeatMaps

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class DecisionStumpSCMNew(BaseEstimator, ClassifierMixin):
    """docstring for SCM
    A hands on class of SCM using decision stump, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ..."""

    def __init__(self, model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(DecisionStumpSCMNew, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state

    def fit(self, X, y):
        self.clf = scm(model_type=self.model_type, max_rules=self.max_rules, p=self.p, random_state=self.random_state)
        self.clf.fit(X=X, y=y)

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value

    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}


def canProbas():
    return False


def fit(DATASET, CLASS_LABELS, randomState, NB_CORES=1, **kwargs):
    modelType = kwargs['0']
    maxRules = int(kwargs['1'])
    p = float(kwargs["2"])
    classifier = DecisionStumpSCMNew(model_type=modelType, max_rules=maxRules, p=p, random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.choice(["conjunction", "disjunction"]), randomState.randint(1, 15), randomState.random_sample()])
    return paramsSet


def getKWARGS(kwargsList):
    kwargsDict = {}
    for (kwargName, kwargValue) in kwargsList:
        if kwargName == "CL_SCM_model_type":
            kwargsDict['0'] = kwargValue
        elif kwargName == "CL_SCM_max_rules":
            kwargsDict['1'] = int(kwargValue)
        elif kwargName == "CL_SCM_p":
            kwargsDict['2'] = float(kwargValue)
        else:
            raise ValueError("Wrong arguments served to SCM")
    return kwargsDict


def randomizedSearch(X_train, y_train, randomState, outputFileName, KFolds=4, metric=["accuracy_score", None], nIter=30,
                     nbCores=1):
    pipeline = Pipeline([('classifier', DecisionStumpSCMNew())])

    param = {"classifier__model_type": ['conjunction', 'disjunction'],
             "classifier__p": uniform(),
             "classifier__max_rules": randint(1,30)}
    metricModule = getattr(Metrics, metric[0])
    if metric[1] is not None:
        metricKWARGS = dict((index, metricConfig) for index, metricConfig in enumerate(metric[1]))
    else:
        metricKWARGS = {}
    scorer = metricModule.get_scorer(**metricKWARGS)
    grid = RandomizedSearchCV(pipeline, n_iter=nIter, param_distributions=param, refit=True, n_jobs=nbCores,
                              scoring=scorer, cv=KFolds, random_state=randomState)
    detector = grid.fit(X_train, y_train)
    desc_estimators = [detector.best_params_["classifier__model_type"],
                       detector.best_params_["classifier__max_rules"],
                       detector.best_params_["classifier__p"]]

    scoresArray = detector.cv_results_['mean_test_score']
    params = [("model_type", np.array(detector.cv_results_['param_classifier__model_type'])),
              ("maxRules", np.array(detector.cv_results_['param_classifier__max_rules'])),
              ("p", np.array(detector.cv_results_['param_classifier__p']))]

    genHeatMaps(params, scoresArray, outputFileName)
    return desc_estimators


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SCM with model_type: " + config.model_type + ", max_rules : " + str(config.max_rules) +\
               ", p : " + str(config.p)
    else:
        try:
            return "\n\t\t- SCM with model_type: " + config[0] + ", max_rules : " + str(config[1]) + ", p : " +\
                   str(config[2])
        except:
            return "\n\t\t- SCM with model_type: " + config["0"] + ", max_rules : " + str(config["1"]) + ", p : " + \
                   str(config["2"])


def getInterpret(classifier, directory):
    return "Model used : " + str(classifier.clf.model_)
