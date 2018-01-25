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
    classifier = DecisionStumpSCMNew(model_type=kwargs['model_type'],
                                     max_rules=kwargs['max_rules'],
                                     p=kwargs['p'],
                                     random_state=randomState)
    classifier.fit(DATASET, CLASS_LABELS)
    return classifier


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append([randomState.choice(["conjunction", "disjunction"]), randomState.randint(1, 15), randomState.random_sample()])
    return paramsSet


def getKWARGS(args):
    kwargsDict = {"model_type": args.SCM_model_type,
                  "p": args.SCM_p,
                  "max_rules": args.SCM_max_rules}
    return kwargsDict


def genPipeline():
    return Pipeline([('classifier', DecisionStumpSCMNew())])


def genParamsDict(randomState):
    return {"classifier__model_type": ['conjunction', 'disjunction'],
             "classifier__p": uniform(),
             "classifier__max_rules": np.arange(1,30)}


def genBestParams(detector):
    return {"model_type": detector.best_params_["classifier__model_type"],
            "p": detector.best_params_["classifier__p"],
            "max_rules": detector.best_params_["classifier__max_rules"]}


def genParamsFromDetector(detector):
    return [("model_type", np.array(detector.cv_results_['param_classifier__model_type'])),
            ("maxRules", np.array(detector.cv_results_['param_classifier__max_rules'])),
            ("p", np.array(detector.cv_results_['param_classifier__p']))]


def getConfig(config):
    if type(config) not in [list, dict]:
        return "\n\t\t- SCM with model_type: " + config.model_type + ", max_rules : " + str(config.max_rules) +\
               ", p : " + str(config.p)
    else:
        return "\n\t\t- SCM with model_type: " + config["model_type"] + ", max_rules : " + str(config["max_rules"]) + ", p : " + \
                   str(config["p"])


def getInterpret(classifier, directory):
    return "Model used : " + str(classifier.clf.model_)
