from sklearn.externals.six import iteritems
from pyscm.scm import SetCoveringMachineClassifier as scm
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time
import os

from ..Monoview.MonoviewUtils import CustomRandint, CustomUniform, BaseMonoviewClassifier
from ..Monoview.Additions.PregenUtils import PregenClassifier
from ..Metrics import zero_one_loss

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class SCMSparsityTree(BaseMonoviewClassifier, PregenClassifier):

    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, n_stumps=1, max_depth=2, **kwargs):
        self.scm_estimators = [scm(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rule+1,
            p=p
            ) for max_rule in range(max_rules)]
        self.model_type = model_type
        self.max_depth=max_depth
        self.p = p
        self.n_stumps = n_stumps
        self.random_state = random_state
        self.max_rules = max_rules
        self.param_names = ["model_type", "max_rules", "p", "random_state", "max_depth"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [random_state], [max_depth]]
        self.classed_params = []
        self.weird_strings = {}

    def get_params(self):
        return {"model_type":self.model_type, "p":self.p, "max_rules":self.max_rules, "random_state":self.random_state, "max_depth":self.max_depth, "n_stumps":self.n_stumps}

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        pregen_X, _ = self.pregen_voters(X, y, generator="Trees")
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x"+str(a)+".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x"+str(a)+".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        for scm_estimator in self.scm_estimators:
            beg = time.time()
            scm_estimator.fit(place_holder, y, tiebreaker=None, iteration_callback=None, **fit_params)
            end = time.time()
        self.times = np.array([end-beg, 0])
        self.train_metrics = [zero_one_loss.score(y, scm_estimator.predict(place_holder)) for scm_estimator in self.scm_estimators]
        return self.scm_estimators[-1]

    def predict(self, X):
        pregen_X, _ = self.pregen_voters(X, generator="Trees")
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x"+str(a)+".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x"+str(a)+".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        self.preds = [scm_estimator.predict(place_holder) for scm_estimator in self.scm_estimators]
        return self.preds[-1]

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        np.savetxt(directory+"test_metrics.csv", np.array([zero_one_loss.score(y_test, pred) for pred in self.preds]))
        np.savetxt(directory + "times.csv", self.times)
        np.savetxt(directory + "train_metrics.csv", self.train_metrics)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"model_type": args.SCST_model_type,
                  "p": args.SCST_p,
                  "max_rules": args.SCST_max_rules,
                  "n_stumps": args.SCST_trees,
                  "max_depth": args.SCST_max_depth}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"model_type": randomState.choice(["conjunction", "disjunction"]),
                          "max_rules": randomState.randint(1, 15),
                          "p": randomState.random_sample()})
    return paramsSet
