import os
import time

import numpy as np
from pyscm.scm import SetCoveringMachineClassifier as scm

from ..metrics import zero_one_loss
from ..monoview.additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class SCMSparsity(BaseMonoviewClassifier, PregenClassifier):

    def __init__(self, random_state=None, model_type="disjunction",
                 max_rules=10, p=0.1, n_stumps=1, self_complemented=True,
                 **kwargs):
        self.scm_estimators = [scm(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rule + 1,
            p=p
        ) for max_rule in range(max_rules)]
        self.model_type = model_type
        self.self_complemented = self_complemented
        self.n_stumps = n_stumps
        self.p = p
        self.random_state = random_state
        self.max_rules = max_rules
        self.param_names = ["model_type", "max_rules", "p", "random_state",
                            "n_stumps"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [random_state],
                         [n_stumps]]
        self.classed_params = []
        self.weird_strings = {}

    def get_params(self):
        return {"model_type": self.model_type, "p": self.p,
                "max_rules": self.max_rules, "random_state": self.random_state,
                "n_stumps": self.n_stumps}

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        pregen_X, _ = self.pregen_voters(X, y)
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x" + str(a) + ".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x" + str(a) + ".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        for scm_estimator in self.scm_estimators:
            beg = time.time()
            scm_estimator.fit(place_holder, y, tiebreaker=None,
                              iteration_callback=None, **fit_params)
            end = time.time()
        self.times = np.array([end - beg, 0])
        self.train_metrics = [
            zero_one_loss.score(y, scm_estimator.predict(place_holder)) for
            scm_estimator in self.scm_estimators]
        return self.scm_estimators[-1]

    def predict(self, X):
        pregen_X, _ = self.pregen_voters(X, )
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x" + str(a) + ".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x" + str(a) + ".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        self.preds = [scm_estimator.predict(place_holder) for scm_estimator in
                      self.scm_estimators]
        return self.preds[-1]

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        np.savetxt(directory + "test_metrics.csv", np.array(
            [zero_one_loss.score(y_test, pred) for pred in self.preds]))
        np.savetxt(directory + "times.csv", self.times)
        np.savetxt(directory + "train_metrics.csv", self.train_metrics)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"model_type": args.SCS_model_type,
                  "p": args.SCS_p,
                  "max_rules": args.SCS_max_rules,
                  "n_stumps": args.SCS_stumps}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append(
            {"model_type": randomState.choice(["conjunction", "disjunction"]),
             "max_rules": randomState.randint(1, 15),
             "p": randomState.random_sample()})
    return paramsSet
