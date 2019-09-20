import os

import numpy as np
from pyscm.scm import SetCoveringMachineClassifier as scm

from ..monoview.additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class SCMPregenTree(scm, BaseMonoviewClassifier, PregenClassifier):

    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, n_stumps=10, self_complemented=True,
                 max_depth=2, **kwargs):
        super(SCMPregenTree, self).__init__(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rules,
            p=p
        )
        self.param_names = ["model_type", "max_rules", "p", "n_stumps",
                            "random_state", "max_depth"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [n_stumps],
                         [random_state], [max_depth]]
        self.classed_params = []
        self.weird_strings = {}
        self.max_depth = max_depth
        self.self_complemented = self_complemented
        self.random_state = random_state
        self.n_stumps = n_stumps
        self.estimators_generator = "Stumps"

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        pregen_X, _ = self.pregen_voters(X, y, generator="Trees")
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
        super(SCMPregenTree, self).fit(place_holder, y, tiebreaker=tiebreaker,
                                       iteration_callback=iteration_callback,
                                       **fit_params)
        return self

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
        return self.classes_[self.model_.predict(place_holder)]

    def get_params(self, deep=True):
        return {"p": self.p, "model_type": self.model_type,
                "max_rules": self.max_rules,
                "random_state": self.random_state, "n_stumps": self.n_stumps,
                "max_depth": self.max_depth}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return False

    def getInterpret(self, directory, y_test):
        interpretString = "Model used : " + str(self.model_)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"model_type": args.SCPT_model_type,
                  "p": args.SCPT_p,
                  "max_rules": args.SCPT_max_rules,
                  "n_stumps": args.SCPT_trees,
                  "max_depth": args.SCPT_max_depth}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append(
            {"model_type": randomState.choice(["conjunction", "disjunction"]),
             "max_rules": randomState.randint(1, 15),
             "p": randomState.random_sample()})
    return paramsSet
