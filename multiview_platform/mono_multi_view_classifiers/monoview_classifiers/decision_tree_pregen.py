import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..monoview.additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier, \
    change_label_to_zero

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "DecisionTreePregen"

class DecisionTreePregen(DecisionTreeClassifier, BaseMonoviewClassifier,
                         PregenClassifier):

    def __init__(self, random_state=None, max_depth=None,
                 criterion='gini', splitter='best', n_stumps=1,
                 self_complemented=True, **kwargs):
        super(DecisionTreePregen, self).__init__(
            max_depth=max_depth,
            criterion=criterion,
            splitter=splitter,
            random_state=random_state
        )
        self.estimators_generator = "Stumps"
        self.n_stumps = n_stumps
        self.self_complemented = self_complemented
        self.param_names = ["max_depth", "criterion", "splitter",
                            'random_state',
                            'n_stumps']
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         ["gini", "entropy"],
                         ["best", "random"], [random_state], [n_stumps]]
        self.weird_strings = {}

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        begin = time.time()
        pregen_X, pregen_y = self.pregen_voters(X, y)
        super(DecisionTreePregen, self).fit(pregen_X, pregen_y,
                                            sample_weight=sample_weight,
                                            check_input=check_input,
                                            X_idx_sorted=X_idx_sorted)
        end = time.time()
        self.train_time = end - begin
        self.train_shape = pregen_X.shape
        return self

    def predict(self, X, check_input=True):
        begin = time.time()
        pregen_X, _ = self.pregen_voters(X)
        pred = super(DecisionTreePregen, self).predict(pregen_X,
                                                       check_input=check_input)
        end = time.time()
        self.pred_time = end - begin
        return change_label_to_zero(pred)

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return False

    def getInterpret(self, directory, y_test):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        np.savetxt(directory + "times.csv",
                   np.array([self.train_time, self.pred_time]), delimiter=',')
        return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"max_depth": args.DTP_depth,
#                   "criterion": args.DTP_criterion,
#                   "splitter": args.DTP_splitter,
#                   "n_stumps": args.DTP_stumps}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"max_depth": random_state.randint(1, 300),
                          "criterion": random_state.choice(["gini", "entropy"]),
                          "splitter": random_state.choice(["best", "random"])})
    return paramsSet
