import numpy as np
from sklearn.linear_model import Lasso as LassoSK

from ..Monoview.MonoviewUtils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class Lasso(LassoSK, BaseMonoviewClassifier):

    def __init__(self, random_state=None, alpha=1.0,
                 max_iter=10, warm_start=False, **kwargs):
        super(Lasso, self).__init__(
            alpha=alpha,
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state
        )
        self.param_names = ["max_iter", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomUniform(), [random_state]]
        self.weird_strings = {}

    def fit(self, X, y, check_input=True):
        neg_y = np.copy(y)
        neg_y[np.where(neg_y == 0)] = -1
        super(Lasso, self).fit(X, neg_y)
        return self

    def predict(self, X):
        prediction = super(Lasso, self).predict(X)
        signed = np.sign(prediction)
        signed[np.where(signed == -1)] = 0
        return signed

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return False

    def getInterpret(self, directory, y_test):
        interpretString = ""
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"alpha": args.LA_alpha,
                  "max_iter": args.LA_n_iter}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"max_iter": randomState.randint(1, 300),
                          "alpha": randomState.uniform(0, 1.0), })
    return paramsSet
