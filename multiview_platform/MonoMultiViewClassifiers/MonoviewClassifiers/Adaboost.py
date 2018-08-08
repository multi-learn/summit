from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class Adaboost(AdaBoostClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_estimators=50,
                 base_estimator=None, **kwargs):
        super(Adaboost, self).__init__(
            random_state=random_state,
            n_estimators=n_estimators,
            base_estimator=base_estimator,
            )
        self.param_names = ["n_estimators", "base_estimator"]
        self.classed_params = ["base_estimator"]
        self.distribs = [CustomRandint(low=1, high=500), [None]]
        self.weird_strings = {"base_estimator":"class_name"}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {'n_estimators': args.Ada_n_est,
                  'base_estimator': DecisionTreeClassifier()}
    return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 500),
                          "base_estimator": None})
    return paramsSet
