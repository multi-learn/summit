from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class GradientBoosting(GradientBoostingClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, loss="exponential", max_depth=1.0,
                 n_estimators=100, init=DecisionTreeClassifier(max_depth=1), **kwargs):
        super(GradientBoosting, self).__init__(
            loss=loss,
            max_depth=max_depth,
            n_estimators=n_estimators,
            init=init,
            random_state=random_state
            )
        self.param_names = ["n_estimators",]
        self.classed_params = []
        self.distribs = [CustomRandint(low=50, high=500),]
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_estimators": args.GB_n_est,}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": randomState.randint(50, 500),})
    return paramsSet