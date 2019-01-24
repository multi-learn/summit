from sklearn.ensemble import RandomForestClassifier

from ..Monoview.MonoviewUtils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class RandomForest(RandomForestClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_estimators=10,
                 max_depth=None, criterion='gini', **kwargs):
        super(RandomForest, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state
            )
        self.param_names = ["n_estimators", "max_depth", "criterion",]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomRandint(low=1, high=300),
                         ["gini", "entropy"], ]
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        interpretString += self.getFeatureImportance(directory)
        return interpretString


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_estimators": args.RF_trees,
                  "max_depth": args.RF_max_depth,
                  "criterion": args.RF_criterion}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": randomState.randint(1, 300),
                          "max_depth": randomState.randint(1, 300),
                          "criterion": randomState.choice(["gini", "entropy"])})
    return paramsSet
