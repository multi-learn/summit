from sklearn.tree import DecisionTreeClassifier

from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "DecisionTree"


class DecisionTree(DecisionTreeClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, max_depth=None,
                 criterion='gini', splitter='best', **kwargs):
        super(DecisionTree, self).__init__(
            max_depth=max_depth,
            criterion=criterion,
            splitter=splitter,
            random_state=random_state
        )
        self.param_names = ["max_depth", "criterion", "splitter",
                            'random_state']
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         ["gini", "entropy"],
                         ["best", "random"], [random_state]]
        self.weird_strings = {}

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return True

    def get_interpretation(self, directory, y_test):
        interpretString = "First featrue : \n\t{} <= {}\n".format(self.tree_.feature[0],
                                                               self.tree_.threshold[0])
        interpretString += self.get_feature_importance(directory)
        return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"max_depth": args.DT_depth,
#                   "criterion": args.DT_criterion,
#                   "splitter": args.DT_splitter}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"max_depth": randomState.randint(1, 300),
                          "criterion": randomState.choice(["gini", "entropy"]),
                          "splitter": randomState.choice(["best", "random"])})
    return paramsSet
