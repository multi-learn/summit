from sklearn.linear_model import SGDClassifier

from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SGD"

class SGD(SGDClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, loss='hinge',
                 penalty='l2', alpha=0.0001, **kwargs):
        super(SGD, self).__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            random_state=random_state
        )
        self.param_names = ["loss", "penalty", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [['log', 'modified_huber'],
                         ["l1", "l2", "elasticnet"],
                         CustomUniform(loc=0, state=1), [random_state]]
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        interpretString = ""
        return interpretString


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"loss": args.SGD_loss,
#                   "penalty": args.SGD_penalty,
#                   "alpha": args.SGD_alpha}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"loss": randomState.choice(['log', 'modified_huber']),
                          "penalty": randomState.choice(
                              ["l1", "l2", "elasticnet"]),
                          "alpha": randomState.random_sample()})
    return paramsSet
