from ..Monoview.MonoviewUtils import BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.QarBoostUtils import ColumnGenerationClassifierQar


class QarBoost(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, **kwargs):
        super(QarBoost, self).__init__(
            random_state=random_state)

        self.param_names = ["self_complemented", "twice_the_same", "old_fashioned", "previous_vote_weighted",
                            "c_bound_choice", "random_start", "two_wieghts_problem"]
        self.distribs = [[True, False], [True, False], [True, False], [True, False],
                         [True, False], [True, False], [True, False]]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        self.getInterpretQar(directory)
        return getInterpretBase(self, directory, "QarBoost", self.weights_, self.break_cause)


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
