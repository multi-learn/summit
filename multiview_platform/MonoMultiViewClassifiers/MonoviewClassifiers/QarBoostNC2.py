from ..Monoview.MonoviewUtils import BaseMonoviewClassifier
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.QarBoostUtils import ColumnGenerationClassifierQar


class QarBoostNC2(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, **kwargs):
        super(QarBoostNC2, self).__init__(
            random_state=random_state,
            self_complemented=False,
            twice_the_same=True
            )
        self.param_names = []
        self.distribs = []
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory):
        return getInterpretBase(self, directory, "QarBoostNC2", self.weights_, self.break_cause)

    def get_name_for_fusion(self):
        return "QBN2"


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