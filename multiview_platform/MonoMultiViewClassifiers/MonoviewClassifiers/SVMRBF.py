from ..Monoview.Additions.SVCClassifier import SVCClassifier
from ..Monoview.MonoviewUtils import CustomUniform, CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class SVMRBF(SVCClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, C=1.0, **kwargs):
        super(SVMRBF, self).__init__(
            C=C,
            kernel='rbf',
            random_state=random_state
        )
        self.param_names = ["C",]
        self.distribs = [CustomUniform(loc=0, state=1),]


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"C": args.SVMRBF_C}
    return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"C": randomState.randint(1, 10000), })
    return paramsSet