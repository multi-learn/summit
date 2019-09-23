from ..monoview.additions.SVCClassifier import SVCClassifier
from ..monoview.monoview_utils import CustomUniform, CustomRandint, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


class SVMPoly(SVCClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, C=1.0, degree=3, **kwargs):
        super(SVMPoly, self).__init__(
            C=C,
            kernel='poly',
            degree=degree,
            random_state=random_state
        )
        self.param_names = ["C", "degree", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1),
                         CustomRandint(low=2, high=30), [random_state]]


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"C": args.SVMPoly_C, "degree": args.SVMPoly_deg}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"C": randomState.randint(1, 10000),
                          "degree": randomState.randint(1, 30)})
    return paramsSet
