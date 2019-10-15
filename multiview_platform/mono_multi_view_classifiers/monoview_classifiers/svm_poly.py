from ..monoview.additions.SVCClassifier import SVCClassifier
from ..monoview.monoview_utils import CustomUniform, CustomRandint, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SVMPoly"

class SVMPoly(SVCClassifier, BaseMonoviewClassifier):
    """
    Class of SVMPoly for SVC Classifier

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.


    C : float, optional (default=1.0)
        Penalty parameter C of the error term.


    degree :

    kwargs : others arguments


    Attributes
    ----------

    param_names : list of parameters names

    distribs :  list of random_state distribution
    """
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
