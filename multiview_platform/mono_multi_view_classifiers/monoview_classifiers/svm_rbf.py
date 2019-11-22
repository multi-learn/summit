from multiview_platform.mono_multi_view_classifiers.monoview_classifiers.additions.SVCClassifier import SVCClassifier
from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "SVMRBF"

class SVMRBF(SVCClassifier, BaseMonoviewClassifier):
    """
    class SVMRBF for classifier SVCC

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    C :

    kwargs : others arguments

    Attributes
    ----------

    param_names : list of parameters names

    distribs :  list of random_state distribution
    """
    def __init__(self, random_state=None, C=1.0, **kwargs):

        super(SVMRBF, self).__init__(
            C=C,
            kernel='rbf',
            random_state=random_state
        )
        self.param_names = ["C", "random_state"]
        self.distribs = [CustomUniform(loc=0, state=1), [random_state]]


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"C": args.SVMRBF_C}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    """

    Parameters
    ----------
    nIter : int number of iterations

    randomState :

    Returns
    -------
    paramsSet list of parameters dictionary  with key "C"
    """
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"C": randomState.randint(1, 10000), })
    return paramsSet
