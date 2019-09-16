from ..Monoview.Additions.CBBoostUtils import CBBoostClassifier
from ..Monoview.MonoviewUtils import BaseMonoviewClassifier, CustomRandint


class CBBoost(CBBoostClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_max_iterations=500, n_stumps=1,
                 **kwargs):
        super(CBBoost, self).__init__(n_max_iterations=n_max_iterations,
                                     random_state=random_state,
                                     self_complemented=True,
                                     twice_the_same=True,
                                     random_start=False,
                                     n_stumps=n_stumps,
                                     c_bound_sol=True,
                                     estimators_generator="Stumps",
                                     mincq_tracking=False
                                     )
        self.param_names = ["n_max_iterations", "n_stumps", "random_state"]
        self.distribs = [CustomRandint(low=2, high=500), [n_stumps],
                         [random_state]]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        return self.getInterpretCBBoost(directory, y_test)

    def get_name_for_fusion(self):
        return "CBB"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_stumps": args.CBB_stumps,
                  "n_max_iterations": args.CBB_n_iter}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
