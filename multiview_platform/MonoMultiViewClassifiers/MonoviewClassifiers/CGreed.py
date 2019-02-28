from ..Monoview.MonoviewUtils import BaseMonoviewClassifier, CustomRandint
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.QarBoostUtils import ColumnGenerationClassifierQar


class CGreed(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_max_iterations=500, n_stumps_per_attribute=1, **kwargs):
        super(CGreed, self).__init__(n_max_iterations=n_max_iterations,
            random_state=random_state,
            self_complemented=True,
            twice_the_same=False,
            c_bound_choice=True,
            random_start=False,
            n_stumps_per_attribute=n_stumps_per_attribute,
            use_r=True,
            c_bound_sol=True
            )

        self.param_names = ["n_max_iterations"]
        self.distribs = [CustomRandint(low=2, high=1000)]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        return self.getInterpretQar(directory, y_test)

    def get_name_for_fusion(self):
        return "CGr"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_stumps_per_attribute":args.CGR_stumps,
    "n_max_iterations":args.CGR_n_iter}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet