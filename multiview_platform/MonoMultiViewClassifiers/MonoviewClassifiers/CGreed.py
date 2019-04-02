from ..Monoview.MonoviewUtils import BaseMonoviewClassifier, CustomRandint
from ..Monoview.Additions.BoostUtils import getInterpretBase
from ..Monoview.Additions.CGDescUtils import ColumnGenerationClassifierQar


class CGreed(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_max_iterations=500, n_stumps=10, **kwargs):
        super(CGreed, self).__init__(n_max_iterations=n_max_iterations,
            random_state=random_state,
            self_complemented=True,
            twice_the_same=False,
            c_bound_choice=True,
            random_start=False,
            n_stumps=n_stumps,
            use_r=True,
            c_bound_sol=True,
            estimators_generator="Stumps"
            )

        self.param_names = ["n_max_iterations", "n_stumps", "random_state"]
        self.distribs = [CustomRandint(low=2, high=1000), [n_stumps], [random_state]]
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
    kwargsDict = {"n_stumps":args.CGR_stumps,
    "n_max_iterations":args.CGR_n_iter}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet