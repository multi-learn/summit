from ..monoview.additions.CGDescUtils import ColumnGenerationClassifierQar
from ..monoview.monoview_utils import BaseMonoviewClassifier, CustomRandint


classifier_class_name = "CGDesc"

class CGDesc(ColumnGenerationClassifierQar, BaseMonoviewClassifier):
    """

    Parameters
    ----------
    random_state
    n_max_iterations
    n_stumps
    estimators_generator
    twice_the_same
    max_depth
    kwargs
    
    """
    def __init__(self, random_state=None, n_max_iterations=500, n_stumps=1,
                 estimators_generator="Stumps", twice_the_same=True, max_depth=1,
                 **kwargs):

        super(CGDesc, self).__init__(n_max_iterations=n_max_iterations,
                                     random_state=random_state,
                                     self_complemented=True,
                                     twice_the_same=twice_the_same,
                                     c_bound_choice=True,
                                     random_start=False,
                                     n_stumps=n_stumps,
                                     use_r=False,
                                     c_bound_sol=True,
                                     estimators_generator=estimators_generator,
                                     max_depth=max_depth,
                                     mincq_tracking=False,
                                     )
        self.param_names = ["n_max_iterations", "n_stumps",
                            "estimators_generator", "max_depth", "random_state", "twice_the_same"]
        self.distribs = [CustomRandint(low=2, high=500), [n_stumps],
                         ["Stumps", "Trees"], CustomRandint(low=1, high=5),
                         [random_state], [True, False]]
        self.classed_params = []
        self.weird_strings = {}

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return False

    def getInterpret(self, directory, y_test):
        return self.getInterpretQar(directory, y_test)

    def get_name_for_fusion(self):
        return "CGD"


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_stumps": args.CGD_stumps,
#                   "n_max_iterations": args.CGD_n_iter}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
