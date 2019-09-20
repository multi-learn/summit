from ..monoview.additions.CGDescUtils import ColumnGenerationClassifierQar
from ..monoview.monoview_utils import BaseMonoviewClassifier, CustomRandint


class CGDescTree(ColumnGenerationClassifierQar, BaseMonoviewClassifier):

    def __init__(self, random_state=None, n_max_iterations=500, n_stumps=1,
                 max_depth=2, **kwargs):
        super(CGDescTree, self).__init__(n_max_iterations=n_max_iterations,
                                         random_state=random_state,
                                         self_complemented=True,
                                         twice_the_same=True,
                                         c_bound_choice=True,
                                         random_start=False,
                                         n_stumps=n_stumps,
                                         use_r=True,
                                         c_bound_sol=True,
                                         estimators_generator="Trees"
                                         )
        self.max_depth = max_depth
        self.param_names = ["n_max_iterations", "n_stumps", "random_state",
                            "max_depth"]
        self.distribs = [CustomRandint(low=2, high=1000), [n_stumps],
                         [random_state], [max_depth]]
        self.classed_params = []
        self.weird_strings = {}

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def getInterpret(self, directory, y_test):
        return self.getInterpretQar(directory, y_test)

    def get_name_for_fusion(self):
        return "CGDT"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"n_stumps": args.CGDT_trees,
                  "n_max_iterations": args.CGDT_n_iter,
                  "max_depth": args.CGDT_max_depth}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
