import numpy as np

from ..monoview.additions.BoostUtils import getInterpretBase
from ..monoview.additions.CQBoostUtils import ColumnGenerationClassifier
from ..monoview.monoview_utils import CustomUniform, CustomRandint, \
    BaseMonoviewClassifier

classifier_class_name = "CQBoost"

class CQBoost(ColumnGenerationClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, epsilon=1e-06, n_stumps=1,
                 n_max_iterations=None, estimators_generator="Stumps",
                 max_depth=1, **kwargs):
        super(CQBoost, self).__init__(
            random_state=random_state,
            mu=mu,
            epsilon=epsilon,
            estimators_generator=estimators_generator,
            n_max_iterations=n_max_iterations,
            max_depth=max_depth
        )
        self.param_names = ["mu", "epsilon", "n_stumps", "random_state",
                            "n_max_iterations", "estimators_generator",
                            "max_depth"]
        self.distribs = [CustomUniform(loc=0.5, state=1.0, multiplier="e-"),
                         CustomRandint(low=1, high=15, multiplier="e-"),
                         [n_stumps], [random_state], [n_max_iterations],
                         ["Stumps", "Trees"], CustomRandint(low=1, high=5)]
        self.classed_params = []
        self.weird_strings = {}
        self.n_stumps = n_stumps
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities"""
    #     return False

    def getInterpret(self, directory, y_test):
        np.savetxt(directory + "train_metrics.csv", self.train_metrics,
                   delimiter=',')
        np.savetxt(directory + "c_bounds.csv", self.c_bounds,
                   delimiter=',')
        np.savetxt(directory + "y_test_step.csv", self.step_decisions,
                   delimiter=',')
        step_metrics = []
        for step_index in range(self.step_decisions.shape[1] - 1):
            step_metrics.append(self.plotted_metric.score(y_test,
                                                          self.step_decisions[:,
                                                          step_index]))
        step_metrics = np.array(step_metrics)
        np.savetxt(directory + "step_test_metrics.csv", step_metrics,
                   delimiter=',')
        return getInterpretBase(self, directory, "CQBoost", self.weights_,
                                y_test)


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"mu": args.CQB_mu,
#                   "epsilon": args.CQB_epsilon,
#                   "n_stumps": args.CQB_stumps,
#                   "n_max_iterations": args.CQB_n_iter}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"mu": 10 ** -randomState.uniform(0.5, 1.5),
                          "epsilon": 10 ** -randomState.randint(1, 15)})
    return paramsSet
