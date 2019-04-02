import numpy as np

from ..Monoview.Additions.MinCQUtils import RegularizedBinaryMinCqClassifier
from ..Monoview.Additions.BoostUtils import TreeClassifiersGenerator
from ..Monoview.MonoviewUtils import BaseMonoviewClassifier, CustomUniform



class MinCQGraalpyTree(RegularizedBinaryMinCqClassifier, BaseMonoviewClassifier):

    def __init__(self, random_state=None, mu=0.01, self_complemented=True, n_stumps_per_attribute=1, max_depth=2, **kwargs):
        super(MinCQGraalpyTree, self).__init__(mu=mu,
            estimators_generator=TreeClassifiersGenerator(random_state=random_state,
                                                          n_trees=n_stumps_per_attribute,
                                                          max_depth=max_depth,
                                                          self_complemented=self_complemented),
        )
        self.param_names = ["mu", "n_stumps_per_attribute", "random_state", "max_depth"]
        self.distribs = [CustomUniform(loc=0.05, state=2.0, multiplier="e-"),
                         [n_stumps_per_attribute], [random_state], [max_depth]]
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.classed_params = []
        self.weird_strings = {}
        self.max_depth = max_depth
        self.random_state = random_state
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    def canProbas(self):
        """Used to know if the classifier can return label probabilities"""
        return True

    def set_params(self, **params):
        self.mu = params["mu"]
        self.random_state = params["random_state"]
        self.n_stumps_per_attribute = params["n_stumps_per_attribute"]
        self.max_depth = params["max_depth"]
        return self

    def get_params(self, deep=True):
        return {"random_state":self.random_state, "mu":self.mu, "n_stumps_per_attribute":self.n_stumps_per_attribute, "max_depth":self.max_depth}

    def getInterpret(self, directory, y_test):
        interpret_string = "Cbound on train :"+str(self.train_cbound)
        np.savetxt(directory+"times.csv", np.array([self.train_time, 0]))
        # interpret_string += "Train C_bound value : "+str(self.cbound_train)
        # y_rework = np.copy(y_test)
        # y_rework[np.where(y_rework==0)] = -1
        # interpret_string += "\n Test c_bound value : "+str(self.majority_vote.cbound_value(self.x_test, y_rework))
        return interpret_string

    def get_name_for_fusion(self):
        return "MCG"


def formatCmdArgs(args):
    """Used to format kwargs for the parsed args"""
    kwargsDict = {"mu":args.MCGT_mu,
                  "n_stumps_per_attribute":args.MCGT_trees,
                  "max_depth":args.MCGT_max_depth}
    return kwargsDict


def paramsToSet(nIter, randomState):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet