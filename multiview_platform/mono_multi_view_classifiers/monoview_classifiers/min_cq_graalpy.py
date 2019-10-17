import numpy as np

from ..monoview.additions.BoostUtils import StumpsClassifiersGenerator
from ..monoview.additions.MinCQUtils import RegularizedBinaryMinCqClassifier
from ..monoview.monoview_utils import BaseMonoviewClassifier, CustomUniform


classifier_class_name = "MinCQGraalpy"

class MinCQGraalpy(RegularizedBinaryMinCqClassifier, BaseMonoviewClassifier):
    """
    MinCQGraalpy extend of ``RegularizedBinaryMinCqClassifier ``

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    mu :  float, (default: 0.01)

    self_complemented : bool (default : True)

    n_stumps_per_attribute : (default: =1

    kwargs : others arguments


    Attributes
    ----------
    param_names

    distribs

    n_stumps_per_attribute

    classed_params

    weird_strings

    nbCores : number of cores

    """
    def __init__(self, random_state=None, mu=0.01, self_complemented=True,
                 n_stumps_per_attribute=1, **kwargs):
        super(MinCQGraalpy, self).__init__(mu=mu,
                                           estimators_generator=StumpsClassifiersGenerator(
                                               n_stumps_per_attribute=n_stumps_per_attribute,
                                               self_complemented=self_complemented),
                                           )
        self.param_names = ["mu", "n_stumps_per_attribute", "random_state"]
        self.distribs = [CustomUniform(loc=0.05, state=2.0, multiplier="e-"),
                         [n_stumps_per_attribute], [random_state]]
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.classed_params = []
        self.weird_strings = {}
        self.random_state = random_state
        if "nbCores" not in kwargs:
            self.nbCores = 1
        else:
            self.nbCores = kwargs["nbCores"]

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #     Returns
    #     -------
    #     False
    #     """
    #     return False

    def set_params(self, **params):
        """
        set parameter 'self.mu', 'self.random_state
        'self.n_stumps_per_attribute

        Parameters
        ----------
        params

        Returns
        -------
        self : object
            Returns self.
        """
        self.mu = params["mu"]
        self.random_state = params["random_state"]
        self.n_stumps_per_attribute = params["n_stumps_per_attribute"]
        return self

    def get_params(self, deep=True):
        """

        Parameters
        ----------
        deep : bool (default : true) not used

        Returns
        -------
        dictianary with "random_state",  "mu", "n_stumps_per_attribute"
        """
        return {"random_state": self.random_state, "mu": self.mu,
                "n_stumps_per_attribute": self.n_stumps_per_attribute}

    def getInterpret(self, directory, y_test):
        """

        Parameters
        ----------
        directory
        y_test

        Returns
        -------
        string of interpret_string
        """
        interpret_string = "Cbound on train :" + str(self.train_cbound)
        np.savetxt(directory + "times.csv", np.array([self.train_time, 0]))
        # interpret_string += "Train C_bound value : "+str(self.cbound_train)
        # y_rework = np.copy(y_test)
        # y_rework[np.where(y_rework==0)] = -1
        # interpret_string += "\n Test c_bound value : "+str(self.majority_vote.cbound_value(self.x_test, y_rework))
        return interpret_string

    def get_name_for_fusion(self):
        return "MCG"


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"mu": args.MCG_mu,
#                   "n_stumps_per_attribute": args.MCG_stumps}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    """Used for weighted linear early fusion to generate random search sets"""
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({})
    return paramsSet
