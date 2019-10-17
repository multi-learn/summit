import os

import numpy as np
from pyscm.scm import SetCoveringMachineClassifier as scm

from ..monoview.additions.PregenUtils import PregenClassifier
from ..monoview.monoview_utils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SCMPregen"

class SCMPregen(BaseMonoviewClassifier, PregenClassifier, scm):
    """

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    model_type
    max_rules
    p
    n_stumps
    self_complemented
    estimators_generator
    max_depth
    kwargs

    Attributes
    ----------
    param_names

    distribs
    classed_params
    weird_strings
    self_complemented
    n_stumps
    estimators_generator
    max_depth
    """
    def __init__(self, random_state=None, model_type="conjunction",
                 max_rules=10, p=0.1, n_stumps=10, self_complemented=True,
                 estimators_generator="Stumps", max_depth=1, **kwargs):
        super(SCMPregen, self).__init__(
            random_state=random_state,
            model_type=model_type,
            max_rules=max_rules,
            p=p
        )
        self.param_names = ["model_type", "max_rules", "p", "n_stumps",
                            "random_state", "estimators_generator", "max_depth"]
        self.distribs = [["conjunction", "disjunction"],
                         CustomRandint(low=1, high=15),
                         CustomUniform(loc=0, state=1), [n_stumps],
                         [random_state], ["Stumps", "Tree"],
                         CustomRandint(low=1, high=5)]
        self.classed_params = []
        self.weird_strings = {}
        self.self_complemented = self_complemented
        self.n_stumps = n_stumps
        self.estimators_generator = estimators_generator
        self.max_depth=1

    def get_params(self, deep=True):
        """

        Parameters
        ----------
        deep : boolean  (default : True) not used

        Returns
        -------
        parameters dictionary
        """
        params = super(SCMPregen, self).get_params(deep)
        params["estimators_generator"] = self.estimators_generator
        params["max_depth"] = self.max_depth
        params["n_stumps"] = self.n_stumps
        return params

    def fit(self, X, y, tiebreaker=None, iteration_callback=None,
            **fit_params):
        """
        fit function

        Parameters
        ----------
        X {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        y : { array-like, shape (n_samples,)
            Target values class labels in classification

        tiebreaker

        iteration_callback : (default : None)

        fit_params : others parameters

        Returns
        -------
        self : object
            Returns self.
        """
        pregen_X, _ = self.pregen_voters(X, y)
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x" + str(a) + ".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x" + str(a) + ".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        super(SCMPregen, self).fit(place_holder, y, tiebreaker=tiebreaker,
                                   iteration_callback=iteration_callback,
                                   **fit_params)
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        pregen_X, _ = self.pregen_voters(X)
        list_files = os.listdir(".")
        a = int(self.random_state.randint(0, 10000))
        if "pregen_x" + str(a) + ".csv" in list_files:
            a = int(np.random.randint(0, 10000))
            file_name = "pregen_x" + str(a) + ".csv"
            while file_name in list_files:
                a = int(np.random.randint(0, 10000))
                file_name = "pregen_x" + str(a) + ".csv"
        else:
            file_name = "pregen_x" + str(a) + ".csv"
        np.savetxt(file_name, pregen_X, delimiter=',')
        place_holder = np.genfromtxt(file_name, delimiter=',')
        os.remove(file_name)
        return self.classes_[self.model_.predict(place_holder)]

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #     Returns
    #     -------
    #     False in any case
    #     """
    #
    #     return False

    def getInterpret(self, directory, y_test):
        """

        Parameters
        ----------
        directory
        y_test

        Returns
        -------
        interpret_string string of interpretation
        """
        interpret_string = "Model used : " + str(self.model_)
        return interpret_string


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"model_type": args.SCP_model_type,
#                   "p": args.SCP_p,
#                   "max_rules": args.SCP_max_rules,
#                   "n_stumps": args.SCP_stumps}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append(
            {"model_type": randomState.choice(["conjunction", "disjunction"]),
             "max_rules": randomState.randint(1, 15),
             "p": randomState.random_sample()})
    return paramsSet
