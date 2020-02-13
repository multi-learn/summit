import numpy as np
from sklearn.linear_model import Lasso as LassoSK

from ..monoview.monoview_utils import CustomRandint, CustomUniform, \
    BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "Lasso"

class Lasso(LassoSK, BaseMonoviewClassifier):
    """

    Parameters
    ----------
    random_state :

    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` is with the Lasso object is
        not advised
        and you should prefer the LinearRegression object. (default( : 10)

    max_iter :  int The maximum number of iterations (default : 10)

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    kwargs : others arguments

    Attributes
    ----------
    param_name :

    classed_params :

    distribs :

    weird_strings :

    """
    def __init__(self, random_state=None, alpha=1.0,
                 max_iter=10, warm_start=False, **kwargs):
        super(Lasso, self).__init__(
            alpha=alpha,
            max_iter=max_iter,
            warm_start=warm_start,
            random_state=random_state
        )
        self.param_names = ["max_iter", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomUniform(), [random_state]]
        self.weird_strings = {}

    def fit(self, X, y, check_input=True):
        neg_y = np.copy(y)
        neg_y[np.where(neg_y == 0)] = -1
        super(Lasso, self).fit(X, neg_y)
        # self.feature_importances_ = self.coef_/np.sum(self.coef_)
        return self

    def predict(self, X):
        prediction = super(Lasso, self).predict(X)
        signed = np.sign(prediction)
        signed[np.where(signed == -1)] = 0
        return signed

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     False
    #     """
    #     return False

    def get_interpretation(self, directory, y_test):
        """
        return the interpreted string

        Parameters
        ----------
        directory :

        y_test : 

        Returns
        -------
        interpreted string, str interpret_string
        """
        interpret_string = ""
        return interpret_string


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"alpha": args.LA_alpha,
#                   "max_iter": args.LA_n_iter}
#     return kwargsDict


def paramsToSet(nIter, randomState):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"max_iter": randomState.randint(1, 300),
                          "alpha": randomState.uniform(0, 1.0), })
    return paramsSet
