from sklearn.linear_model import SGDClassifier

from ..monoview.monoview_utils import CustomUniform, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "SGD"

class SGD(SGDClassifier, BaseMonoviewClassifier):
    """

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    loss : str , (default = "hinge")
    penalty : str, (default = "l2")

    alpha : float, (default = 0.0001)

    kwargs : other arguments


    Attributes
    ----------
    param_names :

    distribs :

    classed_params :

    weird_strings :

    """
    def __init__(self, random_state=None, loss='hinge',
                 penalty='l2', alpha=0.0001, **kwargs):

        super(SGD, self).__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            random_state=random_state
        )
        self.param_names = ["loss", "penalty", "alpha", "random_state"]
        self.classed_params = []
        self.distribs = [['log', 'modified_huber'],
                         ["l1", "l2", "elasticnet"],
                         CustomUniform(loc=0, state=1), [random_state]]
        self.weird_strings = {}

    # def canProbas(self):
    #     """
    #     Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     return True in all case
    #     """
    #
    #     return True

    def getInterpret(self, directory, y_test):
        """

        Parameters
        ----------
        directory

        y_test

        Returns
        -------
        interpret_string str to interpreted
        """
        interpret_string = ""
        return interpret_string


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"loss": args.SGD_loss,
#                   "penalty": args.SGD_penalty,
#                   "alpha": args.SGD_alpha}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"loss": random_state.choice(['log', 'modified_huber']),
                          "penalty": random_state.choice(
                              ["l1", "l2", "elasticnet"]),
                          "alpha": random_state.random_sample()})
    return paramsSet
