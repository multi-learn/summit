from sklearn.ensemble import RandomForestClassifier

from ..monoview.monoview_utils import CustomRandint, BaseMonoviewClassifier

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


classifier_class_name = "RandomForest"

class RandomForest(RandomForestClassifier, BaseMonoviewClassifier):
    """RandomForest Classifier Class

    Parameters
    ----------
    random_state : int seed, RandomState instance, or None (default=None)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    n_estimators : int (default : 10) number of estimators

    max_depth : int , optional (default :  None) maximum of depth

    criterion : criteria (default : 'gini')

    kwargs : others arguments


    Attributes
    ----------
    param_names :

    distribs :

    classed_params :

    weird_strings :

    """
    def __init__(self, random_state=None, n_estimators=10,
                 max_depth=None, criterion='gini', **kwargs):
        """

        Parameters
        ----------
        random_state
        n_estimators
        max_depth
        criterion
        kwargs
        """
        super(RandomForest, self).__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state
        )
        self.param_names = ["n_estimators", "max_depth", "criterion",
                            "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomRandint(low=1, high=300),
                         ["gini", "entropy"], [random_state]]
        self.weird_strings = {}

    # def canProbas(self):
    #     """Used to know if the classifier can return label probabilities
    #
    #     Returns
    #     -------
    #     True
    #     """
    #     return True

    def get_interpretation(self, directory, y_test):
        """

        Parameters
        ----------
        directory
        y_test

        Returns
        -------
        string for interpretation interpret_string
        """
        interpret_string = ""
        interpret_string += self.get_feature_importance(directory)
        return interpret_string


# def formatCmdArgs(args):
#     """Used to format kwargs for the parsed args"""
#     kwargsDict = {"n_estimators": args.RF_trees,
#                   "max_depth": args.RF_max_depth,
#                   "criterion": args.RF_criterion}
#     return kwargsDict


def paramsToSet(nIter, random_state):
    paramsSet = []
    for _ in range(nIter):
        paramsSet.append({"n_estimators": random_state.randint(1, 300),
                          "max_depth": random_state.randint(1, 300),
                          "criterion": random_state.choice(["gini", "entropy"])})
    return paramsSet
