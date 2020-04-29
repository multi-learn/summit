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
        The seed of the pseudo random number multiview_generator to use when
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
        RandomForestClassifier.__init__(self,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        criterion=criterion,
                                        random_state=random_state
                                        )
        self.param_names = ["n_estimators", "max_depth", "criterion",
                            "random_state"]
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         CustomRandint(low=1, high=10),
                         ["gini", "entropy"], [random_state]]
        self.weird_strings = {}

    def get_interpretation(self, directory, base_file_name, y_test, multiclass=False):
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
        interpret_string += self.get_feature_importance(directory, base_file_name)
        return interpret_string
