from sklearn.tree import DecisionTreeClassifier

from ..monoview.monoview_utils import BaseMonoviewClassifier
from summit.multiview_platform.utils.hyper_parameter_search import CustomRandint

# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype

classifier_class_name = "DecisionTree"


class DecisionTree(DecisionTreeClassifier, BaseMonoviewClassifier):
    """
    This class is an adaptation of scikit-learn's `DecisionTreeClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_


    """

    def __init__(self, random_state=None, max_depth=None,
                 criterion='gini', splitter='best', **kwargs):

        DecisionTreeClassifier.__init__(self,
                                        max_depth=max_depth,
                                        criterion=criterion,
                                        splitter=splitter,
                                        random_state=random_state
                                        )
        self.param_names = ["max_depth", "criterion", "splitter",
                            'random_state']
        self.classed_params = []
        self.distribs = [CustomRandint(low=1, high=300),
                         ["gini", "entropy"],
                         ["best", "random"], [random_state]]
        self.weird_strings = {}

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multiclass=False):
        interpretString = "First feature : \n\t{} <= {}\n".format(
            feature_ids[self.tree_.feature[0]],
            self.tree_.threshold[0])
        interpretString += self.get_feature_importance(directory,
                                                       base_file_name,
                                                       feature_ids)
        return interpretString
