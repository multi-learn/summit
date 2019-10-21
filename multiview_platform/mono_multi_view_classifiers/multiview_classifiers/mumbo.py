from sklearn.tree import DecisionTreeClassifier


from multimodalboost.mumbo import MumboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier, \
                                        get_examples_views_indices
from ..utils.hyper_parameter_search import CustomRandint

classifier_class_name = "Mumbo"

class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge"):
        super().__init__(random_state)
        super(BaseMultiviewClassifier, self).__init__(base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,
                                    best_view_mode=best_view_mode)
        self.param_names = ["base_estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [[DecisionTreeClassifier(max_depth=1)],
                         CustomRandint(5,200), [random_state], ["edge", "error"]]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_examples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=train_indices,
                                                view_indices=view_indices)
        return super(Mumbo, self).fit(numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, example_indices=None, view_indices=None):
        example_indices, view_indices = get_examples_views_indices(X,
                                                                 example_indices,
                                                                 view_indices)
        numpy_X, view_limits = X.to_numpy_array(example_indices=example_indices,
                                                view_indices=view_indices)
        return super(Mumbo, self).predict(numpy_X)
