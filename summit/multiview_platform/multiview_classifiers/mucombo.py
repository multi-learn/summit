from sklearn.tree import DecisionTreeClassifier


from multimodal.boosting.combo import MuComboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_samples_views_indices
from ..utils.base import base_boosting_estimators

classifier_class_name = "MuCumbo"


class MuCombo(BaseMultiviewClassifier, MuComboClassifier):

    def __init__(self, estimator=None,
                 n_estimators=50,
                 random_state=None,**kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        estimator = self.set_base_estim_from_dict(estimator, **kwargs)
        MuComboClassifier.__init__(self, estimator=estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,)
        self.param_names = ["estimator", "n_estimators", "random_state",]
        self.distribs = [base_boosting_estimators,
                         CustomRandint(5,200), [random_state],]

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_samples_views_indices(X,
                                                                 train_indices,
                                                                 view_indices)
        self.used_views = view_indices
        numpy_X, view_limits = X.to_numpy_array(sample_indices=train_indices,
                                                view_indices=view_indices)
        return MuComboClassifier.fit(self, numpy_X, y[train_indices],
                                                view_limits)

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        numpy_X, view_limits = X.to_numpy_array(sample_indices=sample_indices,
                                                view_indices=view_indices)
        return MuComboClassifier.predict(self, numpy_X)

    def get_interpretation(self, directory, base_file_name, labels,
                           multiclass=False):
        return ""
