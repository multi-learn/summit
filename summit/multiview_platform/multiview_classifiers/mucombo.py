# ######### COPYRIGHT #########
#
# Copyright(c) 2025
# -----------------
#
#
# * Université d'Aix Marseille (AMU) -
# * Centre National de la Recherche Scientifique (CNRS) -
# * Université de Toulon (UTLN).
# * Copyright © 2019-2025 AMU, CNRS, UTLN
#
# Contributors:
# ------------
#
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
#
# Description:
# -----------
#
#
#
# Version:
# -------
#
# * multiview_generator version = 0.0.1
#
# Licence:
# -------
#
# License: New BSD License
#
#
# ######### COPYRIGHT #########
#
from sklearn.tree import DecisionTreeClassifier


from multimodal.boosting.combo import MuComboClassifier
from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_samples_views_indices
from ..utils.base import base_boosting_estimators

classifier_class_name = "MuCombo"


class MuCombo(BaseMultiviewClassifier, MuComboClassifier):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 random_state=None,**kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        base_estimator = self.set_base_estim_from_dict(base_estimator, **kwargs)
        MuComboClassifier.__init__(self, base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    random_state=random_state,)
        self.param_names = ["base_estimator", "n_estimators", "random_state",]
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

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):
        return ""

    def set_base_estim_from_dict(self, dict):
        key, args = list(dict.items())[0]

        if key == "decision_tree":
            return DecisionTreeClassifier(**args)