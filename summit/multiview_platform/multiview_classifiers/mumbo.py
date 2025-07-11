# -*- coding: utf-8 -*-
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
# * Baptiste Bauvin <baptiste.bauvin_AT_univ-amu.fr>
# * Sokol Koço <sokol.koco_AT_lis-lab.fr>
# * Cécile Capponi <cecile.capponi_AT_univ-amu.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
#
# Description:
# -----------
#
# Supervised MultiModal Integration Tool's Readme
# This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.
#
# Version:
# -------
#
# * summit-multi-learn version = 0.0.2
#
# Licence:
# -------
#
# License: New BSD License : BSD-3-Clause
#
#
# ######### COPYRIGHT #########
#
#
#

from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os

from multimodal.boosting.mumbo import MumboClassifier

from ..multiview.multiview_utils import BaseMultiviewClassifier
from ..utils.hyper_parameter_search import CustomRandint
from ..utils.dataset import get_samples_views_indices
from ..utils.base import base_boosting_estimators
from ..utils.organization import secure_file_path
from .. import monoview_classifiers

classifier_class_name = "Mumbo"


class Mumbo(BaseMultiviewClassifier, MumboClassifier):

    def __init__(self, estimator=None,
                 n_estimators=50,
                 random_state=None,
                 best_view_mode="edge", **kwargs):
        BaseMultiviewClassifier.__init__(self, random_state)
        estimator = self.set_base_estim_from_dict(estimator)
        MumboClassifier.__init__(self, estimator=estimator,
                                 n_estimators=n_estimators,
                                 random_state=random_state,
                                 best_view_mode=best_view_mode)
        self.param_names = ["estimator", "n_estimators", "random_state", "best_view_mode"]
        self.distribs = [base_boosting_estimators,
                         CustomRandint(5, 200), [random_state], ["edge", "error"]]

    def set_params(self, estimator=None, **params):
        """
        Sets the estimator from a dict.
        :param estimator:
        :param params:
        :return:
        """
        if estimator is None:
            self.estimator = DecisionTreeClassifier()
        elif isinstance(estimator, dict):
            self.estimator = self.set_base_estim_from_dict(estimator)
            MumboClassifier.set_params(self, **params)
        else:
            MumboClassifier.set_params(self, estimator=estimator, **params)

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_samples_views_indices(X,
                                                                train_indices,
                                                                view_indices)
        self.used_views = view_indices
        self.view_names = [X.get_view_name(view_index)
                           for view_index in view_indices]
        numpy_X, view_limits = X.to_numpy_array(sample_indices=train_indices,
                                                view_indices=view_indices)
        self.view_shapes = [view_limits[ind + 1] - view_limits[ind]
                            for ind in range(len(self.used_views))]

        return MumboClassifier.fit(self, numpy_X, y[train_indices],
                                   view_limits)

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        numpy_X, view_limits = X.to_numpy_array(sample_indices=sample_indices,
                                                view_indices=view_indices)
        return MumboClassifier.predict(self, numpy_X)

    def get_interpretation(self, directory, base_file_name, y_test, feature_ids,
                           multi_class=False):
        self.view_importances = np.zeros(len(self.used_views))
        self.feature_importances_ = [np.zeros(view_shape)
                                     for view_shape in self.view_shapes]
        for best_view, estimator_weight, estimator in zip(self.best_views_, self.estimator_weights_, self.estimators_):
            self.view_importances[best_view] += estimator_weight
            if hasattr(estimator, "feature_importances_"):
                self.feature_importances_[best_view] += estimator.feature_importances_
        importances_sum = sum([np.sum(feature_importances)
                               for feature_importances
                               in self.feature_importances_])
        self.feature_importances_ = [feature_importances / importances_sum
                                     for feature_importances
                                     in self.feature_importances_]
        for feature_importances, view_name in zip(self.feature_importances_, self.view_names):
            secure_file_path(os.path.join(directory, "feature_importances",
                                          base_file_name + view_name + "-feature_importances.csv"))
            np.savetxt(os.path.join(directory, "feature_importances",
                                    base_file_name + view_name + "-feature_importances.csv"),
                       feature_importances, delimiter=',')
        # CHANGE: Making self.feature_importances_ one array, so he can be easy to use in
        # summit.multiview_platform.result_analysis.feature_importances.get_feature_importances
        self.feature_importances_ = np.concatenate(self.feature_importances_)
        self.view_importances /= np.sum(self.view_importances)
        np.savetxt(os.path.join(directory, base_file_name + "view_importances.csv"), self.view_importances,
                   delimiter=',')

        sorted_view_indices = np.argsort(-self.view_importances)
        interpret_string = "Mumbo used {} iterations to converge.".format(self.best_views_.shape[0])
        interpret_string += "\n\nViews importance : \n"
        for view_index in sorted_view_indices:
            interpret_string += "- View {} ({}), importance {}\n".format(view_index,
                                                                         self.view_names[view_index],
                                                                         self.view_importances[view_index])
        interpret_string += "\n The boosting process selected views : \n" + ", ".join(map(str, self.best_views_))
        interpret_string += "\n\n With estimator weights : \n" + "\n".join(
            map(str, self.estimator_weights_ / np.sum(self.estimator_weights_)))
        return interpret_string

    def set_base_estim_from_dict(self, dict):
        key, args = list(dict.items())[0]
        if key == "decision_tree":
            return DecisionTreeClassifier(**args)
