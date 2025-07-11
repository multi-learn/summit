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
#
import numpy as np

from .late_fusion_utils import LateFusionClassifier
from ...utils.hyper_parameter_search import CustomRandint
from ...utils.dataset import get_samples_views_indices


class BaseJumboFusion(LateFusionClassifier):

    def __init__(self, random_state, classifiers_names=None,
                 classifier_configs=None,
                 nb_cores=1, weights=None, nb_monoview_per_view=1, rs=None):
        LateFusionClassifier.__init__(self, random_state,
                                      classifiers_names=classifiers_names,
                                      classifier_configs=classifier_configs,
                                      nb_cores=nb_cores, weights=weights,
                                      rs=rs)
        self.param_names += ["nb_monoview_per_view", ]
        self.distribs += [CustomRandint(1, 10)]
        self.nb_monoview_per_view = nb_monoview_per_view

    def set_params(self, nb_monoview_per_view=1, **params):
        self.nb_monoview_per_view = nb_monoview_per_view
        LateFusionClassifier.set_params(self, **params)

    def predict(self, X, sample_indices=None, view_indices=None):
        sample_indices, view_indices = get_samples_views_indices(X,
                                                                 sample_indices,
                                                                 view_indices)
        self._check_views(view_indices)
        monoview_decisions = self.predict_monoview(X,
                                                   sample_indices=sample_indices,
                                                   view_indices=view_indices)
        return self.aggregation_estimator.predict(monoview_decisions)

    def fit(self, X, y, train_indices=None, view_indices=None):
        train_indices, view_indices = get_samples_views_indices(X,
                                                                train_indices,
                                                                view_indices)
        self.used_views = view_indices
        self.init_classifiers(len(view_indices),
                              nb_monoview_per_view=self.nb_monoview_per_view)
        self.fit_monoview_estimators(X, y, train_indices=train_indices,
                                     view_indices=view_indices)
        monoview_decisions = self.predict_monoview(X,
                                                   sample_indices=train_indices,
                                                   view_indices=view_indices)
        self.aggregation_estimator.fit(monoview_decisions, y[train_indices])
        return self

    def fit_monoview_estimators(self, X, y, train_indices=None,
                                view_indices=None):
        if np.unique(y).shape[0] > 2:
            multiclass = True
        else:
            multiclass = False
        self.monoview_estimators = [
            [self.init_monoview_estimator(classifier_name,
                                          self.classifier_configs[
                                              classifier_index],
                                          multiclass=multiclass)
             for classifier_index, classifier_name
             in enumerate(self.classifiers_names)]
            for _ in view_indices]

        self.monoview_estimators = [[estimator.fit(
            X.get_v(view_indices[idx], train_indices), y[train_indices])
            for estimator in view_estimators]
            for idx, view_estimators in
            enumerate(self.monoview_estimators)]
        return self

    def predict_monoview(self, X, sample_indices=None, view_indices=None):
        monoview_decisions = np.zeros((len(sample_indices),
                                       len(view_indices) * len(
                                           self.classifiers_names)))
        for idx, view_estimators in enumerate(self.monoview_estimators):
            for estimator_index, estimator in enumerate(view_estimators):
                monoview_decisions[:, len(
                    self.classifiers_names) * idx + estimator_index] = estimator.predict(
                    X.get_v(view_indices[idx],
                            sample_indices))
        return monoview_decisions
